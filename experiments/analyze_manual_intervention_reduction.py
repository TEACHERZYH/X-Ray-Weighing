from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent))

from experiment_versioning import write_run_metadata  # noqa: E402
from run_pseudo_real_experiments import TEST_SITES, TRAIN_SITES, fit_models, y_array  # noqa: E402
from run_sim_baselines import add_online_features, metrics, read_rows  # noqa: E402


RISK_FEATURES = [
    "rx_value",
    "attenuation",
    "tube_voltage_kv",
    "tube_current_ma",
    "source_temp_c",
    "env_temp_c",
    "env_humidity_pct",
    "belt_speed_mps",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate manual intervention reduction on pseudo-real X-ray data.")
    parser.add_argument("--input", type=Path, default=Path("data/real_like/pseudo_real_xray_belt.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/pseudo_real_manual_intervention_reduction.csv"))
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--initial-budget", type=int, default=20)
    parser.add_argument("--extra-budgets", default="10,20,60,180")
    parser.add_argument("--adapt-fraction", type=float, default=0.30)
    parser.add_argument("--maintenance-end-fraction", type=float, default=0.65)
    parser.add_argument("--anchor-bound-pct", type=float, default=5.0)
    return parser.parse_args()


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def group_by_install(rows: list[dict[str, object]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = f"{row['site_id']}|{row['device_id']}|{row['install_id']}"
        groups[key].append(idx)
    return groups


def choose_time_uniform(indices: list[int], budget: int) -> list[int]:
    if budget <= 0 or not indices:
        return []
    n = min(int(budget), len(indices))
    pos = np.linspace(0, len(indices) - 1, n).round().astype(int)
    return sorted({indices[int(i)] for i in pos})


def fit_linear(pred: np.ndarray, y_true: np.ndarray, sample_idxs: list[int]) -> tuple[float, float]:
    if len(sample_idxs) < 2:
        return 1.0, 0.0
    x = pred[sample_idxs]
    y = y_true[sample_idxs]
    a, b = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    return float(a), float(b)


def feature_distance(rows: list[dict[str, object]], reference: list[int], target: list[int]) -> np.ndarray:
    if not reference or not target:
        return np.zeros(len(target), dtype=float)
    ref = np.asarray([[float(rows[i][col]) for col in RISK_FEATURES] for i in reference], dtype=float)
    tar = np.asarray([[float(rows[i][col]) for col in RISK_FEATURES] for i in target], dtype=float)
    center = np.median(ref, axis=0)
    q75 = np.quantile(ref, 0.75, axis=0)
    q25 = np.quantile(ref, 0.25, axis=0)
    scale = q75 - q25
    scale[scale <= 1e-9] = np.std(ref[:, scale <= 1e-9], axis=0) if np.any(scale <= 1e-9) else 1.0
    scale = np.where(scale <= 1e-9, 1.0, scale)
    z = (tar - center) / scale
    return np.sqrt(np.sum(z * z, axis=1))


def risk_scores(rows: list[dict[str, object]], base_pred: np.ndarray, calibrated: np.ndarray, initial_sample: list[int], pool: list[int]) -> np.ndarray:
    if not pool:
        return np.array([], dtype=float)
    sample_values = calibrated[initial_sample] if initial_sample else calibrated[pool]
    lo = float(np.min(sample_values))
    hi = float(np.max(sample_values))
    span = max(hi - lo, 1e-6)
    pool_cal = calibrated[pool]
    edge = np.maximum((lo - pool_cal) / span, (pool_cal - hi) / span)
    edge = np.maximum(edge, 0.0)
    correction = np.abs(calibrated[pool] - base_pred[pool])
    correction = correction / max(float(np.quantile(correction, 0.90)), 1e-6)
    dist = feature_distance(rows, initial_sample, pool)
    dist = dist / max(float(np.quantile(dist, 0.90)), 1e-6)
    return edge + 0.35 * correction + 0.35 * dist


def select_risk_top(pool: list[int], scores: np.ndarray, budget: int) -> list[int]:
    if budget <= 0 or not pool:
        return []
    n = min(int(budget), len(pool))
    order = np.argsort(scores)[-n:]
    return sorted(pool[int(i)] for i in order)


def select_risk_covered(pool: list[int], pred: np.ndarray, scores: np.ndarray, budget: int) -> list[int]:
    if budget <= 0 or not pool:
        return []
    n = min(int(budget), len(pool))
    pred_values = pred[pool]
    edges = np.unique(np.quantile(pred_values, np.linspace(0.0, 1.0, min(n, 8) + 1)))
    selected: list[int] = []
    if len(edges) > 2:
        for lo, hi in zip(edges[:-1], edges[1:]):
            candidates = [j for j, idx in enumerate(pool) if lo <= pred[idx] <= hi]
            if candidates:
                selected.append(pool[int(max(candidates, key=lambda j: scores[j]))])
    selected_set = set(selected)
    for j in np.argsort(scores)[::-1]:
        idx = pool[int(j)]
        if idx not in selected_set:
            selected.append(idx)
            selected_set.add(idx)
        if len(selected) >= n:
            break
    return sorted(selected[:n])


def apply_anchor(calibrated: np.ndarray, y_true: np.ndarray, extra_sample: list[int], final_idxs: list[int], bound_pct: float) -> np.ndarray:
    adjusted = calibrated.copy()
    if len(extra_sample) < 2:
        return adjusted
    observed = float(np.sum(y_true[extra_sample]))
    estimated = max(float(np.sum(calibrated[extra_sample])), 1e-9)
    factor = observed / estimated
    bound = float(bound_pct) / 100.0
    factor = float(np.clip(factor, 1.0 - bound, 1.0 + bound))
    adjusted[final_idxs] = np.maximum(calibrated[final_idxs] * factor, 0.0)
    return adjusted


def install_error_stats(rows: list[dict[str, object]], y_true: np.ndarray, pred: np.ndarray, final_indices: list[int]) -> tuple[float, float, float]:
    final_set = set(final_indices)
    values: list[float] = []
    for idxs in group_by_install(rows).values():
        selected = [i for i in idxs if i in final_set]
        if not selected:
            continue
        denom = max(abs(float(np.sum(y_true[selected]))), 1e-9)
        values.append(abs(float(np.sum(pred[selected]) - np.sum(y_true[selected]))) / denom * 100.0)
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.asarray(values, dtype=float)
    return float(np.median(arr)), float(np.max(arr)), float(np.mean(arr < 1.0) * 100.0)


def evaluate_policy(
    method: str,
    rows: list[dict[str, object]],
    y_true: np.ndarray,
    pred: np.ndarray,
    final_indices: list[int],
    install_count: int,
    initial_budget: int,
    extra_budget: int,
    scheduled_180_labels: int,
) -> dict[str, object]:
    row = {"method": method, "extra_budget": extra_budget, "n": len(final_indices), **metrics(y_true[final_indices], pred[final_indices])}
    median_err, max_err, pass_rate = install_error_stats(rows, y_true, pred, final_indices)
    labels = install_count * (initial_budget + extra_budget)
    row.update(
        {
            "install_count": install_count,
            "total_reference_labels": labels,
            "labels_per_install": initial_budget + extra_budget,
            "label_reduction_vs_scheduled180_pct": (1.0 - labels / max(scheduled_180_labels, 1)) * 100.0,
            "install_cum_err_median_pct": median_err,
            "install_cum_err_max_pct": max_err,
            "install_success_lt1_pct": pass_rate,
        }
    )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "extra_budget",
        "n",
        "mae",
        "rmse",
        "cum_err_pct",
        "install_count",
        "total_reference_labels",
        "labels_per_install",
        "label_reduction_vs_scheduled180_pct",
        "install_cum_err_median_pct",
        "install_cum_err_max_pct",
        "install_success_lt1_pct",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)
    add_online_features(rows)
    train_rows = [row for row in rows if str(row["site_id"]) in TRAIN_SITES]
    test_rows = [row for row in rows if str(row["site_id"]) in TEST_SITES]
    if not train_rows or not test_rows:
        raise RuntimeError("pseudo-real data must contain S01-S03 training sites and S04-S05 test sites")

    y_true = y_array(test_rows)
    base_pred = fit_models(train_rows, test_rows)["Physics-Temporal-Ridge"]
    groups = group_by_install(test_rows)
    install_count = len(groups)
    extra_budgets = parse_ints(args.extra_budgets)
    scheduled_180_labels = install_count * (args.initial_budget + 180)

    policy_preds: dict[tuple[str, int], np.ndarray] = {}
    final_indices_all: list[int] = []
    for key, idxs0 in groups.items():
        idxs = sorted(idxs0, key=lambda i: float(test_rows[i]["timestamp_s"]))
        n = len(idxs)
        adapt_cut = max(2, min(n - 2, int(round(n * args.adapt_fraction))))
        maintenance_cut = max(adapt_cut + 1, min(n - 1, int(round(n * args.maintenance_end_fraction))))
        adapt = idxs[:adapt_cut]
        maintenance = idxs[adapt_cut:maintenance_cut]
        final = idxs[maintenance_cut:]
        final_indices_all.extend(final)

        initial_sample = choose_time_uniform(adapt, args.initial_budget)
        a0, b0 = fit_linear(base_pred, y_true, initial_sample)
        initial_cal = np.maximum(a0 * base_pred + b0, 0.0)
        score = risk_scores(test_rows, base_pred, initial_cal, initial_sample, maintenance)

        for policy, budget, pred in [("install_only", 0, initial_cal)]:
            policy_preds.setdefault((policy, budget), base_pred.copy())
            policy_preds[(policy, budget)][final] = pred[final]

        for budget in extra_budgets:
            scheduled_sample = choose_time_uniform(maintenance, budget)
            risk_top_sample = select_risk_top(maintenance, score, budget)
            risk_covered_sample = select_risk_covered(maintenance, initial_cal, score, budget)
            for policy, sample in [
                ("scheduled_uniform", scheduled_sample),
                ("risk_top_refit", risk_top_sample),
            ]:
                a, b = fit_linear(base_pred, y_true, initial_sample + sample)
                adjusted = np.maximum(a * base_pred + b, 0.0)
                policy_preds.setdefault((policy, budget), base_pred.copy())
                policy_preds[(policy, budget)][final] = adjusted[final]
            anchored = apply_anchor(initial_cal, y_true, risk_covered_sample, final, args.anchor_bound_pct)
            policy_preds.setdefault(("risk_covered_anchor", budget), base_pred.copy())
            policy_preds[("risk_covered_anchor", budget)][final] = anchored[final]

    final_indices = sorted(final_indices_all)
    output_rows = []
    for (policy, budget), pred in sorted(policy_preds.items(), key=lambda x: (x[0][0], x[0][1])):
        output_rows.append(
            evaluate_policy(
                policy,
                test_rows,
                y_true,
                pred,
                final_indices,
                install_count,
                args.initial_budget,
                budget,
                scheduled_180_labels,
            )
        )

    write_csv(args.output, output_rows)
    meta = write_run_metadata(
        args.output,
        script=__file__,
        args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        inputs=[args.input],
        outputs=[args.output],
        notes="Pseudo-real manual intervention reduction experiment. Not field evidence.",
    )
    print(f"wrote={args.output}")
    print(f"wrote={meta}")
    print(f"install_count={install_count}")
    print(f"final_eval_rows={len(final_indices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
