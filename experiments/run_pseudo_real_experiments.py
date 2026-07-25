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
from evaluation_protocol import PROTOCOL_VERSION  # noqa: E402
from run_sim_baselines import (  # noqa: E402
    add_online_features,
    feature_matrix,
    fit_physics,
    metrics,
    read_rows,
    ridge_fit,
    ridge_predict,
)


TRAIN_SITES = {"S01", "S02", "S03"}
TEST_SITES = {"S04", "S05"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight experiments on pseudo-real X-ray belt data.")
    parser.add_argument("--input", type=Path, default=Path("data/real_like/pseudo_real_xray_belt.csv"))
    parser.add_argument("--summary", type=Path, default=Path("results/pseudo_real_experiment_summary_v2.csv"))
    parser.add_argument("--calibration", type=Path, default=Path("results/pseudo_real_calibration_strategy_v2.csv"))
    parser.add_argument("--stratified", type=Path, default=Path("results/pseudo_real_stratified_errors_v2.csv"))
    parser.add_argument("--install-errors", type=Path, default=Path("results/pseudo_real_install_errors_v2.csv"))
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--calibration-fraction", type=float, default=0.20)
    return parser.parse_args()


def y_array(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([float(row["true_flow_kgps"]) for row in rows], dtype=float)


def group_by_install(rows: list[dict[str, object]]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        key = f"{row['site_id']}|{row['device_id']}|{row['install_id']}"
        groups[key].append(idx)
    return groups


def eval_indices_by_install(rows: list[dict[str, object]], calibration_fraction: float) -> tuple[list[int], dict[str, list[int]], dict[str, list[int]]]:
    calibration: dict[str, list[int]] = {}
    evaluation: dict[str, list[int]] = {}
    eval_all: list[int] = []
    for key, idxs in group_by_install(rows).items():
        idxs = sorted(idxs, key=lambda i: float(rows[i]["timestamp_s"]))
        cut = max(1, min(len(idxs) - 1, int(round(len(idxs) * calibration_fraction))))
        calibration[key] = idxs[:cut]
        evaluation[key] = idxs[cut:]
        eval_all.extend(idxs[cut:])
    return sorted(eval_all), calibration, evaluation


def choose_samples(
    rows: list[dict[str, object]],
    pred: np.ndarray,
    candidate_idxs: list[int],
    budget: int,
    strategy: str,
    rng: random.Random,
) -> list[int]:
    if budget <= 0 or not candidate_idxs:
        return []
    n = min(budget, len(candidate_idxs))
    if strategy == "time_uniform":
        pos = np.linspace(0, len(candidate_idxs) - 1, n).round().astype(int)
        return sorted({candidate_idxs[int(i)] for i in pos})
    if strategy == "prediction_stratified":
        ordered = sorted(candidate_idxs, key=lambda i: float(pred[i]))
        pos = np.linspace(0, len(ordered) - 1, n).round().astype(int)
        return sorted({ordered[int(i)] for i in pos})
    if strategy == "load_stratified":
        ordered = sorted(candidate_idxs, key=lambda i: float(rows[i]["true_flow_kgps"]))
        pos = np.linspace(0, len(ordered) - 1, n).round().astype(int)
        return sorted({ordered[int(i)] for i in pos})
    if strategy == "random":
        return sorted(rng.sample(candidate_idxs, n))
    raise ValueError(f"unknown strategy: {strategy}")


def calibrate_per_install(
    rows: list[dict[str, object]],
    pred: np.ndarray,
    y_true: np.ndarray,
    calibration: dict[str, list[int]],
    evaluation: dict[str, list[int]],
    budget: int,
    strategy: str,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    adjusted = pred.copy()
    eval_all: list[int] = []
    rng = random.Random(seed)
    for key, candidate_idxs in calibration.items():
        sample_idxs = choose_samples(rows, pred, candidate_idxs, budget, strategy, rng)
        eval_idxs = evaluation[key]
        eval_all.extend(eval_idxs)
        if len(sample_idxs) < 2:
            continue
        x = pred[sample_idxs]
        y = y_true[sample_idxs]
        xb = np.column_stack([x, np.ones_like(x)])
        slope, intercept = np.linalg.lstsq(xb, y, rcond=None)[0]
        adjusted[eval_idxs] = slope * pred[eval_idxs] + intercept
    return np.maximum(adjusted, 0.0), sorted(eval_all)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_row(method: str, rows: list[dict[str, object]], y_true: np.ndarray, pred: np.ndarray, indices: list[int], extra: dict[str, object] | None = None) -> dict[str, object]:
    item = {"method": method, "n": len(indices), **metrics(y_true[indices], pred[indices])}
    if extra:
        item.update(extra)
    return item


def fit_models(train_rows: list[dict[str, object]], test_rows: list[dict[str, object]]) -> dict[str, np.ndarray]:
    y_train = y_array(train_rows)
    predictions: dict[str, np.ndarray] = {}
    k = fit_physics(train_rows)
    predictions["Physics+I0"] = np.asarray(
        [k * float(row["attenuation"]) * float(row["belt_speed_mps"]) for row in test_rows],
        dtype=float,
    )
    configs = [
        ("DataOnly-Ridge", False, True, False),
        ("Physics-Ridge", True, True, False),
        ("Physics-Temporal-Ridge", True, True, True),
        ("NoMissingFlag-Ablation", True, False, True),
    ]
    for name, include_physics, include_missing, temporal in configs:
        x_train = feature_matrix(train_rows, include_physics, include_missing, temporal)
        x_test = feature_matrix(test_rows, include_physics, include_missing, temporal)
        beta, mean, std = ridge_fit(x_train, y_train, lam=10.0)
        predictions[name] = ridge_predict(x_test, beta, mean, std)
    return predictions


def bin_rows(rows: list[dict[str, object]], indices: list[int], column: str, labels: tuple[str, str, str]) -> list[tuple[str, list[int]]]:
    values = np.asarray([float(rows[i][column]) for i in indices], dtype=float)
    q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
    bins = [(labels[0], []), (labels[1], []), (labels[2], [])]
    for i in indices:
        value = float(rows[i][column])
        if value <= q1:
            bins[0][1].append(i)
        elif value <= q2:
            bins[1][1].append(i)
        else:
            bins[2][1].append(i)
    return bins


def stratified_rows(rows: list[dict[str, object]], y_true: np.ndarray, predictions: dict[str, np.ndarray], eval_indices: list[int]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    methods = ["Physics-Temporal-Ridge", "PG-TSC-lite budget=180 strategy=time_uniform"]
    groups: list[tuple[str, str, list[int]]] = []
    for label, idxs in bin_rows(rows, eval_indices, "env_temp_c", ("low", "mid", "high")):
        groups.append(("env_temp_c", label, idxs))
    for label, idxs in bin_rows(rows, eval_indices, "env_humidity_pct", ("low", "mid", "high")):
        groups.append(("env_humidity_pct", label, idxs))
    for coal in sorted({str(rows[i]["coal_type"]) for i in eval_indices}):
        groups.append(("coal_type", coal, [i for i in eval_indices if str(rows[i]["coal_type"]) == coal]))
    for site in sorted({str(rows[i]["site_id"]) for i in eval_indices}):
        groups.append(("site_id", site, [i for i in eval_indices if str(rows[i]["site_id"]) == site]))
    for dimension, group, idxs in groups:
        for method in methods:
            if idxs:
                output.append(metric_row(method, rows, y_true, predictions[method], idxs, {"dimension": dimension, "group": group}))
    return output


def install_rows(rows: list[dict[str, object]], y_true: np.ndarray, predictions: dict[str, np.ndarray], eval_indices: list[int]) -> list[dict[str, object]]:
    eval_set = set(eval_indices)
    output: list[dict[str, object]] = []
    for key, idxs in group_by_install(rows).items():
        selected = [idx for idx in idxs if idx in eval_set]
        if not selected:
            continue
        site, device, install = key.split("|")
        for method in ["Physics-Temporal-Ridge", "PG-TSC-lite budget=180 strategy=time_uniform"]:
            output.append(
                metric_row(
                    method,
                    rows,
                    y_true,
                    predictions[method],
                    selected,
                    {"site_id": site, "device_id": device, "install_id": install},
                )
            )
    return output


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input)
    add_online_features(rows)
    train_rows = [row for row in rows if str(row["site_id"]) in TRAIN_SITES]
    test_rows = [row for row in rows if str(row["site_id"]) in TEST_SITES]
    if not train_rows or not test_rows:
        raise RuntimeError("pseudo-real data must contain train sites S01-S03 and test sites S04-S05")

    y_true = y_array(test_rows)
    predictions = fit_models(train_rows, test_rows)
    eval_indices, calibration, evaluation = eval_indices_by_install(test_rows, args.calibration_fraction)

    summary: list[dict[str, object]] = []
    for name, pred in predictions.items():
        summary.append(metric_row(name, test_rows, y_true, np.maximum(pred, 0.0), eval_indices, {"split": "S04_S05_eval"}))

    calibration_rows: list[dict[str, object]] = []
    base_pred = predictions["Physics-Temporal-Ridge"]
    for budget in [0, 5, 20, 60, 180]:
        for strategy in ["time_uniform", "prediction_stratified", "load_stratified", "random"]:
            if budget == 0 and strategy != "time_uniform":
                continue
            seed = args.seed + budget + len(strategy)
            calibrated, cal_eval_indices = calibrate_per_install(
                test_rows,
                base_pred,
                y_true,
                calibration,
                evaluation,
                budget,
                strategy,
                seed,
            )
            method = f"PG-TSC-lite budget={budget} strategy={strategy}"
            predictions[method] = calibrated
            row = metric_row(method, test_rows, y_true, calibrated, cal_eval_indices, {"budget": budget, "strategy": strategy})
            calibration_rows.append(row)
            if budget in {0, 20, 180} and strategy == "time_uniform":
                summary.append({**row, "split": "S04_S05_eval"})

    stratified = stratified_rows(test_rows, y_true, predictions, eval_indices)
    installs = install_rows(test_rows, y_true, predictions, eval_indices)

    write_csv(args.summary, summary, ["method", "split", "n", "mae", "rmse", "cum_err_pct", "budget", "strategy"])
    write_csv(args.calibration, calibration_rows, ["method", "budget", "strategy", "n", "mae", "rmse", "cum_err_pct"])
    write_csv(args.stratified, stratified, ["method", "dimension", "group", "n", "mae", "rmse", "cum_err_pct"])
    write_csv(args.install_errors, installs, ["method", "site_id", "device_id", "install_id", "n", "mae", "rmse", "cum_err_pct"])

    outputs = [args.summary, args.calibration, args.stratified, args.install_errors]
    for output in outputs:
        meta = write_run_metadata(
            output,
            script=__file__,
            args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            inputs=[args.input],
            outputs=[output],
            notes=(
                "Pseudo-real experiment output using causal I0 features and chronological "
                f"installation holdout protocol {PROTOCOL_VERSION}. Not field evidence."
            ),
        )
        print(f"wrote={output}")
        print(f"wrote={meta}")
    print(f"train_rows={len(train_rows)}")
    print(f"test_rows={len(test_rows)}")
    print(f"eval_rows={len(eval_indices)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
