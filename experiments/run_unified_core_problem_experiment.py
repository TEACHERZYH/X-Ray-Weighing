from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation_protocol import PROTOCOL_VERSION
from experiment_versioning import write_run_metadata
from run_monte_carlo_robustness import simulate_rows
from run_monte_carlo_strong_baselines import make_frame


@dataclass(frozen=True)
class Scenario:
    name: str
    hidden_coal_gain: float
    env_drift_gain: float
    install_shift_gain: float


SCENARIOS = [
    Scenario("nominal", 0.0, 0.0, 0.0),
    Scenario("hidden_coal_high", 0.055, 0.0, 0.0),
    Scenario("env_drift_high", 0.0, 0.075, 0.0),
    Scenario("install_shift_high", 0.0, 0.0, 0.065),
    Scenario("combined_high", 0.045, 0.060, 0.055),
]


def parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def apply_scenario(base: pd.DataFrame, scenario: Scenario, seed: int) -> pd.DataFrame:
    df = base.copy()
    rng = np.random.default_rng(seed)
    install_keys = list(df.groupby(["site_id", "device_id", "install_id"], sort=False).groups.keys())
    install_factor = {key: rng.normal(0.0, 1.0) for key in install_keys}

    rx = df["rx_value"].to_numpy(float)
    hum = df["env_humidity_pct"].to_numpy(float)
    temp = df["env_temp_c"].to_numpy(float)
    ash = df["ash_pct"].to_numpy(float)
    t = df["timestamp_s"].to_numpy(float)

    coal_hidden = scenario.hidden_coal_gain * (
        0.60 * np.sin(t / 230.0 + seed % 17)
        + 0.025 * (ash - np.nanmean(ash))
        + rng.normal(0.0, 0.25, len(df))
    )
    env_hidden = scenario.env_drift_gain * (
        0.018 * (hum - np.nanmean(hum))
        - 0.012 * (temp - np.nanmean(temp))
        + np.sin(t / 900.0)
    )
    install_hidden = np.zeros(len(df), dtype=float)
    for key, idx in df.groupby(["site_id", "device_id", "install_id"], sort=False).groups.items():
        install_hidden[np.asarray(idx, dtype=int)] = scenario.install_shift_gain * install_factor[key]

    multiplier = np.clip(1.0 + coal_hidden + env_hidden + install_hidden, 0.45, 1.65)
    df["rx_value"] = np.maximum(rx * multiplier + rng.normal(0.0, 8.0 * max(scenario.hidden_coal_gain, scenario.env_drift_gain, 0.01), len(df)), 1.0)

    if scenario.name in {"hidden_coal_high", "combined_high"}:
        p = 1.0 / (1.0 + np.exp(-0.10 * (hum - 55.0) + 0.11 * (ash - np.nanmean(ash))))
        missing = rng.random(len(df)) < np.clip(0.35 + 0.40 * p, 0.15, 0.90)
        coal_mean = df.groupby("coal_type")["ash_pct"].transform("mean").to_numpy(float)
        df["ash_pct"] = np.where(missing, coal_mean + rng.normal(0.0, 1.0, len(df)), df["ash_pct"].to_numpy(float))
        df["ash_missing"] = missing.astype(int)

    return recompute_features(df)


def recompute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["site_id", "device_id", "install_id", "timestamp_s"]).copy()
    frames: list[pd.DataFrame] = []
    for _, g0 in out.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = g0.copy()
        rx = g["rx_value"].astype(float)
        empty = g["empty_flag"].astype(int).eq(1)
        if empty.any():
            i0_seed = float(rx[empty].quantile(0.95))
        else:
            i0_seed = float(rx.quantile(0.98))
        i0 = []
        current = max(i0_seed, 1.0)
        for value, is_empty in zip(rx.to_numpy(float), empty.to_numpy(bool)):
            if is_empty:
                current = 0.995 * current + 0.005 * value
            i0.append(max(current, value + 1e-3, 1.0))
        g["i0_online"] = i0
        g["attenuation"] = np.log(np.maximum(g["i0_online"].to_numpy(float), 1.0) / np.maximum(rx.to_numpy(float), 1.0))
        for col, source in [
            ("attenuation_mean30", "attenuation"),
            ("attenuation_std30", "attenuation"),
            ("rx_mean30", "rx_value"),
            ("rx_std30", "rx_value"),
        ]:
            roll = g[source].rolling(30, min_periods=3)
            if col.endswith("std30"):
                g[col] = roll.std().fillna(0.0)
            else:
                g[col] = roll.mean().bfill().ffill()
        g["phys_flow_feature"] = g["attenuation"] * g["belt_speed_mps"]
        frames.append(g)
    return pd.concat(frames, axis=0).reset_index(drop=True)


def features(df: pd.DataFrame, *, physics: bool, temporal: bool, missing: bool = True) -> pd.DataFrame:
    cols = [
        "tube_voltage_kv",
        "tube_current_ma",
        "source_temp_c",
        "env_temp_c",
        "env_humidity_pct",
        "belt_speed_mps",
        "ash_pct",
    ]
    if missing:
        cols.append("ash_missing")
    if physics:
        cols += ["attenuation", "i0_online", "phys_flow_feature"]
    if temporal:
        cols += ["attenuation_mean30", "attenuation_std30", "rx_mean30", "rx_std30"]
    x = df[cols].copy()
    coal = pd.get_dummies(df["coal_type"], prefix="coal", dtype=float)
    return pd.concat([x, coal], axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.maximum(y_pred, 0.0)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "cum_err_pct": float(abs(pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-9) * 100.0),
    }


def install_errors(test: pd.DataFrame, pred: np.ndarray) -> list[float]:
    df = test.copy()
    df["pred"] = np.maximum(pred, 0.0)
    out: list[float] = []
    for _, g in df.groupby(["site_id", "device_id", "install_id"], sort=False):
        y = g["true_flow_kgps"].to_numpy(float)
        p = g["pred"].to_numpy(float)
        out.append(float(abs(p.sum() - y.sum()) / max(y.sum(), 1e-9) * 100.0))
    return out


def select_prediction_stratified(pred: np.ndarray, candidates: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0 or len(candidates) == 0:
        return np.array([], dtype=int)
    n = min(int(budget), len(candidates))
    p = pred[candidates]
    order: list[int] = []
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, min(n, 10) + 1)))
    if len(edges) > 2:
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = candidates[(p >= lo) & (p <= hi)]
            if len(idx):
                center = 0.5 * (lo + hi)
                order.append(int(idx[np.argmin(np.abs(pred[idx] - center))]))
    for idx in candidates[np.argsort(p)]:
        value = int(idx)
        if value not in order:
            order.append(value)
        if len(order) >= n:
            break
    return np.array(order[:n], dtype=int)


def calibrate_linear(pred: np.ndarray, y_true: np.ndarray, sample_idx: np.ndarray) -> np.ndarray:
    if len(sample_idx) < 2:
        return np.maximum(pred, 0.0)
    x = pred[sample_idx]
    y = y_true[sample_idx]
    a, b = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    return np.maximum(a * pred + b, 0.0)


def calibrate_mass_anchor(pred: np.ndarray, y_true: np.ndarray, sample_idx: np.ndarray, bound_pct: float) -> np.ndarray:
    linear = calibrate_linear(pred, y_true, sample_idx)
    if len(sample_idx) < 2:
        return linear
    factor = y_true[sample_idx].sum() / max(linear[sample_idx].sum(), 1e-9)
    bound = bound_pct / 100.0
    factor = float(np.clip(factor, 1.0 - bound, 1.0 + bound))
    return np.maximum(linear * factor, 0.0)


def calibrate_installwise(
    test: pd.DataFrame,
    pred: np.ndarray,
    y_true: np.ndarray,
    *,
    budget: int,
    mass_anchor: bool,
    bound_pct: float,
    adapt_fraction: float,
) -> np.ndarray:
    out = np.maximum(pred.copy(), 0.0)
    for _, g in test.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = g.sort_values("timestamp_s", kind="mergesort")
        idx = g.index.to_numpy(dtype=int)
        cut = max(1, min(len(idx) - 1, int(np.floor(len(idx) * adapt_fraction))))
        adapt_idx = idx[:cut]
        candidates = adapt_idx[pred[adapt_idx] > 5.0]
        sample_idx = select_prediction_stratified(pred, candidates, budget)
        if len(sample_idx) < 2:
            continue
        x = pred[sample_idx]
        y = y_true[sample_idx]
        a, b = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
        install_pred = np.maximum(a * pred[idx] + b, 0.0)
        if mass_anchor:
            local_sample = np.searchsorted(idx, sample_idx)
            valid = (local_sample >= 0) & (local_sample < len(idx)) & (idx[local_sample] == sample_idx)
            local_sample = local_sample[valid]
            if len(local_sample) >= 2:
                factor = y_true[sample_idx].sum() / max(install_pred[local_sample].sum(), 1e-9)
                bound = bound_pct / 100.0
                factor = float(np.clip(factor, 1.0 - bound, 1.0 + bound))
                install_pred = np.maximum(install_pred * factor, 0.0)
        out[idx] = install_pred
    return out


def evaluation_indices(test: pd.DataFrame, adapt_fraction: float) -> np.ndarray:
    indices: list[int] = []
    for _, group in test.groupby(["site_id", "device_id", "install_id"], sort=False):
        ordered = group.sort_values("timestamp_s", kind="mergesort").index.to_numpy(dtype=int)
        cut = max(1, min(len(ordered) - 1, int(np.floor(len(ordered) * adapt_fraction))))
        indices.extend(ordered[cut:].tolist())
    return np.asarray(sorted(indices), dtype=int)


def add_result(
    rows: list[dict[str, object]],
    *,
    scenario: str,
    replicate: int,
    heldout_site: str,
    method: str,
    budget: int,
    y_true: np.ndarray,
    pred: np.ndarray,
    test: pd.DataFrame,
) -> None:
    inst = install_errors(test, pred)
    m = metrics(y_true, pred)
    rows.append(
        {
            "scenario": scenario,
            "replicate": replicate,
            "heldout_site": heldout_site,
            "method": method,
            "label_budget_per_install": budget,
            **m,
            "install_cum_err_median_pct": float(np.median(inst)) if inst else 0.0,
            "install_cum_err_p90_pct": float(np.quantile(inst, 0.90)) if inst else 0.0,
            "install_success_lt1_pct": float(np.mean(np.array(inst) < 1.0) * 100.0) if inst else 0.0,
            "test_rows": int(len(test)),
            "test_ash_missing_ratio": float(test["ash_missing"].astype(float).mean()),
        }
    )


def evaluate_scenario(
    df: pd.DataFrame,
    *,
    scenario: str,
    replicate: int,
    heldout_site: str,
    budgets: list[int],
    seed: int,
    estimator: str,
    hgb_max_iter: int,
    mass_anchor_bound_pct: float,
    adapt_fraction: float,
) -> list[dict[str, object]]:
    train = df[df["site_id"].ne(heldout_site)].copy().reset_index(drop=True)
    test = df[df["site_id"].eq(heldout_site)].copy().reset_index(drop=True)
    y_train = train["true_flow_kgps"].to_numpy(float)
    y_test = test["true_flow_kgps"].to_numpy(float)
    eval_idx = evaluation_indices(test, adapt_fraction)
    eval_test = test.iloc[eval_idx].reset_index(drop=True)

    rows: list[dict[str, object]] = []
    suffix = "HGB" if estimator == "hgb" else "Ridge"
    model_specs = [
        (f"DataOnly-{suffix}", False, False),
        (f"Physics-Temporal-{suffix}", True, True),
    ]
    predictions: dict[str, np.ndarray] = {}
    for model_idx, (name, physics, temporal) in enumerate(model_specs):
        x_train = features(train, physics=physics, temporal=temporal)
        x_test = features(test, physics=physics, temporal=temporal).reindex(columns=x_train.columns, fill_value=0.0)
        if estimator == "hgb":
            model = HistGradientBoostingRegressor(
                max_iter=hgb_max_iter,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=0.01,
                random_state=seed + model_idx,
            )
        elif estimator == "ridge":
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        else:
            raise ValueError(f"unknown estimator: {estimator}")
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        predictions[name] = pred
        add_result(
            rows,
            scenario=scenario,
            replicate=replicate,
            heldout_site=heldout_site,
            method=name,
            budget=0,
            y_true=y_test[eval_idx],
            pred=pred[eval_idx],
            test=eval_test,
        )

    base = predictions[f"Physics-Temporal-{suffix}"]
    for budget in budgets:
        add_result(
            rows,
            scenario=scenario,
            replicate=replicate,
            heldout_site=heldout_site,
            method=f"PG-TSC-lite-{suffix} calibrate={budget}",
            budget=budget,
            y_true=y_test[eval_idx],
            pred=calibrate_installwise(
                test,
                base,
                y_test,
                budget=budget,
                mass_anchor=False,
                bound_pct=mass_anchor_bound_pct,
                adapt_fraction=adapt_fraction,
            )[eval_idx],
            test=eval_test,
        )
        add_result(
            rows,
            scenario=scenario,
            replicate=replicate,
            heldout_site=heldout_site,
            method=f"PG-TSC-lite-{suffix} mass-anchor={budget}",
            budget=budget,
            y_true=y_test[eval_idx],
            pred=calibrate_installwise(
                test,
                base,
                y_test,
                budget=budget,
                mass_anchor=True,
                bound_pct=mass_anchor_bound_pct,
                adapt_fraction=adapt_fraction,
            )[eval_idx],
            test=eval_test,
        )
    for row in rows:
        row["protocol_version"] = PROTOCOL_VERSION
        row["adapt_fraction"] = adapt_fraction
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    df = pd.DataFrame(rows)
    base = df[df["method"].astype(str).str.startswith("Physics-Temporal-")][
        ["scenario", "replicate", "heldout_site", "cum_err_pct"]
    ]
    base = base.rename(columns={"cum_err_pct": "base_cum_err_pct"})
    merged = df.merge(base, on=["scenario", "replicate", "heldout_site"], how="left")
    merged["cum_err_recovery_pct"] = (
        (merged["base_cum_err_pct"] - merged["cum_err_pct"]) / merged["base_cum_err_pct"].clip(lower=1e-9) * 100.0
    )
    summary_rows: list[dict[str, object]] = []
    for (scenario, method, budget), g in merged.groupby(["scenario", "method", "label_budget_per_install"], sort=False):
        summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "label_budget_per_install": int(budget),
                "splits": int(len(g)),
                "mae_median": float(g["mae"].median()),
                "cum_err_median_pct": float(g["cum_err_pct"].median()),
                "cum_err_p90_pct": float(g["cum_err_pct"].quantile(0.90)),
                "install_cum_err_median_pct": float(g["install_cum_err_median_pct"].median()),
                "install_cum_err_p90_pct": float(g["install_cum_err_p90_pct"].median()),
                "success_lt1_pct": float((g["cum_err_pct"] < 1.0).mean() * 100.0),
                "install_success_lt1_median_pct": float(g["install_success_lt1_pct"].median()),
                "recovery_median_pct": float(g["cum_err_recovery_pct"].median()),
                "ash_missing_ratio_mean": float(g["test_ash_missing_ratio"].mean()),
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified core-problem simulation experiment for PG-TSC.")
    parser.add_argument("--output", default="results/sim_unified_core_problem_v2.csv")
    parser.add_argument("--summary", default="results/sim_unified_core_problem_summary_v2.csv")
    parser.add_argument("--replicates", type=int, default=6)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--devices-per-site", type=int, default=2)
    parser.add_argument("--installs-per-device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--budgets", default="20,60")
    parser.add_argument("--estimator", default="ridge", choices=["ridge", "hgb"])
    parser.add_argument("--hgb-max-iter", type=int, default=120)
    parser.add_argument("--mass-anchor-bound-pct", type=float, default=5.0)
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    args = parser.parse_args()

    budgets = parse_ints(args.budgets)
    all_rows: list[dict[str, object]] = []
    for replicate in range(args.replicates):
        seed = args.seed + replicate * 1000
        rows = simulate_rows(
            seed=seed,
            sites=args.sites,
            devices_per_site=args.devices_per_site,
            installs_per_device=args.installs_per_device,
            seconds=args.seconds,
        )
        base = make_frame(rows)
        for scenario_idx, scenario in enumerate(SCENARIOS):
            df = apply_scenario(base, scenario, seed + scenario_idx * 97)
            for site_idx in range(1, args.sites + 1):
                all_rows.extend(
                    evaluate_scenario(
                        df,
                        scenario=scenario.name,
                        replicate=replicate,
                        heldout_site=f"S{site_idx:02d}",
                        budgets=budgets,
                        seed=seed + scenario_idx * 101 + site_idx,
                        estimator=args.estimator,
                        hgb_max_iter=args.hgb_max_iter,
                        mass_anchor_bound_pct=args.mass_anchor_bound_pct,
                        adapt_fraction=args.adapt_fraction,
                    )
                )
            print(
                f"replicate={replicate + 1}/{args.replicates} scenario={scenario.name} rows={len(all_rows)}",
                flush=True,
            )

    output = Path(args.output)
    summary = Path(args.summary)
    summary_rows = summarize(all_rows)
    write_csv(output, all_rows)
    write_csv(summary, summary_rows)
    meta = write_run_metadata(
        output,
        script="experiments/run_unified_core_problem_experiment.py",
        args=vars(args),
        inputs=[],
        outputs=[output, summary],
        notes=(
            "Unified simulation matrix using causal target adaptation and independent "
            f"evaluation under protocol {PROTOCOL_VERSION}."
        ),
    )
    summary_meta = write_run_metadata(
        summary,
        script="experiments/run_unified_core_problem_experiment.py",
        args=vars(args),
        inputs=[],
        outputs=[output, summary],
        notes="Summary table for the unified core-problem simulation matrix.",
    )
    print(f"wrote={output}")
    print(f"wrote={summary}")
    print(f"wrote={meta}")
    print(f"wrote={summary_meta}")
    print(f"detail_rows={len(all_rows)}")
    print(f"summary_rows={len(summary_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
