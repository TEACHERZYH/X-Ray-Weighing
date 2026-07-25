from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent))

from experiment_versioning import write_run_metadata  # noqa: E402
from run_sim_baselines import add_online_features  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SITES = {"S01", "S02", "S03"}
TEST_SITES = {"S04", "S05"}
COAL_ORDER = ["anthracite", "bituminous_A", "bituminous_B", "lignite"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-data X-ray belt self-calibration experiments.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "real" / "real_xray_belt_clean_qc.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "real_xray_experiment_summary.csv")
    parser.add_argument("--calibration", type=Path, default=ROOT / "results" / "real_xray_calibration_strategy.csv")
    parser.add_argument("--stratified", type=Path, default=ROOT / "results" / "real_xray_stratified_errors.csv")
    parser.add_argument("--install-errors", type=Path, default=ROOT / "results" / "real_xray_install_errors.csv")
    parser.add_argument("--qc-holdout", type=Path, default=ROOT / "results" / "real_xray_qc_holdout_errors.csv")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--calibration-fraction", type=float, default=0.30)
    parser.add_argument("--mass-anchor-bound-pct", type=float, default=15.0)
    parser.add_argument("--max-rows", type=int, default=0, help="Connectivity-test row cap after sorting; 0 uses all rows.")
    parser.add_argument("--quick", action="store_true", help="Skip heavier nonlinear baselines.")
    return parser.parse_args()


def rows_from_frame(df: pd.DataFrame) -> list[dict[str, object]]:
    records = df.to_dict(orient="records")
    numeric_fields = {
        "timestamp_s",
        "rx_value",
        "tube_voltage_kv",
        "tube_current_ma",
        "source_temp_c",
        "env_temp_c",
        "env_humidity_pct",
        "belt_speed_mps",
        "empty_flag",
        "ash_pct",
        "ash_missing",
        "true_loading_kgpm",
        "true_flow_kgps",
        "true_cum_mass_kg",
    }
    out: list[dict[str, object]] = []
    for row in records:
        parsed: dict[str, object] = {}
        for key, value in row.items():
            parsed[key] = float(value) if key in numeric_fields else value
        out.append(parsed)
    return out


def frame_with_online_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = rows_from_frame(df)
    add_online_features(rows)
    return pd.DataFrame(rows)


def feature_frame(df: pd.DataFrame, *, physics: bool, temporal: bool, missing: bool = True) -> pd.DataFrame:
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
        df = df.copy()
        df["phys_flow_feature"] = df["attenuation"] * df["belt_speed_mps"]
        cols = ["attenuation", "phys_flow_feature", "i0_online", *cols]
    if temporal:
        cols += ["attenuation_mean30", "attenuation_std30", "rx_mean30", "rx_std30"]
    x = df[cols].astype(float).copy()
    coal = pd.get_dummies(df["coal_type"], prefix="coal", dtype=float)
    for name in COAL_ORDER:
        col = f"coal_{name}"
        if col not in coal.columns:
            coal[col] = 0.0
    coal = coal[[f"coal_{name}" for name in COAL_ORDER]]
    return pd.concat([x.reset_index(drop=True), coal.reset_index(drop=True)], axis=1)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.maximum(y_pred, 0.0)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "cum_err_pct": float(abs(pred.sum() - y_true.sum()) / max(float(y_true.sum()), 1e-9) * 100.0),
        "true_total_kg": float(y_true.sum()),
        "pred_total_kg": float(pred.sum()),
    }


def metric_row(method: str, split: str, df: pd.DataFrame, pred: np.ndarray, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    y = df["true_flow_kgps"].to_numpy(float)
    row = {"method": method, "split": split, "n": len(df), **metrics(y, pred)}
    loaded = df["true_flow_kgps"].to_numpy(float) > 1e-9
    if loaded.any():
        loaded_metrics = metrics(y[loaded], pred[loaded])
        row.update({f"loaded_{key}": value for key, value in loaded_metrics.items()})
        row["loaded_n"] = int(loaded.sum())
    else:
        row["loaded_n"] = 0
    if extra:
        row.update(extra)
    return row


def fit_physics(df: pd.DataFrame) -> float:
    x = (df["attenuation"] * df["belt_speed_mps"]).to_numpy(float)
    y = df["true_flow_kgps"].to_numpy(float)
    return float((x @ y) / max(float(x @ x), 1e-9))


def fit_predictions(train: pd.DataFrame, test: pd.DataFrame, quick: bool, seed: int) -> dict[str, np.ndarray]:
    y = train["true_flow_kgps"].to_numpy(float)
    predictions: dict[str, np.ndarray] = {}
    k = fit_physics(train)
    predictions["Physics+I0"] = k * (test["attenuation"] * test["belt_speed_mps"]).to_numpy(float)

    configs = [
        ("DataOnly-Ridge", False, False, True),
        ("Physics-Ridge", True, False, True),
        ("Physics-Temporal-Ridge", True, True, True),
        ("NoMissingFlag-Ablation", True, True, False),
    ]
    for name, physics, temporal, missing in configs:
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
        model.fit(feature_frame(train, physics=physics, temporal=temporal, missing=missing), y)
        predictions[name] = model.predict(feature_frame(test, physics=physics, temporal=temporal, missing=missing))

    if not quick:
        x_train = feature_frame(train, physics=True, temporal=True, missing=True)
        x_test = feature_frame(test, physics=True, temporal=True, missing=True)
        hgb = HistGradientBoostingRegressor(max_iter=240, learning_rate=0.055, l2_regularization=0.05, random_state=seed)
        hgb.fit(x_train, y)
        predictions["HGB"] = hgb.predict(x_test)
        rf = RandomForestRegressor(n_estimators=180, min_samples_leaf=8, random_state=seed, n_jobs=-1)
        rf.fit(x_train, y)
        predictions["RandomForest"] = rf.predict(x_test)
    return predictions


def split_calibration_eval(test: pd.DataFrame, calibration_fraction: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    calibration: dict[str, np.ndarray] = {}
    evaluation: dict[str, np.ndarray] = {}
    eval_indices: list[int] = []
    for key, g in test.groupby(["site_id", "device_id", "install_id"], sort=False):
        idx = g.sort_values("timestamp_s").index.to_numpy(dtype=int)
        cut = max(1, min(len(idx) - 1, int(round(len(idx) * calibration_fraction))))
        k = "|".join(map(str, key))
        calibration[k] = idx[:cut]
        evaluation[k] = idx[cut:]
        eval_indices.extend(idx[cut:].tolist())
    return calibration, evaluation, np.asarray(sorted(eval_indices), dtype=int)


def choose_samples(
    test: pd.DataFrame,
    pred: np.ndarray,
    candidates: np.ndarray,
    budget: int,
    strategy: str,
    rng: random.Random,
) -> np.ndarray:
    candidates = candidates[(test.loc[candidates, "empty_flag"].to_numpy(float) == 0) & (pred[candidates] > 1e-9)]
    if budget <= 0 or len(candidates) == 0:
        return np.array([], dtype=int)
    n = min(int(budget), len(candidates))
    if strategy == "random":
        return np.asarray(sorted(rng.sample(candidates.tolist(), n)), dtype=int)
    if strategy == "time_uniform":
        order = candidates[np.argsort(test.loc[candidates, "timestamp_s"].to_numpy(float))]
    elif strategy == "prediction_stratified":
        order = candidates[np.argsort(pred[candidates])]
    elif strategy == "load_stratified":
        order = candidates[np.argsort(test.loc[candidates, "true_flow_kgps"].to_numpy(float))]
    else:
        raise ValueError(f"unknown strategy: {strategy}")
    pos = np.linspace(0, len(order) - 1, n).round().astype(int)
    return np.asarray(sorted(set(order[pos].tolist())), dtype=int)


def calibrate_per_install(
    test: pd.DataFrame,
    pred: np.ndarray,
    calibration: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    budget: int,
    strategy: str,
    seed: int,
    mass_anchor_bound_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted = np.maximum(pred.copy(), 0.0)
    eval_all: list[int] = []
    rng = random.Random(seed)
    y = test["true_flow_kgps"].to_numpy(float)
    for key, cal_idx in calibration.items():
        eval_idx = evaluation[key]
        eval_all.extend(eval_idx.tolist())
        sample_idx = choose_samples(test, pred, cal_idx, budget, strategy, rng)
        if len(sample_idx) < 2:
            continue
        x = pred[sample_idx]
        target = y[sample_idx]
        slope, intercept = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), target, rcond=None)[0]
        local = np.maximum(slope * pred[eval_idx] + intercept, 0.0)
        sample_pred = np.maximum(slope * pred[sample_idx] + intercept, 0.0)
        factor = target.sum() / max(float(sample_pred.sum()), 1e-9)
        bound = mass_anchor_bound_pct / 100.0
        factor = float(np.clip(factor, 1.0 - bound, 1.0 + bound))
        adjusted[eval_idx] = np.maximum(local * factor, 0.0)
    return adjusted, np.asarray(sorted(eval_all), dtype=int)


def calibrate_mass_scale_per_install(
    test: pd.DataFrame,
    pred: np.ndarray,
    calibration: dict[str, np.ndarray],
    evaluation: dict[str, np.ndarray],
    budget: int,
    strategy: str,
    seed: int,
    mass_anchor_bound_pct: float,
) -> tuple[np.ndarray, np.ndarray]:
    adjusted = np.maximum(pred.copy(), 0.0)
    eval_all: list[int] = []
    rng = random.Random(seed)
    y = test["true_flow_kgps"].to_numpy(float)
    for key, cal_idx in calibration.items():
        eval_idx = evaluation[key]
        eval_all.extend(eval_idx.tolist())
        sample_idx = choose_samples(test, pred, cal_idx, budget, strategy, rng)
        if len(sample_idx) < 2:
            continue
        factor = y[sample_idx].sum() / max(float(np.maximum(pred[sample_idx], 0.0).sum()), 1e-9)
        bound = mass_anchor_bound_pct / 100.0
        factor = float(np.clip(factor, 1.0 - bound, 1.0 + bound))
        adjusted[eval_idx] = np.maximum(pred[eval_idx] * factor, 0.0)
    return adjusted, np.asarray(sorted(eval_all), dtype=int)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def stratified_rows(test: pd.DataFrame, pred_map: dict[str, np.ndarray], eval_idx: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = [
        name
        for name in [
            "Physics-Temporal-Ridge",
            "HGB",
            "PG-TSC-real-mass-scale budget=5 strategy=prediction_stratified",
            "PG-TSC-real-mass-scale budget=60 strategy=prediction_stratified",
        ]
        if name in pred_map
    ]
    groups: list[tuple[str, str, np.ndarray]] = []
    for col, labels in [("env_temp_c", ("low", "mid", "high")), ("env_humidity_pct", ("low", "mid", "high"))]:
        values = test.loc[eval_idx, col].to_numpy(float)
        q1, q2 = np.quantile(values, [1 / 3, 2 / 3])
        groups.extend(
            [
                (col, labels[0], eval_idx[values <= q1]),
                (col, labels[1], eval_idx[(values > q1) & (values <= q2)]),
                (col, labels[2], eval_idx[values > q2]),
            ]
        )
    for col in ["coal_type", "site_id", "device_id", "ash_missing"]:
        for value in sorted(test.loc[eval_idx, col].astype(str).unique()):
            idx = eval_idx[test.loc[eval_idx, col].astype(str).to_numpy() == value]
            groups.append((col, value, idx))
    for dimension, group, idx in groups:
        if len(idx) == 0:
            continue
        subset = test.loc[idx]
        for method in methods:
            rows.append(metric_row(method, f"{dimension}={group}", subset, pred_map[method][idx]))
    return rows


def install_rows(test: pd.DataFrame, pred_map: dict[str, np.ndarray], eval_idx: np.ndarray) -> list[dict[str, Any]]:
    eval_set = set(eval_idx.tolist())
    rows: list[dict[str, Any]] = []
    methods = [
        name
        for name in [
            "Physics-Temporal-Ridge",
            "HGB",
            "PG-TSC-real-mass-scale budget=5 strategy=prediction_stratified",
            "PG-TSC-real-mass-scale budget=60 strategy=prediction_stratified",
        ]
        if name in pred_map
    ]
    for (site, device, install), g in test.groupby(["site_id", "device_id", "install_id"], sort=True):
        idx = np.asarray([i for i in g.index.to_numpy(dtype=int) if i in eval_set], dtype=int)
        if len(idx) == 0:
            continue
        subset = test.loc[idx]
        for method in methods:
            rows.append(metric_row(method, "install_eval", subset, pred_map[method][idx], {"site_id": site, "device_id": device, "install_id": install}))
    return rows


def main() -> int:
    args = parse_args()
    df = pd.read_csv(args.input)
    df = df.sort_values(["site_id", "device_id", "install_id", "timestamp_s"], kind="mergesort").reset_index(drop=True)
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()
    for col in ["train_excluded", "eval_excluded"]:
        if col not in df.columns:
            df[col] = 0
    df = frame_with_online_features(df)
    df["train_excluded"] = pd.to_numeric(df["train_excluded"], errors="coerce").fillna(0).astype(int)
    df["eval_excluded"] = pd.to_numeric(df["eval_excluded"], errors="coerce").fillna(0).astype(int)

    train = df[df["site_id"].isin(TRAIN_SITES) & df["train_excluded"].eq(0)].copy()
    test_all = df[df["site_id"].isin(TEST_SITES)].copy()
    test_clean = test_all[test_all["eval_excluded"].eq(0)].copy().reset_index(drop=True)
    test_anomaly = test_all[test_all["train_excluded"].eq(1) & test_all["eval_excluded"].eq(0)].copy().reset_index(drop=True)
    test_for_eval = test_clean[test_clean["train_excluded"].eq(0)].copy().reset_index(drop=True)
    if train.empty or test_for_eval.empty:
        raise RuntimeError("real data split is empty; expected train S01-S03 and test S04-S05")

    predictions = fit_predictions(train, test_for_eval, quick=args.quick, seed=args.seed)
    calibration, evaluation, eval_idx = split_calibration_eval(test_for_eval, args.calibration_fraction)

    summary: list[dict[str, Any]] = []
    for name, pred in predictions.items():
        summary.append(metric_row(name, "S04_S05_eval_clean", test_for_eval.loc[eval_idx], pred[eval_idx]))

    base = predictions["Physics-Temporal-Ridge"]
    calibration_rows: list[dict[str, Any]] = []
    for budget in [0, 5, 20, 60, 180]:
        for strategy in ["time_uniform", "prediction_stratified", "load_stratified", "random"]:
            if budget == 0 and strategy != "time_uniform":
                continue
            pred, cal_eval = calibrate_per_install(
                test_for_eval,
                base,
                calibration,
                evaluation,
                budget,
                strategy,
                args.seed + budget + len(strategy),
                args.mass_anchor_bound_pct,
            )
            method = f"PG-TSC-real budget={budget} strategy={strategy}"
            predictions[method] = pred
            row = metric_row(method, "S04_S05_eval_clean", test_for_eval.loc[cal_eval], pred[cal_eval], {"budget": budget, "strategy": strategy})
            calibration_rows.append(row)
            if strategy == "prediction_stratified" and budget in {5, 20, 60, 180}:
                summary.append(row)
            if budget == 0 and strategy == "time_uniform":
                summary.append(row)
            if budget > 0:
                scale_pred, scale_eval = calibrate_mass_scale_per_install(
                    test_for_eval,
                    base,
                    calibration,
                    evaluation,
                    budget,
                    strategy,
                    args.seed + 1000 + budget + len(strategy),
                    args.mass_anchor_bound_pct,
                )
                scale_method = f"PG-TSC-real-mass-scale budget={budget} strategy={strategy}"
                predictions[scale_method] = scale_pred
                scale_row = metric_row(
                    scale_method,
                    "S04_S05_eval_clean",
                    test_for_eval.loc[scale_eval],
                    scale_pred[scale_eval],
                    {"budget": budget, "strategy": strategy, "calibration_mode": "mass_scale"},
                )
                calibration_rows.append(scale_row)
                if strategy == "prediction_stratified" and budget in {5, 20, 60, 180}:
                    summary.append(scale_row)

    stratified = stratified_rows(test_for_eval, predictions, eval_idx)
    installs = install_rows(test_for_eval, predictions, eval_idx)

    qc_rows: list[dict[str, Any]] = []
    if not test_anomaly.empty:
        anomaly_predictions = fit_predictions(train, test_anomaly, quick=args.quick, seed=args.seed)
        for name, pred in anomaly_predictions.items():
            qc_rows.append(metric_row(name, "S04_S05_anomaly_holdout", test_anomaly, pred))

    for path, rows in [
        (args.summary, summary),
        (args.calibration, calibration_rows),
        (args.stratified, stratified),
        (args.install_errors, installs),
        (args.qc_holdout, qc_rows),
    ]:
        write_csv(path, rows)
        meta = write_run_metadata(
            path,
            script=__file__,
            args={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
            inputs=[args.input],
            outputs=[path],
            notes="Real X-ray belt data experiment output. Main split trains on S01-S03 and evaluates S04-S05 with per-install few-shot calibration.",
        )
        print(f"wrote={path}")
        print(f"wrote={meta}")
    print(f"train_rows={len(train)}")
    print(f"test_clean_rows={len(test_for_eval)}")
    print(f"eval_rows={len(eval_idx)}")
    print(f"test_anomaly_rows={len(test_anomaly)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
