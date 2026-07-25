from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from evaluation_protocol import PROTOCOL_VERSION, evaluate_installation_holdout, protocol_fields
from experiment_versioning import write_run_metadata
from run_monte_carlo_robustness import simulate_rows


@dataclass(frozen=True)
class ModelConfig:
    name: str
    estimator: str
    include_physics: bool
    temporal: bool
    include_missing: bool = True


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.maximum(y_pred, 0.0)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "cum_err_pct": float(abs(pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-9) * 100),
    }


def make_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["phys_flow_feature"] = df["attenuation"] * df["belt_speed_mps"]
    return df


def get_features(df: pd.DataFrame, include_physics: bool, temporal: bool, include_missing: bool) -> pd.DataFrame:
    cols = [
        "tube_voltage_kv",
        "tube_current_ma",
        "source_temp_c",
        "env_temp_c",
        "env_humidity_pct",
        "belt_speed_mps",
        "ash_pct",
    ]
    if include_missing:
        cols.append("ash_missing")
    if include_physics:
        cols += ["attenuation", "i0_online", "phys_flow_feature"]
    if temporal:
        cols += ["attenuation_mean30", "attenuation_std30", "rx_mean30", "rx_std30"]
    x = df[cols].copy()
    coal = pd.get_dummies(df["coal_type"], prefix="coal", dtype=float)
    return pd.concat([x, coal], axis=1)


def test_context(test: pd.DataFrame, y_eval: np.ndarray) -> dict[str, float]:
    return {
        "test_rows": float(len(y_eval)),
        "test_mass_kg": float(y_eval.sum()),
        "test_mean_flow_kgps": float(y_eval.mean()),
        "target_empty_ratio": float(test["empty_flag"].astype(float).mean()),
        "target_ash_missing_ratio": float(test["ash_missing"].astype(float).mean()),
    }


def build_estimator(kind: str, *, seed: int, hgb_max_iter: int, rf_estimators: int):
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=hgb_max_iter,
            learning_rate=0.06,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=seed,
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=rf_estimators,
            max_depth=18,
            min_samples_leaf=8,
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"unknown estimator kind: {kind}")


def evaluate_split(
    df: pd.DataFrame,
    *,
    replicate: int,
    seed: int,
    heldout_site: str,
    calibrate_samples: list[int],
    hgb_max_iter: int,
    rf_estimators: int,
    adapt_fraction: float,
) -> list[dict[str, object]]:
    train = df[df["site_id"].ne(heldout_site)].copy()
    test = df[df["site_id"].eq(heldout_site)].copy()
    y_train = train["true_flow_kgps"].to_numpy(float)
    configs = [
        ModelConfig("DataOnly-HGB", "hgb", False, False),
        ModelConfig("Physics-HGB", "hgb", True, False),
        ModelConfig("Physics-Temporal-HGB", "hgb", True, True),
        ModelConfig("Physics-Temporal-RF", "rf", True, True),
    ]

    out_rows: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}

    def append_prediction(method: str, prediction: np.ndarray, calibration_samples: int = 0) -> None:
        prediction_frame = test[["site_id", "device_id", "install_id", "timestamp_s"]].copy()
        prediction_frame["y_true"] = test["true_flow_kgps"].to_numpy(float)
        prediction_frame["y_pred"] = prediction
        evaluation = evaluate_installation_holdout(
            prediction_frame,
            calibration_samples=calibration_samples,
            adapt_fraction=adapt_fraction,
        )
        out_rows.append(
            {
                "replicate": replicate,
                "seed": seed,
                "heldout_site": heldout_site,
                "method": method,
                "calibration_samples_per_install": calibration_samples,
                **metrics(evaluation.y_true, evaluation.y_pred),
                **test_context(test, evaluation.y_true),
                **protocol_fields(evaluation, adapt_fraction),
            }
        )
    for model_idx, config in enumerate(configs):
        x_train = get_features(train, config.include_physics, config.temporal, config.include_missing)
        x_test = get_features(test, config.include_physics, config.temporal, config.include_missing)
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0.0)
        model = build_estimator(
            config.estimator,
            seed=seed + model_idx,
            hgb_max_iter=hgb_max_iter,
            rf_estimators=rf_estimators,
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        predictions[config.name] = pred
        append_prediction(config.name, pred)

    base_pred = predictions["Physics-Temporal-HGB"]
    for n_samples in calibrate_samples:
        append_prediction(f"PG-TSC-lite-HGB calibrate={n_samples}", base_pred, n_samples)
    return out_rows


def parse_samples(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Monte Carlo strong-baseline evaluation for PG-TSC simulation.")
    parser.add_argument("--output", default="results/sim_monte_carlo_strong_baselines_v2.csv")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--devices-per-site", type=int, default=2)
    parser.add_argument("--installs-per-device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--calibrate-samples", default="20,60,180")
    parser.add_argument("--hgb-max-iter", type=int, default=180)
    parser.add_argument("--rf-estimators", type=int, default=80)
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    args = parser.parse_args()

    calibrate_samples = parse_samples(args.calibrate_samples)
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
        df = make_frame(rows)
        for site_idx in range(1, args.sites + 1):
            heldout_site = f"S{site_idx:02d}"
            all_rows.extend(
                evaluate_split(
                    df,
                    replicate=replicate,
                    seed=seed,
                    heldout_site=heldout_site,
                    calibrate_samples=calibrate_samples,
                    hgb_max_iter=args.hgb_max_iter,
                    rf_estimators=args.rf_estimators,
                    adapt_fraction=args.adapt_fraction,
                )
            )
        print(f"replicate={replicate + 1}/{args.replicates} rows={len(all_rows)}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].keys())
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    meta = write_run_metadata(
        out,
        script="experiments/run_monte_carlo_strong_baselines.py",
        args=vars(args),
        inputs=[],
        outputs=[out],
        notes=(
            "Monte Carlo leave-one-site evaluation against nonlinear baselines using causal I0 "
            f"features and protocol {PROTOCOL_VERSION}."
        ),
    )
    print(f"wrote={out}")
    print(f"wrote={meta}")
    print(f"evaluation_rows={len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
