from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from evaluation_protocol import PROTOCOL_VERSION, evaluate_installation_holdout, protocol_fields
from experiment_versioning import write_run_metadata
from run_monte_carlo_robustness import parse_samples
from run_sim_baselines import add_online_features
from simulate_xray_belt import simulate_install


@dataclass(frozen=True)
class MismatchConfig:
    name: str
    order: int
    attenuation_scale: float
    detector_bias: float
    drift_scale: float
    noise_std: float
    humidity_coupling: float


MISMATCH_LEVELS = [
    MismatchConfig("nominal", 0, 1.00, 0.0, 0.000, 0.0, 0.0000),
    MismatchConfig("mild", 1, 1.03, 25.0, 0.010, 10.0, -0.0006),
    MismatchConfig("moderate", 2, 1.08, 60.0, 0.022, 22.0, -0.0013),
    MismatchConfig("strong", 3, 1.15, 110.0, 0.040, 38.0, -0.0022),
]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    include_physics: bool
    temporal: bool
    include_missing: bool = True


def smooth_walk(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    if scale <= 0:
        return np.zeros(n)
    steps = rng.normal(0.0, scale / max(n, 1) ** 0.5, size=n)
    walk = np.cumsum(steps)
    walk -= walk.mean()
    return walk


def simulate_raw_rows(
    *,
    seed: int,
    sites: int,
    devices_per_site: int,
    installs_per_device: int,
    seconds: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    run_seed = seed
    for site_idx in range(1, sites + 1):
        for device_idx in range(1, devices_per_site + 1):
            for install_idx in range(1, installs_per_device + 1):
                rows.extend(simulate_install(site_idx, device_idx, install_idx, seconds, run_seed))
                run_seed += 1
    return rows


def grouped_install_rows(rows: list[dict[str, object]]) -> dict[tuple[str, str, str], list[dict[str, object]]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["site_id"]), str(row["device_id"]), str(row["install_id"]))
        groups.setdefault(key, []).append(row)
    return groups


def apply_mismatch(rows: list[dict[str, object]], config: MismatchConfig, *, seed: int) -> list[dict[str, object]]:
    shifted = deepcopy(rows)
    if config.name == "nominal":
        return shifted

    rng = np.random.default_rng(seed)
    for install_rows in grouped_install_rows(shifted).values():
        install_rows.sort(key=lambda x: float(x["timestamp_s"]))
        rx = np.array([float(r["rx_value"]) for r in install_rows], dtype=float)
        empty = np.array([int(float(r["empty_flag"])) for r in install_rows], dtype=int)
        humidity = np.array([float(r["env_humidity_pct"]) for r in install_rows], dtype=float)
        if np.any(empty == 1):
            i0_ref = float(np.median(rx[empty == 1]))
        else:
            i0_ref = float(np.percentile(rx, 95))
        attenuation = np.log(np.maximum(i0_ref, rx + 1e-6) / np.maximum(rx, 1.0))
        drift = smooth_walk(len(install_rows), config.drift_scale, rng)
        humidity_term = config.humidity_coupling * (humidity - 55.0)
        transformed = i0_ref * np.exp(-config.attenuation_scale * attenuation)
        transformed *= 1.0 + drift + humidity_term
        transformed += config.detector_bias + rng.normal(0.0, config.noise_std, size=len(install_rows))
        transformed = np.maximum(transformed, 1.0)
        for row, value in zip(install_rows, transformed):
            row["rx_value"] = round(float(value), 6)
    return shifted


def make_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    add_online_features(rows)
    df = pd.DataFrame(rows)
    df["phys_flow_feature"] = df["attenuation"] * df["belt_speed_mps"]
    return df


def get_features(df: pd.DataFrame, config: ModelConfig) -> pd.DataFrame:
    cols = [
        "tube_voltage_kv",
        "tube_current_ma",
        "source_temp_c",
        "env_temp_c",
        "env_humidity_pct",
        "belt_speed_mps",
        "ash_pct",
    ]
    if config.include_missing:
        cols.append("ash_missing")
    if config.include_physics:
        cols += ["attenuation", "i0_online", "phys_flow_feature"]
    if config.temporal:
        cols += ["attenuation_mean30", "attenuation_std30", "rx_mean30", "rx_std30"]
    x = df[cols].copy()
    coal = pd.get_dummies(df["coal_type"], prefix="coal", dtype=float)
    return pd.concat([x, coal], axis=1)


def build_estimator(seed: int, hgb_max_iter: int) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        max_iter=hgb_max_iter,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=seed,
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.maximum(y_pred, 0.0)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "cum_err_pct": float(abs(pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-9) * 100),
    }


def test_context(test: pd.DataFrame, y_eval: np.ndarray) -> dict[str, float]:
    return {
        "test_rows": float(len(y_eval)),
        "test_mass_kg": float(y_eval.sum()),
        "test_mean_flow_kgps": float(y_eval.mean()),
        "target_empty_ratio": float(test["empty_flag"].astype(float).mean()),
        "target_ash_missing_ratio": float(test["ash_missing"].astype(float).mean()),
    }


def fit_source_models(train: pd.DataFrame, *, seed: int, hgb_max_iter: int) -> dict[str, tuple[ModelConfig, HistGradientBoostingRegressor, pd.Index]]:
    configs = [
        ModelConfig("DataOnly-HGB", False, False),
        ModelConfig("Physics-Temporal-HGB", True, True),
    ]
    y_train = train["true_flow_kgps"].to_numpy(float)
    models: dict[str, tuple[ModelConfig, HistGradientBoostingRegressor, pd.Index]] = {}
    for idx, config in enumerate(configs):
        x_train = get_features(train, config)
        model = build_estimator(seed + idx, hgb_max_iter)
        model.fit(x_train, y_train)
        models[config.name] = (config, model, x_train.columns)
    return models


def evaluate_target(
    *,
    models: dict[str, tuple[ModelConfig, HistGradientBoostingRegressor, pd.Index]],
    target_raw: list[dict[str, object]],
    mismatch: MismatchConfig,
    replicate: int,
    seed: int,
    heldout_site: str,
    calibrate_samples: list[int],
    adapt_fraction: float,
) -> list[dict[str, object]]:
    test_rows = apply_mismatch(target_raw, mismatch, seed=seed + mismatch.order * 997)
    test = make_frame(test_rows)

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
                "mismatch": mismatch.name,
                "mismatch_order": mismatch.order,
                "attenuation_scale": mismatch.attenuation_scale,
                "detector_bias": mismatch.detector_bias,
                "drift_scale": mismatch.drift_scale,
                "noise_std": mismatch.noise_std,
                "humidity_coupling": mismatch.humidity_coupling,
                "method": method,
                "calibration_samples_per_install": calibration_samples,
                **metrics(evaluation.y_true, evaluation.y_pred),
                **test_context(test, evaluation.y_true),
                **protocol_fields(evaluation, adapt_fraction),
            }
        )

    for method, (config, model, train_columns) in models.items():
        x_test = get_features(test, config).reindex(columns=train_columns, fill_value=0.0)
        pred = model.predict(x_test)
        predictions[method] = pred
        append_prediction(method, pred)

    base_pred = predictions["Physics-Temporal-HGB"]
    for n_samples in calibrate_samples:
        append_prediction(f"PG-TSC-lite-HGB calibrate={n_samples}", base_pred, n_samples)
    return out_rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulator-mismatch boundary analysis for PG-TSC calibration.")
    parser.add_argument("--output", default="results/sim_simulator_mismatch_boundary_v2.csv")
    parser.add_argument("--replicates", type=int, default=20)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--devices-per-site", type=int, default=2)
    parser.add_argument("--installs-per-device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--calibrate-samples", default="20,60,180")
    parser.add_argument("--hgb-max-iter", type=int, default=180)
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    args = parser.parse_args()

    calibrate_samples = parse_samples(args.calibrate_samples)
    all_rows: list[dict[str, object]] = []
    for replicate in range(args.replicates):
        seed = args.seed + replicate * 1000
        raw_rows = simulate_raw_rows(
            seed=seed,
            sites=args.sites,
            devices_per_site=args.devices_per_site,
            installs_per_device=args.installs_per_device,
            seconds=args.seconds,
        )
        for site_idx in range(1, args.sites + 1):
            heldout_site = f"S{site_idx:02d}"
            train_rows = [deepcopy(r) for r in raw_rows if str(r["site_id"]) != heldout_site]
            train = make_frame(train_rows)
            models = fit_source_models(train, seed=seed, hgb_max_iter=args.hgb_max_iter)
            target_raw = [deepcopy(r) for r in raw_rows if str(r["site_id"]) == heldout_site]
            for mismatch in MISMATCH_LEVELS:
                all_rows.extend(
                    evaluate_target(
                        models=models,
                        target_raw=target_raw,
                        mismatch=mismatch,
                        replicate=replicate,
                        seed=seed,
                        heldout_site=heldout_site,
                        calibrate_samples=calibrate_samples,
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
        script="experiments/run_simulator_mismatch_boundary.py",
        args=vars(args),
        inputs=[],
        outputs=[out],
        notes=(
            "Boundary analysis for simulator mismatch using causal I0 features and "
            f"chronological installation holdout protocol {PROTOCOL_VERSION}."
        ),
    )
    print(f"wrote={out}")
    print(f"wrote={meta}")
    print(f"evaluation_rows={len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
