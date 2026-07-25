from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation_protocol import PROTOCOL_VERSION, evaluate_installation_holdout, protocol_fields
from experiment_versioning import write_run_metadata
from run_sim_baselines import add_online_features, limit_rows_per_install, read_rows


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.maximum(y_pred, 0.0)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "cum_err_pct": abs(y_pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-9) * 100,
    }


def make_frame(path: Path, *, initial_empty_samples: int = 60, max_rows_per_install: int = 0) -> pd.DataFrame:
    rows = read_rows(path)
    rows = limit_rows_per_install(rows, max_rows_per_install)
    add_online_features(rows, initial_empty_samples=initial_empty_samples)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/simulated/xray_belt_sim.csv")
    parser.add_argument("--output", default="results/sim_ml_results_v2.csv")
    parser.add_argument("--predictions-output", default="results/sim_predictions_v2.csv")
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-samples", default="0,20,60,180,600")
    parser.add_argument("--initial-empty-samples", type=int, default=60)
    parser.add_argument("--max-rows-per-install", type=int, default=0)
    args = parser.parse_args()

    df = make_frame(
        Path(args.input),
        initial_empty_samples=args.initial_empty_samples,
        max_rows_per_install=args.max_rows_per_install,
    )
    train = df[df["site_id"].isin(["S01", "S02", "S03"])].copy()
    test = df[df["site_id"].eq("S04")].copy()
    y_train = train["true_flow_kgps"].to_numpy(float)

    configs = [
        ("DataOnly-Ridge", Ridge(alpha=10.0), False, False, True),
        ("Physics-Ridge", Ridge(alpha=10.0), True, False, True),
        ("Physics-Temporal-Ridge", Ridge(alpha=10.0), True, True, True),
        ("DataOnly-HGB", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=31, random_state=7), False, False, True),
        ("Physics-HGB", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=31, random_state=7), True, False, True),
        ("Physics-Temporal-HGB", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=31, random_state=7), True, True, True),
        ("Physics-Temporal-RF", RandomForestRegressor(n_estimators=120, max_depth=18, min_samples_leaf=8, n_jobs=-1, random_state=7), True, True, True),
        ("NoMissingFlag-HGB", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.06, max_leaf_nodes=31, random_state=7), True, True, False),
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
            adapt_fraction=args.adapt_fraction,
        )
        out_rows.append(
            {
                "method": method,
                "calibration_samples_per_install": calibration_samples,
                **metrics(evaluation.y_true, evaluation.y_pred),
                **protocol_fields(evaluation, args.adapt_fraction),
            }
        )
    for name, estimator, include_physics, temporal, include_missing in configs:
        x_train = get_features(train, include_physics, temporal, include_missing)
        x_test = get_features(test, include_physics, temporal, include_missing)
        x_test = x_test.reindex(columns=x_train.columns, fill_value=0)
        if isinstance(estimator, Ridge):
            model = make_pipeline(StandardScaler(), estimator)
        else:
            model = estimator
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        predictions[name] = pred
        append_prediction(name, pred)

    base_pred = predictions["Physics-Temporal-HGB"]
    for n in [int(value.strip()) for value in args.calibration_samples.split(",") if value.strip()]:
        append_prediction(f"PG-TSC-lite-HGB calibrate={n}", base_pred, n)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out_rows).to_csv(out, index=False)

    pred_out = Path(args.predictions_output)
    pred_df = test[["timestamp_s", "site_id", "device_id", "install_id", "true_flow_kgps"]].copy()
    pred_df["protocol_partition"] = (
        pred_df.sort_values(["site_id", "device_id", "install_id", "timestamp_s"])
        .groupby(["site_id", "device_id", "install_id"])["timestamp_s"]
        .transform(lambda s: np.where(np.arange(len(s)) < max(1, int(np.floor(len(s) * args.adapt_fraction))), "adapt", "eval"))
    )
    for name, pred in predictions.items():
        if name in {"Physics-Ridge", "Physics-Temporal-Ridge", "Physics-Temporal-HGB", "DataOnly-HGB"}:
            pred_df[name] = np.maximum(pred, 0.0)
    pred_df.to_csv(pred_out, index=False)

    meta = write_run_metadata(
        out,
        script=__file__,
        args=vars(args),
        inputs=[args.input],
        outputs=[out, pred_out],
        notes=(
            "Scikit-learn models and prediction table using causal I0 features and "
            f"chronological installation holdout protocol {PROTOCOL_VERSION}."
        ),
    )
    print(f"wrote={out}")
    print(f"wrote={pred_out}")
    print(f"wrote={meta}")
    for row in out_rows:
        print(f"{row['method']}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, CumErr={row['cum_err_pct']:.3f}%")


if __name__ == "__main__":
    main()
