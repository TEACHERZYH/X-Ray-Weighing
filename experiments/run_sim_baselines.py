from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from evaluation_protocol import (
    PROTOCOL_VERSION,
    evaluate_installation_holdout,
    prediction_frame_from_rows,
    protocol_fields,
)
from experiment_versioning import write_run_metadata


COAL_ORDER = ["anthracite", "bituminous_A", "bituminous_B", "lignite"]


def read_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
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
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            parsed: dict[str, object] = {}
            for k, v in row.items():
                if k in numeric_fields:
                    parsed[k] = float(v)
                else:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def add_online_features(
    rows: list[dict[str, object]],
    ema_alpha: float = 0.985,
    initial_empty_samples: int = 60,
    empty_attenuation_threshold: float = 0.035,
    empty_relative_threshold: float = 0.04,
) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for r in rows:
        groups[(str(r["site_id"]), str(r["device_id"]), str(r["install_id"]))].append(r)

    for group_rows in groups.values():
        group_rows.sort(key=lambda x: float(x["timestamp_s"]))
        warmup_count = min(max(int(initial_empty_samples), 1), len(group_rows))
        warmup_rx = [max(float(r["rx_value"]), 1.0) for r in group_rows[:warmup_count]]
        i0 = float(np.median(warmup_rx))
        hist_a: deque[float] = deque(maxlen=30)
        hist_rx: deque[float] = deque(maxlen=30)
        for index, r in enumerate(group_rows):
            rx = max(float(r["rx_value"]), 1.0)
            attenuation_before_update = max(0.0, math.log(max(i0, rx + 1e-6) / rx))
            relative_gap = abs(rx - i0) / max(i0, 1.0)
            update_gate = (
                index >= warmup_count
                and attenuation_before_update <= empty_attenuation_threshold
                and relative_gap <= empty_relative_threshold
            )
            if update_gate:
                i0 = ema_alpha * i0 + (1 - ema_alpha) * rx
            attenuation = max(0.0, math.log(max(i0, rx + 1e-6) / rx))
            hist_a.append(attenuation)
            hist_rx.append(rx)
            r["i0_online"] = i0
            r["attenuation"] = attenuation
            r["attenuation_mean30"] = float(np.mean(hist_a))
            r["attenuation_std30"] = float(np.std(hist_a))
            r["rx_mean30"] = float(np.mean(hist_rx))
            r["rx_std30"] = float(np.std(hist_rx))
            r["i0_ready"] = int(index >= warmup_count)
            r["i0_update_gate"] = int(update_gate)


def ridge_fit(x: np.ndarray, y: np.ndarray, lam: float = 1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std[std < 1e-8] = 1.0
    xs = (x - mean) / std
    xb = np.column_stack([np.ones(xs.shape[0]), xs])
    eye = np.eye(xb.shape[1])
    eye[0, 0] = 0
    beta = np.linalg.solve(xb.T @ xb + lam * eye, xb.T @ y)
    return beta, mean, std


def ridge_predict(x: np.ndarray, beta: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    xs = (x - mean) / std
    xb = np.column_stack([np.ones(xs.shape[0]), xs])
    return xb @ beta


def feature_matrix(rows: list[dict[str, object]], include_physics: bool, include_missing_flag: bool, temporal: bool) -> np.ndarray:
    feats: list[list[float]] = []
    for r in rows:
        coal = str(r["coal_type"])
        one_hot = [1.0 if coal == c else 0.0 for c in COAL_ORDER]
        base = [
            float(r["tube_voltage_kv"]),
            float(r["tube_current_ma"]),
            float(r["source_temp_c"]),
            float(r["env_temp_c"]),
            float(r["env_humidity_pct"]),
            float(r["belt_speed_mps"]),
            float(r["ash_pct"]),
            *one_hot,
        ]
        if include_missing_flag:
            base.append(float(r["ash_missing"]))
        if include_physics:
            phys_flow = float(r["attenuation"]) * float(r["belt_speed_mps"])
            base = [float(r["attenuation"]), phys_flow, float(r["i0_online"])] + base
        if temporal:
            base += [
                float(r["attenuation_mean30"]),
                float(r["attenuation_std30"]),
                float(r["rx_mean30"]),
                float(r["rx_std30"]),
            ]
        feats.append(base)
    return np.asarray(feats, dtype=float)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_pred = np.maximum(y_pred, 0.0)
    err = y_pred - y_true
    true_total = float(y_true.sum())
    pred_total = float(y_pred.sum())
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "cum_err_pct": abs(pred_total - true_total) / max(true_total, 1e-9) * 100,
    }


def fit_physics(train_rows: list[dict[str, object]]) -> float:
    x = np.array([float(r["attenuation"]) * float(r["belt_speed_mps"]) for r in train_rows], dtype=float)
    y = np.array([float(r["true_flow_kgps"]) for r in train_rows], dtype=float)
    return float((x @ y) / max(x @ x, 1e-9))


def limit_rows_per_install(rows: list[dict[str, object]], maximum: int) -> list[dict[str, object]]:
    if maximum <= 0:
        return rows
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["site_id"]), str(row["device_id"]), str(row["install_id"]))].append(row)
    limited: list[dict[str, object]] = []
    for group_rows in groups.values():
        group_rows.sort(key=lambda x: float(x["timestamp_s"]))
        limited.extend(group_rows[:maximum])
    return limited


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/simulated/xray_belt_sim.csv")
    parser.add_argument("--output", default="results/sim_baselines_v2.csv")
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-samples", default="0,5,20,60,180")
    parser.add_argument("--initial-empty-samples", type=int, default=60)
    parser.add_argument("--max-rows-per-install", type=int, default=0)
    args = parser.parse_args()

    rows = read_rows(Path(args.input))
    rows = limit_rows_per_install(rows, args.max_rows_per_install)
    add_online_features(rows, initial_empty_samples=args.initial_empty_samples)

    train_rows = [r for r in rows if str(r["site_id"]) in {"S01", "S02", "S03"}]
    test_rows = [r for r in rows if str(r["site_id"]) == "S04"]
    y_train = np.array([float(r["true_flow_kgps"]) for r in train_rows])
    results: list[dict[str, object]] = []

    def append_result(method: str, prediction: np.ndarray, calibration_samples: int = 0) -> None:
        prediction_frame = prediction_frame_from_rows(test_rows, prediction)
        evaluation = evaluate_installation_holdout(
            prediction_frame,
            calibration_samples=calibration_samples,
            adapt_fraction=args.adapt_fraction,
        )
        results.append(
            {
                "method": method,
                "calibration_samples_per_install": calibration_samples,
                **metrics(evaluation.y_true, evaluation.y_pred),
                **protocol_fields(evaluation, args.adapt_fraction),
            }
        )

    k = fit_physics(train_rows)
    phys_pred = np.array([k * float(r["attenuation"]) * float(r["belt_speed_mps"]) for r in test_rows])
    append_result("Physics+I0", phys_pred)

    for name, include_physics, include_missing, temporal in [
        ("DataOnly-Ridge", False, True, False),
        ("Physics-Ridge", True, True, False),
        ("Physics-Temporal-Ridge", True, True, True),
        ("NoMissingFlag-Ablation", True, False, True),
    ]:
        x_train = feature_matrix(train_rows, include_physics, include_missing, temporal)
        x_test = feature_matrix(test_rows, include_physics, include_missing, temporal)
        beta, mean, std = ridge_fit(x_train, y_train, lam=10.0)
        pred = ridge_predict(x_test, beta, mean, std)
        append_result(name, pred)

        if name == "Physics-Temporal-Ridge":
            samples = [int(value.strip()) for value in args.calibration_samples.split(",") if value.strip()]
            for n in samples:
                append_result(f"PG-TSC-lite calibrate={n}", pred, n)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(results[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"wrote={out}")
    meta = write_run_metadata(
        out,
        script=__file__,
        args=vars(args),
        inputs=[args.input],
        outputs=[out],
        notes=(
            "Deterministic physics and ridge baselines using causal I0 features and "
            f"chronological installation holdout protocol {PROTOCOL_VERSION}."
        ),
    )
    print(f"wrote={meta}")
    for row in results:
        print(f"{row['method']}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, CumErr={row['cum_err_pct']:.3f}%")


if __name__ == "__main__":
    main()
