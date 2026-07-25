from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from evaluation_protocol import (
    PROTOCOL_VERSION,
    evaluate_installation_holdout,
    prediction_frame_from_rows,
    protocol_fields,
)
from experiment_versioning import write_run_metadata
from run_sim_baselines import (
    add_online_features,
    feature_matrix,
    fit_physics,
    metrics,
    ridge_fit,
    ridge_predict,
)
from simulate_xray_belt import simulate_install


def simulate_rows(
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
    add_online_features(rows)
    return rows


def test_context(test_rows: list[dict[str, object]], y_eval: np.ndarray) -> dict[str, float]:
    return {
        "test_rows": float(len(y_eval)),
        "test_mass_kg": float(y_eval.sum()),
        "test_mean_flow_kgps": float(y_eval.mean()),
        "target_empty_ratio": float(np.mean([float(r["empty_flag"]) for r in test_rows])),
        "target_ash_missing_ratio": float(np.mean([float(r["ash_missing"]) for r in test_rows])),
    }


def evaluate_split(
    rows: list[dict[str, object]],
    *,
    replicate: int,
    seed: int,
    heldout_site: str,
    calibrate_samples: list[int],
    adapt_fraction: float,
) -> list[dict[str, object]]:
    train_rows = [r for r in rows if str(r["site_id"]) != heldout_site]
    test_rows = [r for r in rows if str(r["site_id"]) == heldout_site]
    y_train = np.array([float(r["true_flow_kgps"]) for r in train_rows], dtype=float)
    out_rows: list[dict[str, object]] = []

    def append_prediction(method: str, prediction: np.ndarray, calibration_samples: int = 0) -> None:
        evaluation = evaluate_installation_holdout(
            prediction_frame_from_rows(test_rows, prediction),
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
                **test_context(test_rows, evaluation.y_true),
                **protocol_fields(evaluation, adapt_fraction),
            }
        )

    k = fit_physics(train_rows)
    phys_pred = np.array([k * float(r["attenuation"]) * float(r["belt_speed_mps"]) for r in test_rows], dtype=float)
    append_prediction("Physics+I0", phys_pred)

    temporal_pred: np.ndarray | None = None
    configs = [
        ("DataOnly-Ridge", False, True, False),
        ("Physics-Ridge", True, True, False),
        ("Physics-Temporal-Ridge", True, True, True),
    ]
    for name, include_physics, include_missing, temporal in configs:
        x_train = feature_matrix(train_rows, include_physics, include_missing, temporal)
        x_test = feature_matrix(test_rows, include_physics, include_missing, temporal)
        beta, mean, std = ridge_fit(x_train, y_train, lam=10.0)
        pred = ridge_predict(x_test, beta, mean, std)
        if name == "Physics-Temporal-Ridge":
            temporal_pred = pred
        append_prediction(name, pred)

    if temporal_pred is None:
        return out_rows
    for n_samples in calibrate_samples:
        append_prediction(f"PG-TSC-lite calibrate={n_samples}", temporal_pred, n_samples)
    return out_rows


def parse_samples(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Monte Carlo leave-one-site robustness evaluation for PG-TSC simulation.")
    parser.add_argument("--output", default="results/sim_monte_carlo_robustness_v2.csv")
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--devices-per-site", type=int, default=2)
    parser.add_argument("--installs-per-device", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260628)
    parser.add_argument("--calibrate-samples", default="20,60,180")
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
        for site_idx in range(1, args.sites + 1):
            heldout_site = f"S{site_idx:02d}"
            all_rows.extend(
                evaluate_split(
                    rows,
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
        script="experiments/run_monte_carlo_robustness.py",
        args=vars(args),
        inputs=[],
        outputs=[out],
        notes=(
            "Monte Carlo leave-one-site robustness evaluation using causal I0 features and "
            f"chronological installation holdout protocol {PROTOCOL_VERSION}."
        ),
    )
    print(f"wrote={out}")
    print(f"wrote={meta}")
    print(f"evaluation_rows={len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
