from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_active_calibration import candidate_rows, selected_span_ratio, select_indices
from experiment_versioning import write_run_metadata
from run_sim_tcn import metric_row
from run_transfer_finetune import (
    eval_region,
    load_checkpoint,
    make_model,
    predict_rows,
    prepare_target_frame,
    split_adapt_eval,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "clean": (0.0, 0.0),
    "mild": (0.0025, 0.005),
    "moderate": (0.005, 0.010),
    "high": (0.010, 0.020),
    "severe": (0.020, 0.050),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate few-sample calibration under noisy reference labels.")
    parser.add_argument("--target-input", default="data/simulated/xray_belt_transfer_target.csv")
    parser.add_argument("--pretrained-checkpoint", default="models/pgt_sc_tcn_pretrained_v2.pt")
    parser.add_argument("--base-model", default="models/pgt_sc_base_model_v2.pkl")
    parser.add_argument("--output", default="results/sim_calibration_label_noise_v2.csv")
    parser.add_argument("--summary", default="results/sim_calibration_label_noise_summary.csv")
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    parser.add_argument("--budgets", default="20,60,180")
    parser.add_argument("--replicates", type=int, default=30)
    parser.add_argument("--min-pred-flow", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=20260629)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def attach_keys(pred_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    keys = ["site_id", "device_id", "install_id", "timestamp_s"]
    feature_df = df[keys].copy()
    feature_df["timestamp_s"] = feature_df["timestamp_s"].astype(float)
    out = pred_df.copy()
    out["timestamp_s"] = out["timestamp_s"].astype(float)
    return out.merge(feature_df, on=keys, how="left", validate="many_to_one")


def calibrate_with_noisy_labels(
    pred_df: pd.DataFrame,
    budget: int,
    adapt_fraction: float,
    min_pred_flow: float,
    systematic_std: float,
    random_std: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    calibrated = []
    spans = []
    selected_counts = []
    label_abs_pct_errors = []
    slope_values = []
    intercept_values = []

    for _, group in pred_df.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = group.sort_values("timestamp_s").copy()
        candidates = candidate_rows(g, adapt_fraction, min_pred_flow)
        local_idx = select_indices(candidates, budget, "prediction-stratified", rng)
        sample = candidates.iloc[local_idx].copy()

        if len(sample) >= 2:
            install_bias = rng.normal(0.0, systematic_std)
            true_y = sample["y_true"].to_numpy(float)
            point_noise = rng.normal(0.0, random_std, size=len(sample))
            noisy_y = np.maximum(true_y * (1.0 + install_bias + point_noise), 0.0)
            a, b = np.linalg.lstsq(
                np.column_stack([sample["y_pred"].to_numpy(float), np.ones(len(sample))]),
                noisy_y,
                rcond=None,
            )[0]
            label_abs_pct_errors.extend(np.abs(noisy_y - true_y) / np.maximum(np.abs(true_y), 1e-9) * 100)
            slope_values.append(float(a))
            intercept_values.append(float(b))
        else:
            a, b = 1.0, 0.0

        g["y_pred"] = np.maximum(a * g["y_pred"].to_numpy(float) + b, 0.0)
        calibrated.append(g)
        spans.append(selected_span_ratio(candidates, sample) if len(sample) else 0.0)
        selected_counts.append(len(sample))

    stats = {
        "selected_span_ratio": float(np.mean(spans)) if spans else 0.0,
        "selected_rows_per_install": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "label_abs_pct_error_mean": float(np.mean(label_abs_pct_errors)) if label_abs_pct_errors else 0.0,
        "calibration_slope_mean": float(np.mean(slope_values)) if slope_values else 1.0,
        "calibration_intercept_mean": float(np.mean(intercept_values)) if intercept_values else 0.0,
    }
    return pd.concat(calibrated, ignore_index=True), stats


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    summary_rows: list[dict[str, object]] = []
    keys = ["scenario", "systematic_std_pct", "random_std_pct", "label_budget_per_install"]
    for key, part in rows.groupby(keys, sort=False):
        scenario, systematic_std_pct, random_std_pct, budget = key
        summary_rows.append(
            {
                "scenario": scenario,
                "systematic_std_pct": systematic_std_pct,
                "random_std_pct": random_std_pct,
                "label_budget_per_install": budget,
                "replicates": int(len(part)),
                "mae_median": float(part["mae"].median()),
                "rmse_median": float(part["rmse"].median()),
                "cum_err_median_pct": float(part["cum_err_pct"].median()),
                "cum_err_q25_pct": float(part["cum_err_pct"].quantile(0.25)),
                "cum_err_q75_pct": float(part["cum_err_pct"].quantile(0.75)),
                "cum_err_p90_pct": float(part["cum_err_pct"].quantile(0.90)),
                "success_lt_1pct": float((part["cum_err_pct"] < 1.0).mean() * 100),
                "label_abs_pct_error_mean": float(part["label_abs_pct_error_mean"].mean()),
                "selected_span_ratio_mean": float(part["selected_span_ratio"].mean()),
            }
        )
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)

    target_input = resolve_project_path(args.target_input)
    checkpoint_path = resolve_project_path(args.pretrained_checkpoint)
    base_model_path = resolve_project_path(args.base_model)
    checkpoint = load_checkpoint(checkpoint_path)
    seq_len = int(checkpoint["model_config"]["seq_len"])
    source_mean = pd.Series(checkpoint["feature_mean"], dtype=float)
    source_std = pd.Series(checkpoint["feature_std"], dtype=float).replace(0, 1)
    source_y_mean = float(checkpoint["residual_mean"])
    source_y_std = float(checkpoint["residual_std"]) or 1.0

    df = prepare_target_frame(target_input, base_model_path)
    adapt_df, eval_df = split_adapt_eval(df, args.adapt_fraction)
    model = make_model(checkpoint).to(device)
    direct_pred = predict_rows(
        model,
        df,
        source_mean,
        source_std,
        source_y_mean,
        source_y_std,
        seq_len,
        args.batch_size,
        device,
    )
    pred_df = attach_keys(direct_pred, df)

    rows: list[dict[str, object]] = []
    for budget in parse_int_list(args.budgets):
        for scenario, (systematic_std, random_std) in SCENARIOS.items():
            for replicate in range(args.replicates):
                seed = args.seed + 1009 * replicate + 37 * budget + int(systematic_std * 1_000_000)
                cal_pred, stats = calibrate_with_noisy_labels(
                    pred_df,
                    budget,
                    args.adapt_fraction,
                    args.min_pred_flow,
                    systematic_std,
                    random_std,
                    seed,
                )
                y_true, y_pred = eval_region(cal_pred, args.adapt_fraction)
                row = metric_row("prediction-stratified-noisy-calibration", y_true, y_pred)
                row.update(
                    {
                        "scenario": scenario,
                        "replicate": replicate,
                        "label_budget_per_install": budget,
                        "systematic_std_pct": systematic_std * 100,
                        "random_std_pct": random_std * 100,
                        "target_rows": int(len(df)),
                        "adapt_rows": int(len(adapt_df)),
                        "eval_rows": int(len(eval_df)),
                        "seq_len": seq_len,
                        "adapt_fraction": args.adapt_fraction,
                        "device": str(device),
                    }
                )
                row.update(stats)
                rows.append(row)

    out = pd.DataFrame(rows)
    summary = summarize(out)

    output = Path(args.output)
    summary_path = Path(args.summary)
    if not output.is_absolute():
        output = ROOT / output
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    summary.to_csv(summary_path, index=False)
    metadata_kwargs = {
        "script": __file__,
        "args": vars(args),
        "inputs": [target_input, checkpoint_path, base_model_path],
        "outputs": [output, summary_path],
        "notes": "Few-sample calibration robustness under noisy reference weighing labels.",
    }
    write_run_metadata(
        output,
        **metadata_kwargs,
    )
    write_run_metadata(
        summary_path,
        **metadata_kwargs,
    )
    print(
        f"wrote={output.relative_to(ROOT)} summary={summary_path.relative_to(ROOT)} "
        f"rows={len(out)} elapsed_s={time.perf_counter() - started:.3f} device={device}"
    )


if __name__ == "__main__":
    main()
