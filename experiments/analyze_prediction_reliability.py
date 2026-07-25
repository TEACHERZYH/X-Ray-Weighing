from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from analyze_active_calibration import DIVERSITY_COLS, attach_features, candidate_rows, select_indices
from experiment_versioning import write_run_metadata
from run_sim_tcn import metric_row
from run_transfer_finetune import load_checkpoint, make_model, predict_rows, prepare_target_frame, split_adapt_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate prediction reliability calibration and recalibration triggers.")
    parser.add_argument("--target-input", default="data/simulated/xray_belt_transfer_target.csv")
    parser.add_argument("--pretrained-checkpoint", default="models/pgt_sc_tcn_pretrained_v2.pt")
    parser.add_argument("--base-model", default="models/pgt_sc_base_model_v2.pkl")
    parser.add_argument("--output", default="results/sim_prediction_reliability_results_v2.csv")
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    parser.add_argument("--budgets", default="20,60,180")
    parser.add_argument("--strategies", default="time-uniform,prediction-stratified")
    parser.add_argument("--interval-methods", default="global,install,flow-bin")
    parser.add_argument("--nominal-coverages", default="0.80,0.90,0.95")
    parser.add_argument("--min-pred-flow", type=float, default=5.0)
    parser.add_argument("--flow-bins", type=int, default=4)
    parser.add_argument("--drift-quantile", type=float, default=0.95)
    parser.add_argument("--risk-quantile", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_str_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def cutoff_for_group(g: pd.DataFrame, adapt_fraction: float) -> float:
    return float(g["timestamp_s"].min()) + adapt_fraction * (
        float(g["timestamp_s"].max()) - float(g["timestamp_s"].min())
    )


def robust_feature_scores(reference: pd.DataFrame, target: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not cols:
        return np.zeros(len(target)), np.zeros(len(reference))
    ref = reference[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    tar = target[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    center = ref.median(numeric_only=True)
    scale = (ref.quantile(0.75, numeric_only=True) - ref.quantile(0.25, numeric_only=True)).replace(0, np.nan)
    scale = scale.fillna(ref.std(numeric_only=True).replace(0, np.nan)).fillna(1.0)
    ref_z = ((ref.fillna(center) - center) / scale).to_numpy(float)
    tar_z = ((tar.fillna(center) - center) / scale).to_numpy(float)
    return np.sqrt((tar_z**2).sum(axis=1)), np.sqrt((ref_z**2).sum(axis=1))


def residual_quantile(values: np.ndarray, nominal: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.quantile(np.abs(values.astype(float)), nominal))


def fit_install_calibration(sample: pd.DataFrame) -> tuple[float, float]:
    if len(sample) < 2:
        return 1.0, 0.0
    a, b = np.linalg.lstsq(
        np.column_stack([sample["y_pred"].to_numpy(float), np.ones(len(sample))]),
        sample["y_true"].to_numpy(float),
        rcond=None,
    )[0]
    return float(a), float(b)


def prepare_calibrated_frame(
    pred_df: pd.DataFrame,
    strategy: str,
    budget: int,
    adapt_fraction: float,
    min_pred_flow: float,
    seed: int,
    risk_quantile: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    calibrated_groups = []
    calibration_samples = []
    feature_cols = [col for col in DIVERSITY_COLS if col in pred_df.columns]
    for install_key, g in pred_df.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = g.sort_values("timestamp_s").copy()
        cutoff = cutoff_for_group(g, adapt_fraction)
        candidates = candidate_rows(g, adapt_fraction, min_pred_flow)
        local_idx = select_indices(candidates, budget, strategy, rng)
        sample = candidates.iloc[local_idx].copy()
        a, b = fit_install_calibration(sample)
        g["y_cal"] = np.maximum(a * g["y_pred"].to_numpy(float) + b, 0.0)
        sample["y_cal"] = np.maximum(a * sample["y_pred"].to_numpy(float) + b, 0.0)
        sample["residual"] = sample["y_true"].to_numpy(float) - sample["y_cal"].to_numpy(float)
        g["correction_abs"] = np.abs(g["y_cal"].to_numpy(float) - g["y_pred"].to_numpy(float))
        sample["correction_abs"] = np.abs(sample["y_cal"].to_numpy(float) - sample["y_pred"].to_numpy(float))
        cal_lo = float(sample["y_cal"].min()) if len(sample) else float(g["y_cal"].min())
        cal_hi = float(sample["y_cal"].max()) if len(sample) else float(g["y_cal"].max())
        cal_range = max(cal_hi - cal_lo, 1e-6)
        g["flow_edge_score"] = np.maximum(
            (cal_lo - g["y_cal"].to_numpy(float)) / cal_range,
            (g["y_cal"].to_numpy(float) - cal_hi) / cal_range,
        )
        g["flow_edge_score"] = np.maximum(g["flow_edge_score"].to_numpy(float), 0.0)
        sample["is_calibration_sample"] = True
        sample["adapt_cutoff"] = cutoff
        ref = g[g["timestamp_s"] <= cutoff].copy()
        if len(ref) == 0:
            ref = candidates.copy()
        target_scores, ref_scores = robust_feature_scores(ref, g, feature_cols)
        g["feature_drift_score"] = target_scores
        g["feature_drift_threshold"] = float(np.quantile(ref_scores, 0.95)) if len(ref_scores) else np.inf
        g["correction_threshold"] = float(np.quantile(ref["correction_abs"].to_numpy(float), risk_quantile))
        g["calibration_a"] = a
        g["calibration_b"] = b
        for col, value in zip(["site_id", "device_id", "install_id"], install_key):
            sample[col] = value
        calibrated_groups.append(g)
        calibration_samples.append(sample)
    return pd.concat(calibrated_groups, ignore_index=True), pd.concat(calibration_samples, ignore_index=True)


def assign_flow_bin_half_widths(
    eval_df: pd.DataFrame,
    samples: pd.DataFrame,
    install_q: dict[tuple[str, str, str], float],
    nominal: float,
    flow_bins: int,
) -> np.ndarray:
    half_width = np.zeros(len(eval_df), dtype=float)
    for key, idx in eval_df.groupby(["site_id", "device_id", "install_id"], sort=False).groups.items():
        sample = samples[
            (samples["site_id"] == key[0]) & (samples["device_id"] == key[1]) & (samples["install_id"] == key[2])
        ].copy()
        fallback = install_q.get(key, residual_quantile(samples["residual"].to_numpy(float), nominal))
        if len(sample) < max(flow_bins * 2, 8) or sample["y_cal"].nunique() < 3:
            half_width[list(idx)] = fallback
            continue
        quantiles = np.linspace(0, 1, flow_bins + 1)
        edges = np.unique(np.quantile(sample["y_cal"].to_numpy(float), quantiles))
        if len(edges) <= 2:
            half_width[list(idx)] = fallback
            continue
        sample_bins = np.digitize(sample["y_cal"].to_numpy(float), edges[1:-1], right=True)
        bin_q = {}
        for bin_id in range(len(edges) - 1):
            vals = sample.loc[sample_bins == bin_id, "residual"].to_numpy(float)
            bin_q[bin_id] = residual_quantile(vals, nominal) if len(vals) >= 3 else fallback
        eval_vals = eval_df.loc[list(idx), "y_cal"].to_numpy(float)
        eval_bins = np.digitize(eval_vals, edges[1:-1], right=True)
        half_width[list(idx)] = np.array([bin_q.get(int(bin_id), fallback) for bin_id in eval_bins], dtype=float)
    return half_width


def evaluate_interval(
    eval_df: pd.DataFrame,
    samples: pd.DataFrame,
    interval_method: str,
    nominal: float,
    flow_bins: int,
) -> dict[str, object]:
    y_true = eval_df["y_true"].to_numpy(float)
    y_pred = eval_df["y_cal"].to_numpy(float)
    abs_error = np.abs(y_true - y_pred)
    global_q = residual_quantile(samples["residual"].to_numpy(float), nominal)
    install_q = {
        key: residual_quantile(g["residual"].to_numpy(float), nominal)
        for key, g in samples.groupby(["site_id", "device_id", "install_id"], sort=False)
    }
    if interval_method == "global":
        half_width = np.full(len(eval_df), global_q, dtype=float)
    elif interval_method == "install":
        half_width = np.array(
            [
                install_q.get((row.site_id, row.device_id, row.install_id), global_q)
                for row in eval_df[["site_id", "device_id", "install_id"]].itertuples(index=False)
            ],
            dtype=float,
        )
    elif interval_method == "flow-bin":
        half_width = assign_flow_bin_half_widths(eval_df, samples, install_q, nominal, flow_bins)
    else:
        raise ValueError(f"unknown interval method: {interval_method}")

    covered = abs_error <= half_width
    width_threshold = np.quantile(half_width, 0.75)
    interval_trigger = half_width > width_threshold
    drift_trigger = eval_df["feature_drift_score"].to_numpy(float) > eval_df["feature_drift_threshold"].to_numpy(float)
    joint_trigger = interval_trigger | drift_trigger
    correction_trigger = eval_df["correction_abs"].to_numpy(float) > eval_df["correction_threshold"].to_numpy(float)
    flow_edge_trigger = eval_df["flow_edge_score"].to_numpy(float) > 0
    risk_trigger = correction_trigger | flow_edge_trigger
    top10_threshold = np.quantile(abs_error, 0.90)
    top10 = abs_error >= top10_threshold

    base = metric_row("calibrated-eval", y_true, y_pred)
    triggered = metric_row("triggered", y_true[joint_trigger], y_pred[joint_trigger]) if joint_trigger.any() else {}
    non_triggered = (
        metric_row("non-triggered", y_true[~joint_trigger], y_pred[~joint_trigger]) if (~joint_trigger).any() else {}
    )
    risk_triggered = metric_row("risk-triggered", y_true[risk_trigger], y_pred[risk_trigger]) if risk_trigger.any() else {}
    risk_non_triggered = (
        metric_row("risk-non-triggered", y_true[~risk_trigger], y_pred[~risk_trigger]) if (~risk_trigger).any() else {}
    )
    return {
        "mae": base["mae"],
        "rmse": base["rmse"],
        "cum_err_pct": base["cum_err_pct"],
        "coverage_pct": float(covered.mean() * 100),
        "mean_half_width_kgps": float(np.mean(half_width)),
        "median_half_width_kgps": float(np.median(half_width)),
        "miss_mae": float(abs_error[~covered].mean()) if (~covered).any() else 0.0,
        "trigger_rate_pct": float(joint_trigger.mean() * 100),
        "interval_trigger_rate_pct": float(interval_trigger.mean() * 100),
        "drift_trigger_rate_pct": float(drift_trigger.mean() * 100),
        "capture_top10_error_pct": float((joint_trigger & top10).sum() / max(top10.sum(), 1) * 100),
        "triggered_mae": triggered.get("mae", 0.0),
        "non_triggered_mae": non_triggered.get("mae", 0.0),
        "triggered_cum_err_pct": triggered.get("cum_err_pct", 0.0),
        "non_triggered_cum_err_pct": non_triggered.get("cum_err_pct", 0.0),
        "risk_enrichment": float(triggered.get("mae", 0.0) / max(non_triggered.get("mae", 0.0), 1e-9)),
        "correction_trigger_rate_pct": float(correction_trigger.mean() * 100),
        "flow_edge_trigger_rate_pct": float(flow_edge_trigger.mean() * 100),
        "risk_trigger_rate_pct": float(risk_trigger.mean() * 100),
        "risk_capture_top10_error_pct": float((risk_trigger & top10).sum() / max(top10.sum(), 1) * 100),
        "risk_triggered_mae": risk_triggered.get("mae", 0.0),
        "risk_non_triggered_mae": risk_non_triggered.get("mae", 0.0),
        "risk_triggered_cum_err_pct": risk_triggered.get("cum_err_pct", 0.0),
        "risk_non_triggered_cum_err_pct": risk_non_triggered.get("cum_err_pct", 0.0),
        "risk_mae_enrichment": float(
            risk_triggered.get("mae", 0.0) / max(risk_non_triggered.get("mae", 0.0), 1e-9)
        ),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    torch.set_num_threads(args.threads)
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    checkpoint = load_checkpoint(Path(args.pretrained_checkpoint))
    model = make_model(checkpoint).to(device)
    source_mean = pd.Series(checkpoint["feature_mean"], dtype=float)
    source_std = pd.Series(checkpoint["feature_std"], dtype=float).replace(0, 1)
    source_y_mean = float(checkpoint["residual_mean"])
    source_y_std = float(checkpoint["residual_std"]) or 1.0
    seq_len = int(checkpoint["model_config"]["seq_len"])

    df = prepare_target_frame(Path(args.target_input), Path(args.base_model))
    adapt_df, eval_df_raw = split_adapt_eval(df, args.adapt_fraction)
    pred_df = predict_rows(model, df, source_mean, source_std, source_y_mean, source_y_std, seq_len, args.batch_size, device)
    pred_df = attach_features(pred_df, df)

    rows: list[dict[str, object]] = []
    for strategy in parse_str_list(args.strategies):
        for budget in parse_int_list(args.budgets):
            calibrated_df, samples = prepare_calibrated_frame(
                pred_df,
                strategy=strategy,
                budget=budget,
                adapt_fraction=args.adapt_fraction,
                min_pred_flow=args.min_pred_flow,
                seed=args.seed,
                risk_quantile=args.risk_quantile,
            )
            eval_parts = []
            for _, g in calibrated_df.groupby(["site_id", "device_id", "install_id"], sort=False):
                eval_parts.append(g[g["timestamp_s"] > cutoff_for_group(g, args.adapt_fraction)])
            eval_df = pd.concat(eval_parts, ignore_index=True)
            for nominal in parse_float_list(args.nominal_coverages):
                for interval_method in parse_str_list(args.interval_methods):
                    metrics = evaluate_interval(eval_df, samples, interval_method, nominal, args.flow_bins)
                    rows.append(
                        {
                            "strategy": strategy,
                            "label_budget_per_install": budget,
                            "interval_method": interval_method,
                            "nominal_coverage_pct": nominal * 100,
                            **metrics,
                            "selected_rows": int(len(samples)),
                            "target_rows": int(len(df)),
                            "adapt_rows": int(len(adapt_df)),
                            "eval_rows": int(len(eval_df_raw)),
                            "seq_len": seq_len,
                            "adapt_fraction": args.adapt_fraction,
                            "device": str(device),
                        }
                    )

    out = pd.DataFrame(rows)
    out["elapsed_s"] = time.time() - started
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    write_run_metadata(
        out_path,
        script="experiments/analyze_prediction_reliability.py",
        args=vars(args),
        inputs=[args.target_input, args.pretrained_checkpoint, args.base_model],
        outputs=[out_path],
        notes="Prediction interval calibration and high-risk recalibration trigger simulation.",
    )
    print(f"wrote={out_path} rows={len(out)} device={device} elapsed_s={time.time() - started:.1f}")


if __name__ == "__main__":
    main()
