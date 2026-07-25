from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiment_versioning import write_run_metadata
from run_sim_tcn import FEATURE_COLS, metric_row
from run_transfer_finetune import (
    eval_region,
    load_checkpoint,
    make_model,
    predict_rows,
    prepare_target_frame,
    split_adapt_eval,
)


DIVERSITY_COLS = [
    "y_pred",
    "attenuation",
    "phys_flow_feature",
    "rx_value",
    "tube_voltage_kv",
    "tube_current_ma",
    "source_temp_c",
    "env_temp_c",
    "env_humidity_pct",
    "belt_speed_mps",
    "ash_missing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare active calibration sample selection strategies.")
    parser.add_argument("--target-input", default="data/simulated/xray_belt_transfer_target.csv")
    parser.add_argument("--pretrained-checkpoint", default="models/pgt_sc_tcn_pretrained_v2.pt")
    parser.add_argument("--base-model", default="models/pgt_sc_base_model_v2.pkl")
    parser.add_argument("--output", default="results/sim_active_calibration_results_v2.csv")
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    parser.add_argument("--budgets", default="10,20,60,180")
    parser.add_argument("--min-pred-flow", type=float, default=5.0)
    parser.add_argument("--random-seeds", default="1,7,11,23,37")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def attach_features(pred_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    keys = ["site_id", "device_id", "install_id", "timestamp_s"]
    feature_df = df[keys + [col for col in FEATURE_COLS if col in df.columns]].copy()
    out = pred_df.copy()
    out["timestamp_s"] = out["timestamp_s"].astype(float)
    feature_df["timestamp_s"] = feature_df["timestamp_s"].astype(float)
    return out.merge(feature_df, on=keys, how="left", validate="many_to_one")


def candidate_rows(g: pd.DataFrame, adapt_fraction: float, min_pred_flow: float) -> pd.DataFrame:
    cutoff = float(g["timestamp_s"].min()) + adapt_fraction * (
        float(g["timestamp_s"].max()) - float(g["timestamp_s"].min())
    )
    adapt = g[g["timestamp_s"] <= cutoff].sort_values("timestamp_s").copy()
    loaded = adapt[adapt["y_pred"] >= min_pred_flow]
    return loaded if len(loaded) >= 2 else adapt


def select_indices(candidates: pd.DataFrame, budget: int, strategy: str, rng: np.random.Generator) -> np.ndarray:
    n = min(int(budget), len(candidates))
    if n <= 0:
        return np.array([], dtype=int)
    if n >= len(candidates):
        return np.arange(len(candidates), dtype=int)

    if strategy == "head-contiguous":
        return np.arange(n, dtype=int)

    if strategy == "time-uniform":
        return np.linspace(0, len(candidates) - 1, n).astype(int)

    if strategy == "random":
        return np.sort(rng.choice(len(candidates), size=n, replace=False))

    if strategy == "prediction-stratified":
        order = np.argsort(candidates["y_pred"].to_numpy(float))
        take = np.linspace(0, len(order) - 1, n).astype(int)
        return np.sort(order[take])

    if strategy == "feature-diverse":
        cols = [col for col in DIVERSITY_COLS if col in candidates.columns]
        x = candidates[cols].astype(float).replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median(numeric_only=True)).fillna(0.0).to_numpy(float)
        x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x.std(axis=0, keepdims=True), 1e-6)
        if "y_pred" in cols:
            x[:, cols.index("y_pred")] *= 2.0
        center = np.median(x, axis=0, keepdims=True)
        selected = [int(np.argmin(((x - center) ** 2).sum(axis=1)))]
        min_dist = ((x - x[selected[0]]) ** 2).sum(axis=1)
        while len(selected) < n:
            min_dist[selected] = -1.0
            nxt = int(np.argmax(min_dist))
            selected.append(nxt)
            min_dist = np.minimum(min_dist, ((x - x[nxt]) ** 2).sum(axis=1))
        return np.sort(np.array(selected, dtype=int))

    raise ValueError(f"unknown strategy: {strategy}")


def selected_span_ratio(candidates: pd.DataFrame, sample: pd.DataFrame) -> float:
    denom = float(candidates["y_pred"].max() - candidates["y_pred"].min())
    if denom <= 1e-9:
        return 1.0
    return float((sample["y_pred"].max() - sample["y_pred"].min()) / denom)


def calibrate_with_strategy(
    pred_df: pd.DataFrame,
    strategy: str,
    budget: int,
    adapt_fraction: float,
    min_pred_flow: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(seed)
    calibrated = []
    spans = []
    selected_counts = []
    candidate_counts = []
    for _, g in pred_df.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = g.sort_values("timestamp_s").copy()
        candidates = candidate_rows(g, adapt_fraction, min_pred_flow)
        local_idx = select_indices(candidates, budget, strategy, rng)
        sample = candidates.iloc[local_idx]
        if len(sample) >= 2:
            a, b = np.linalg.lstsq(
                np.column_stack([sample["y_pred"].to_numpy(float), np.ones(len(sample))]),
                sample["y_true"].to_numpy(float),
                rcond=None,
            )[0]
        else:
            a, b = 1.0, 0.0
        g["y_pred"] = np.maximum(a * g["y_pred"].to_numpy(float) + b, 0.0)
        calibrated.append(g)
        spans.append(selected_span_ratio(candidates, sample) if len(sample) else 0.0)
        selected_counts.append(len(sample))
        candidate_counts.append(len(candidates))
    stats = {
        "selected_span_ratio": float(np.mean(spans)) if spans else 0.0,
        "selected_rows_per_install": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "candidate_rows_per_install": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
    }
    return pd.concat(calibrated, ignore_index=True), stats


def append_strategy_row(
    rows: list[dict[str, object]],
    method: str,
    pred_df: pd.DataFrame,
    budget: int,
    adapt_fraction: float,
    stats: dict[str, float],
    random_runs: int = 0,
    stds: dict[str, float] | None = None,
) -> None:
    y_true, y_pred = eval_region(pred_df, adapt_fraction)
    row = metric_row(method, y_true, y_pred)
    row.update(
        {
            "label_budget_per_install": budget,
            "selected_span_ratio": stats["selected_span_ratio"],
            "selected_rows_per_install": stats["selected_rows_per_install"],
            "candidate_rows_per_install": stats["candidate_rows_per_install"],
            "random_runs": random_runs,
        }
    )
    if stds:
        row.update(stds)
    rows.append(row)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)

    checkpoint = load_checkpoint(Path(args.pretrained_checkpoint))
    seq_len = int(checkpoint["model_config"]["seq_len"])
    source_mean = pd.Series(checkpoint["feature_mean"], dtype=float)
    source_std = pd.Series(checkpoint["feature_std"], dtype=float).replace(0, 1)
    source_y_mean = float(checkpoint["residual_mean"])
    source_y_std = float(checkpoint["residual_std"]) or 1.0

    df = prepare_target_frame(Path(args.target_input), Path(args.base_model))
    adapt_df, eval_df = split_adapt_eval(df, args.adapt_fraction)
    model = make_model(checkpoint).to(device)
    direct_pred = predict_rows(model, df, source_mean, source_std, source_y_mean, source_y_std, seq_len, args.batch_size, device)
    pred_df = attach_features(direct_pred, df)

    rows: list[dict[str, object]] = []
    y_true, y_pred = eval_region(pred_df, args.adapt_fraction)
    direct = metric_row("direct-no-calibration", y_true, y_pred)
    direct.update(
        {
            "label_budget_per_install": 0,
            "selected_span_ratio": 0.0,
            "selected_rows_per_install": 0.0,
            "candidate_rows_per_install": 0.0,
            "random_runs": 0,
        }
    )
    rows.append(direct)

    budgets = parse_int_list(args.budgets)
    random_seeds = parse_int_list(args.random_seeds)
    strategies = ["head-contiguous", "time-uniform", "prediction-stratified", "feature-diverse"]
    for budget in budgets:
        for strategy in strategies:
            cal_pred, stats = calibrate_with_strategy(
                pred_df, strategy, budget, args.adapt_fraction, args.min_pred_flow, args.seed
            )
            append_strategy_row(rows, strategy, cal_pred, budget, args.adapt_fraction, stats)

        random_metrics = []
        random_stats = []
        for seed in random_seeds:
            cal_pred, stats = calibrate_with_strategy(
                pred_df, "random", budget, args.adapt_fraction, args.min_pred_flow, seed
            )
            y_true, y_pred = eval_region(cal_pred, args.adapt_fraction)
            random_metrics.append(metric_row("random", y_true, y_pred))
            random_stats.append(stats)
        metric_df = pd.DataFrame(random_metrics)
        stats_df = pd.DataFrame(random_stats)
        row = {
            "method": "random-mean",
            "mae": float(metric_df["mae"].mean()),
            "rmse": float(metric_df["rmse"].mean()),
            "cum_err_pct": float(metric_df["cum_err_pct"].mean()),
            "label_budget_per_install": budget,
            "selected_span_ratio": float(stats_df["selected_span_ratio"].mean()),
            "selected_rows_per_install": float(stats_df["selected_rows_per_install"].mean()),
            "candidate_rows_per_install": float(stats_df["candidate_rows_per_install"].mean()),
            "random_runs": len(random_seeds),
            "mae_std": float(metric_df["mae"].std(ddof=0)),
            "rmse_std": float(metric_df["rmse"].std(ddof=0)),
            "cum_err_pct_std": float(metric_df["cum_err_pct"].std(ddof=0)),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    out["target_rows"] = int(len(df))
    out["adapt_rows"] = int(len(adapt_df))
    out["eval_rows"] = int(len(eval_df))
    out["seq_len"] = seq_len
    out["adapt_fraction"] = args.adapt_fraction
    out["device"] = str(device)
    out["elapsed_s"] = round(time.perf_counter() - started, 3)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)
    write_run_metadata(
        output,
        script=__file__,
        args=vars(args),
        inputs=[args.target_input, args.pretrained_checkpoint, args.base_model],
        outputs=[output],
        notes="Active calibration sample selection for target-domain transfer simulation.",
    )
    print(f"wrote={output} rows={len(out)} device={device}")


if __name__ == "__main__":
    main()
