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
from run_transfer_finetune import load_checkpoint, make_model, predict_rows, prepare_target_frame


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate triggered recalibration with additional reference labels.")
    parser.add_argument("--target-input", default="data/simulated/xray_belt_transfer_target.csv")
    parser.add_argument("--pretrained-checkpoint", default="models/pgt_sc_tcn_pretrained_v2.pt")
    parser.add_argument("--base-model", default="models/pgt_sc_base_model_v2.pkl")
    parser.add_argument("--output", default="results/sim_triggered_recalibration_v2.csv")
    parser.add_argument("--summary", default="results/sim_triggered_recalibration_summary.csv")
    parser.add_argument("--initial-budget", type=int, default=20)
    parser.add_argument("--extra-budgets", default="5,10,20")
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    parser.add_argument("--trigger-fraction", type=float, default=0.55)
    parser.add_argument("--min-pred-flow", type=float, default=5.0)
    parser.add_argument("--random-seeds", default="1,7,11,23,37")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=31)
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def split_cutoffs(g: pd.DataFrame, adapt_fraction: float, trigger_fraction: float) -> tuple[float, float]:
    t_min = float(g["timestamp_s"].min())
    span = float(g["timestamp_s"].max()) - t_min
    return t_min + adapt_fraction * span, t_min + trigger_fraction * span


def fit_calibration(sample: pd.DataFrame) -> tuple[float, float]:
    if len(sample) < 2:
        return 1.0, 0.0
    a, b = np.linalg.lstsq(
        np.column_stack([sample["y_pred"].to_numpy(float), np.ones(len(sample))]),
        sample["y_true"].to_numpy(float),
        rcond=None,
    )[0]
    return float(a), float(b)


def apply_calibration(g: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    a, b = fit_calibration(sample)
    out = g.copy()
    out["y_cal"] = np.maximum(a * out["y_pred"].to_numpy(float) + b, 0.0)
    out["calibration_a"] = a
    out["calibration_b"] = b
    return out


def robust_feature_score(reference: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    cols = [col for col in DIVERSITY_COLS if col in reference.columns and col in target.columns]
    if not cols:
        return np.zeros(len(target), dtype=float)
    ref = reference[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    tar = target[cols].astype(float).replace([np.inf, -np.inf], np.nan)
    center = ref.median(numeric_only=True)
    scale = (ref.quantile(0.75, numeric_only=True) - ref.quantile(0.25, numeric_only=True)).replace(0, np.nan)
    scale = scale.fillna(ref.std(numeric_only=True).replace(0, np.nan)).fillna(1.0)
    z = ((tar.fillna(center) - center) / scale).to_numpy(float)
    return np.sqrt((z**2).sum(axis=1))


def selected_span_ratio(candidates: pd.DataFrame, sample: pd.DataFrame) -> float:
    if len(candidates) == 0 or len(sample) == 0:
        return 0.0
    denom = float(candidates["y_pred"].max() - candidates["y_pred"].min())
    if denom <= 1e-9:
        return 1.0
    return float((sample["y_pred"].max() - sample["y_pred"].min()) / denom)


def risk_score(trigger_pool: pd.DataFrame, initial_sample: pd.DataFrame, adapt_ref: pd.DataFrame) -> np.ndarray:
    cal_lo = float(initial_sample["y_cal"].min()) if len(initial_sample) else float(trigger_pool["y_cal_initial"].min())
    cal_hi = float(initial_sample["y_cal"].max()) if len(initial_sample) else float(trigger_pool["y_cal_initial"].max())
    cal_range = max(cal_hi - cal_lo, 1e-6)
    y_cal = trigger_pool["y_cal_initial"].to_numpy(float)
    edge_score = np.maximum((cal_lo - y_cal) / cal_range, (y_cal - cal_hi) / cal_range)
    edge_score = np.maximum(edge_score, 0.0)
    correction = np.abs(trigger_pool["y_cal_initial"].to_numpy(float) - trigger_pool["y_pred"].to_numpy(float))
    correction_score = correction / max(float(np.quantile(correction, 0.90)), 1e-6)
    feature_score = robust_feature_score(adapt_ref, trigger_pool)
    feature_score = feature_score / max(float(np.quantile(feature_score, 0.90)), 1e-6)
    return edge_score + 0.35 * correction_score + 0.35 * feature_score


def select_extra_sample(
    trigger_pool: pd.DataFrame,
    strategy: str,
    budget: int,
    rng: np.random.Generator,
    scores: np.ndarray,
) -> pd.DataFrame:
    n = min(int(budget), len(trigger_pool))
    if n <= 0:
        return trigger_pool.iloc[[]].copy()
    base_strategy = strategy.replace("-gated", "").replace("-anchor", "")
    if base_strategy == "random-extra":
        idx = np.sort(rng.choice(len(trigger_pool), size=n, replace=False))
    elif base_strategy == "risk-triggered-extra":
        idx = np.argsort(scores)[-n:]
        idx = np.sort(idx)
    elif base_strategy == "risk-covered-extra":
        y_pred = trigger_pool["y_pred"].to_numpy(float)
        quantiles = np.linspace(0.0, 1.0, min(n, 8) + 1)
        edges = np.unique(np.quantile(y_pred, quantiles))
        selected: list[int] = []
        if len(edges) > 2:
            for lo, hi in zip(edges[:-1], edges[1:]):
                in_bin = np.flatnonzero((y_pred >= lo) & (y_pred <= hi))
                if len(in_bin) == 0:
                    continue
                selected.append(int(in_bin[np.argmax(scores[in_bin])]))
        remaining = [int(i) for i in np.argsort(scores)[::-1] if int(i) not in set(selected)]
        idx = np.array((selected + remaining)[:n], dtype=int)
        idx = np.sort(idx)
    elif base_strategy == "oracle-error-extra":
        error = np.abs(trigger_pool["y_true"].to_numpy(float) - trigger_pool["y_cal_initial"].to_numpy(float))
        idx = np.argsort(error)[-n:]
        idx = np.sort(idx)
    else:
        raise ValueError(f"unknown extra selection strategy: {strategy}")
    return trigger_pool.iloc[idx].copy()


def mass_error_pct(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = max(abs(float(np.sum(y_true))), 1e-9)
    return abs(float(np.sum(y_pred) - np.sum(y_true))) / denom * 100.0


def candidate_passes_mass_gate(extra_sample: pd.DataFrame, a: float, b: float) -> bool:
    if len(extra_sample) < 2:
        return False
    y_true = extra_sample["y_true"].to_numpy(float)
    initial = extra_sample["y_cal_initial"].to_numpy(float)
    candidate = np.maximum(a * extra_sample["y_pred"].to_numpy(float) + b, 0.0)
    return mass_error_pct(y_true, candidate) <= mass_error_pct(y_true, initial)


def apply_mass_anchor(initial_cal: pd.DataFrame, extra_sample: pd.DataFrame, max_adjust_pct: float = 5.0) -> pd.DataFrame:
    out = initial_cal.copy()
    if len(extra_sample) < 2:
        out["mass_anchor_factor"] = 1.0
        return out
    observed = float(extra_sample["y_true"].sum())
    estimated = float(extra_sample["y_cal_initial"].sum())
    factor = observed / max(estimated, 1e-9)
    bound = float(max_adjust_pct) / 100.0
    factor = float(np.clip(factor, 1.0 - bound, 1.0 + bound))
    out["y_cal"] = np.maximum(out["y_cal"].to_numpy(float) * factor, 0.0)
    out["mass_anchor_factor"] = factor
    return out


def install_cum_errors(eval_df: pd.DataFrame) -> list[float]:
    values: list[float] = []
    for _, g in eval_df.groupby(["site_id", "device_id", "install_id"], sort=False):
        denom = max(abs(float(g["y_true"].sum())), 1e-9)
        values.append(abs(float(g["y_cal"].sum() - g["y_true"].sum())) / denom * 100.0)
    return values


def evaluate(eval_df: pd.DataFrame, method: str, extra: dict[str, object]) -> dict[str, object]:
    row = metric_row(method, eval_df["y_true"].to_numpy(float), eval_df["y_cal"].to_numpy(float))
    install_errors = install_cum_errors(eval_df)
    row.update(
        {
            "install_cum_err_median_pct": float(np.median(install_errors)) if install_errors else 0.0,
            "install_success_lt1_pct": float(np.mean(np.array(install_errors) < 1.0) * 100.0) if install_errors else 0.0,
            "install_count": len(install_errors),
        }
    )
    row.update(extra)
    return row


def run_strategy(
    pred_df: pd.DataFrame,
    strategy: str,
    initial_budget: int,
    extra_budget: int,
    adapt_fraction: float,
    trigger_fraction: float,
    min_pred_flow: float,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    eval_parts = []
    selected_spans = []
    risk_scores = []
    selected_counts = []
    candidate_counts = []
    high_initial_error_rates = []
    no_update_eval_parts = []
    accepted_updates = []

    for _, g0 in pred_df.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = g0.sort_values("timestamp_s").copy()
        adapt_cutoff, trigger_cutoff = split_cutoffs(g, adapt_fraction, trigger_fraction)
        initial_candidates = candidate_rows(g, adapt_fraction, min_pred_flow)
        initial_idx = select_indices(initial_candidates, initial_budget, "prediction-stratified", rng)
        initial_sample = initial_candidates.iloc[initial_idx].copy()
        initial_cal = apply_calibration(g, initial_sample)
        initial_sample = initial_cal.loc[initial_sample.index].copy()
        initial_sample["y_cal"] = initial_sample["y_cal"].to_numpy(float)

        trigger_pool = initial_cal[
            (initial_cal["timestamp_s"] > adapt_cutoff)
            & (initial_cal["timestamp_s"] <= trigger_cutoff)
            & (initial_cal["y_cal"] >= min_pred_flow)
        ].copy()
        if len(trigger_pool) < 2:
            trigger_pool = initial_cal[
                (initial_cal["timestamp_s"] > adapt_cutoff) & (initial_cal["timestamp_s"] <= trigger_cutoff)
            ].copy()
        trigger_pool["y_cal_initial"] = trigger_pool["y_cal"].to_numpy(float)
        adapt_ref = initial_cal[initial_cal["timestamp_s"] <= adapt_cutoff].copy()
        scores = risk_score(trigger_pool, initial_sample, adapt_ref) if len(trigger_pool) else np.array([], dtype=float)

        if strategy == "no-update":
            extra_sample = trigger_pool.iloc[[]].copy()
            final_cal = initial_cal
            accepted_update = False
        else:
            extra_sample = select_extra_sample(trigger_pool, strategy, extra_budget, rng, scores)
            accepted_update = True
            if strategy.endswith("-anchor-extra"):
                final_cal = apply_mass_anchor(initial_cal, extra_sample)
            else:
                combined = pd.concat(
                    [initial_sample[["y_pred", "y_true"]], extra_sample[["y_pred", "y_true"]]],
                    ignore_index=True,
                )
                candidate_a, candidate_b = fit_calibration(combined)
                if strategy.endswith("-gated-extra"):
                    accepted_update = candidate_passes_mass_gate(extra_sample, candidate_a, candidate_b)
                final_cal = apply_calibration(g, combined) if accepted_update else initial_cal

        eval_part = final_cal[final_cal["timestamp_s"] > trigger_cutoff].copy()
        no_update_eval = initial_cal[initial_cal["timestamp_s"] > trigger_cutoff].copy()
        eval_parts.append(eval_part)
        no_update_eval_parts.append(no_update_eval)
        selected_counts.append(len(extra_sample))
        candidate_counts.append(len(trigger_pool))
        selected_spans.append(selected_span_ratio(trigger_pool, extra_sample))
        accepted_updates.append(float(accepted_update))
        if len(extra_sample) and len(scores):
            selected_positions = trigger_pool.index.get_indexer(extra_sample.index)
            risk_scores.extend(scores[selected_positions].tolist())
            abs_error = np.abs(trigger_pool["y_true"].to_numpy(float) - trigger_pool["y_cal_initial"].to_numpy(float))
            high_threshold = np.quantile(abs_error, 0.90)
            high_initial_error_rates.append(float((np.abs(extra_sample["y_true"] - extra_sample["y_cal_initial"]) >= high_threshold).mean() * 100.0))
        else:
            high_initial_error_rates.append(0.0)

    eval_df = pd.concat(eval_parts, ignore_index=True)
    no_update_df = pd.concat(no_update_eval_parts, ignore_index=True)
    base_cum = abs(float(no_update_df["y_cal"].sum() - no_update_df["y_true"].sum())) / max(abs(float(no_update_df["y_true"].sum())), 1e-9) * 100.0
    extra = {
        "initial_budget_per_install": initial_budget,
        "extra_budget_per_install": extra_budget,
        "selected_rows_per_install": float(np.mean(selected_counts)) if selected_counts else 0.0,
        "candidate_rows_per_install": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "selected_span_ratio": float(np.mean(selected_spans)) if selected_spans else 0.0,
        "selected_risk_score_mean": float(np.mean(risk_scores)) if risk_scores else 0.0,
        "selected_top10_initial_error_pct": float(np.mean(high_initial_error_rates)) if high_initial_error_rates else 0.0,
        "accepted_update_pct": float(np.mean(accepted_updates) * 100.0) if accepted_updates else 0.0,
        "cum_err_reduction_vs_no_update_pct": 0.0,
        "adapt_fraction": adapt_fraction,
        "trigger_fraction": trigger_fraction,
        "seed": seed,
    }
    row = evaluate(eval_df, strategy, extra)
    row["cum_err_reduction_vs_no_update_pct"] = float((base_cum - row["cum_err_pct"]) / max(base_cum, 1e-9) * 100.0)
    return row


def write_outputs(rows: list[dict[str, object]], output: Path, summary: Path) -> None:
    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    grouped = []
    for (method, budget), g in df.groupby(["method", "extra_budget_per_install"], sort=False):
        grouped.append(
            {
                "method": method,
                "extra_budget_per_install": budget,
                "runs": len(g),
                "mae_mean": float(g["mae"].mean()),
                "rmse_mean": float(g["rmse"].mean()),
                "cum_err_mean_pct": float(g["cum_err_pct"].mean()),
                "cum_err_std_pct": float(g["cum_err_pct"].std(ddof=0)),
                "install_success_lt1_mean_pct": float(g["install_success_lt1_pct"].mean()),
                "cum_err_reduction_vs_no_update_mean_pct": float(g["cum_err_reduction_vs_no_update_pct"].mean()),
                "selected_top10_initial_error_mean_pct": float(g["selected_top10_initial_error_pct"].mean()),
                "accepted_update_mean_pct": float(g["accepted_update_pct"].mean()),
                "selected_span_ratio_mean": float(g["selected_span_ratio"].mean()),
            }
        )
    pd.DataFrame(grouped).to_csv(summary, index=False)


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    device = torch.device(
        "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    )

    checkpoint = load_checkpoint(ROOT / args.pretrained_checkpoint)
    model = make_model(checkpoint).to(device)
    seq_len = int(checkpoint["model_config"]["seq_len"])
    source_mean = pd.Series(checkpoint["feature_mean"], dtype=float)
    source_std = pd.Series(checkpoint["feature_std"], dtype=float).replace(0, 1)
    source_y_mean = float(checkpoint["residual_mean"])
    source_y_std = float(checkpoint["residual_std"]) or 1.0

    df = prepare_target_frame(ROOT / args.target_input, ROOT / args.base_model)
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
    pred_df = attach_features(direct_pred, df)

    rows: list[dict[str, object]] = [
        run_strategy(
            pred_df,
            "no-update",
            args.initial_budget,
            0,
            args.adapt_fraction,
            args.trigger_fraction,
            args.min_pred_flow,
            args.seed,
        )
    ]
    for budget in parse_int_list(args.extra_budgets):
        for strategy in ["random-extra", "random-gated-extra"]:
            for seed in parse_int_list(args.random_seeds):
                rows.append(
                    run_strategy(
                        pred_df,
                        strategy,
                        args.initial_budget,
                        budget,
                        args.adapt_fraction,
                        args.trigger_fraction,
                        args.min_pred_flow,
                        seed,
                    )
                )
        for strategy in [
            "risk-triggered-extra",
            "risk-triggered-gated-extra",
            "risk-covered-extra",
            "risk-covered-anchor-extra",
            "oracle-error-extra",
        ]:
            rows.append(
                run_strategy(
                    pred_df,
                    strategy,
                    args.initial_budget,
                    budget,
                    args.adapt_fraction,
                    args.trigger_fraction,
                    args.min_pred_flow,
                    args.seed,
                )
            )

    output = ROOT / args.output
    summary = ROOT / args.summary
    write_outputs(rows, output, summary)
    elapsed = time.perf_counter() - started
    meta = write_run_metadata(
        output,
        script=__file__,
        args={**vars(args), "elapsed_s": elapsed, "device": str(device), "seq_len": seq_len},
        inputs=[ROOT / args.target_input, ROOT / args.pretrained_checkpoint, ROOT / args.base_model],
        outputs=[output, summary],
        notes="Triggered recalibration closed-loop simulation using initial prediction-stratified calibration plus extra reference labels.",
    )
    summary_meta = write_run_metadata(
        summary,
        script=__file__,
        args={**vars(args), "elapsed_s": elapsed, "device": str(device), "seq_len": seq_len},
        inputs=[output],
        outputs=[summary],
        notes="Summary of triggered recalibration closed-loop simulation.",
    )
    print(f"wrote={output}")
    print(f"wrote={summary}")
    print(f"wrote={meta}")
    print(f"wrote={summary_meta}")
    print(f"rows={len(rows)}")
    print(f"elapsed_s={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
