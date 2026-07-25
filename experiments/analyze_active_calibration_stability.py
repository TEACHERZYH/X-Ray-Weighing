from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiment_versioning import write_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize stability of active calibration selection.")
    parser.add_argument("--input", default="results/sim_active_calibration_results.csv")
    parser.add_argument("--output", default="results/sim_active_calibration_stability.csv")
    return parser.parse_args()


def verdict(delta_std: float, delta_pct: float) -> str:
    if delta_pct > 0 and delta_std >= 0.5:
        return "stable-improvement"
    if delta_pct > 0:
        return "weak-improvement"
    return "not-better-than-random"


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    df = pd.read_csv(input_path)
    rows: list[dict[str, object]] = []
    for budget in sorted(df["label_budget_per_install"].unique()):
        if int(budget) <= 0:
            continue
        subset = df[df["label_budget_per_install"] == budget]
        strat = subset[subset["method"] == "prediction-stratified"]
        rand = subset[subset["method"] == "random-mean"]
        if strat.empty or rand.empty:
            continue
        strat_row = strat.iloc[0]
        rand_row = rand.iloc[0]
        rand_std = float(rand_row["cum_err_pct_std"]) if pd.notna(rand_row["cum_err_pct_std"]) else 0.0
        delta = float(rand_row["cum_err_pct"] - strat_row["cum_err_pct"])
        rel = 100.0 * delta / max(float(rand_row["cum_err_pct"]), 1e-9)
        delta_std = delta / rand_std if rand_std > 0 else float("nan")
        rows.append(
            {
                "label_budget_per_install": int(budget),
                "prediction_stratified_cum_err_pct": float(strat_row["cum_err_pct"]),
                "random_mean_cum_err_pct": float(rand_row["cum_err_pct"]),
                "random_cum_err_pct_std": rand_std,
                "cum_err_reduction_pct_point": delta,
                "cum_err_relative_reduction_pct": rel,
                "reduction_in_random_std": delta_std,
                "prediction_stratified_span_ratio": float(strat_row["selected_span_ratio"]),
                "random_mean_span_ratio": float(rand_row["selected_span_ratio"]),
                "span_ratio_gain": float(strat_row["selected_span_ratio"] - rand_row["selected_span_ratio"]),
                "random_runs": int(rand_row["random_runs"]),
                "verdict": verdict(delta_std, rel),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
    write_run_metadata(
        output_path,
        script="experiments/analyze_active_calibration_stability.py",
        args=vars(args),
        inputs=[input_path],
        outputs=[output_path],
        notes="Post-hoc stability summary for active calibration sampling using existing multi-seed random baseline.",
    )
    print(f"wrote={output_path}")


if __name__ == "__main__":
    main()
