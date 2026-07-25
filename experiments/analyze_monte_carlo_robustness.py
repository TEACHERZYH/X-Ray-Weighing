from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_versioning import write_run_metadata


ROOT = Path(__file__).resolve().parents[1]
METHOD_ORDER = [
    "DataOnly-Ridge",
    "Physics+I0",
    "Physics-Ridge",
    "Physics-Temporal-Ridge",
    "PG-TSC-lite calibrate=20",
    "PG-TSC-lite calibrate=60",
    "PG-TSC-lite calibrate=180",
]
METHOD_LABELS = {
    "DataOnly-Ridge": "Data only",
    "Physics+I0": "Physics",
    "Physics-Ridge": "Physics\nRidge",
    "Physics-Temporal-Ridge": "Physics +\ntemporal",
    "PG-TSC-lite calibrate=20": "PG-TSC\n20",
    "PG-TSC-lite calibrate=60": "PG-TSC\n60",
    "PG-TSC-lite calibrate=180": "PG-TSC\n180",
}
PALETTE = {
    "DataOnly-Ridge": "#8A8F98",
    "Physics+I0": "#6E8FB2",
    "Physics-Ridge": "#4776A8",
    "Physics-Temporal-Ridge": "#2F5F9E",
    "PG-TSC-lite calibrate=20": "#4C9A7F",
    "PG-TSC-lite calibrate=60": "#2F7F68",
    "PG-TSC-lite calibrate=180": "#1F6656",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.frameon": False,
        }
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["method"].eq("DataOnly-Ridge")][["replicate", "heldout_site", "mae", "cum_err_pct"]].rename(
        columns={"mae": "base_mae", "cum_err_pct": "base_cum_err_pct"}
    )
    paired = df.merge(base, on=["replicate", "heldout_site"], how="left")
    paired["paired_mae_reduction_pct"] = (paired["base_mae"] - paired["mae"]) / paired["base_mae"].clip(lower=1e-9) * 100
    paired["paired_cum_reduction_pct"] = (
        (paired["base_cum_err_pct"] - paired["cum_err_pct"]) / paired["base_cum_err_pct"].clip(lower=1e-9) * 100
    )

    rows: list[dict[str, object]] = []
    for method in METHOD_ORDER:
        part = paired[paired["method"].eq(method)].copy()
        if part.empty:
            continue
        rows.append(
            {
                "method": method,
                "n_splits": int(len(part)),
                "mae_median": float(part["mae"].median()),
                "mae_q25": float(part["mae"].quantile(0.25)),
                "mae_q75": float(part["mae"].quantile(0.75)),
                "cum_err_median_pct": float(part["cum_err_pct"].median()),
                "cum_err_q25_pct": float(part["cum_err_pct"].quantile(0.25)),
                "cum_err_q75_pct": float(part["cum_err_pct"].quantile(0.75)),
                "cum_err_p90_pct": float(part["cum_err_pct"].quantile(0.90)),
                "success_lt_1pct": float((part["cum_err_pct"] < 1.0).mean() * 100),
                "paired_cum_improved_pct": float((part["cum_err_pct"] < part["base_cum_err_pct"]).mean() * 100),
                "paired_mae_reduction_median_pct": float(part["paired_mae_reduction_pct"].median()),
                "paired_cum_reduction_median_pct": float(part["paired_cum_reduction_pct"].median()),
                "paired_cum_reduction_q25_pct": float(part["paired_cum_reduction_pct"].quantile(0.25)),
                "paired_cum_reduction_q75_pct": float(part["paired_cum_reduction_pct"].quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def draw_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")


def plot_robustness(df: pd.DataFrame, summary: pd.DataFrame, out_base: Path) -> None:
    configure_style()
    ordered = [method for method in METHOD_ORDER if method in set(df["method"])]
    fig = plt.figure(figsize=(7.2, 4.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
    ax_box = fig.add_subplot(gs[:, 0])
    ax_delta = fig.add_subplot(gs[0, 1])
    ax_success = fig.add_subplot(gs[1, 1])

    box_data = [df[df["method"].eq(method)]["cum_err_pct"].to_numpy(float) for method in ordered]
    bp = ax_box.boxplot(box_data, patch_artist=True, widths=0.58, showfliers=False)
    for patch, method in zip(bp["boxes"], ordered):
        patch.set_facecolor(PALETTE[method])
        patch.set_alpha(0.76)
        patch.set_edgecolor("#333333")
    for key in ["whiskers", "caps", "medians"]:
        for artist in bp[key]:
            artist.set_color("#333333")
            artist.set_linewidth(0.8)
    rng = np.random.default_rng(17)
    for idx, method in enumerate(ordered, start=1):
        vals = df[df["method"].eq(method)]["cum_err_pct"].to_numpy(float)
        x = idx + rng.normal(0, 0.035, size=len(vals))
        ax_box.scatter(x, vals, s=5, color="#222222", alpha=0.22, linewidths=0)
    ax_box.axhline(1.0, color="#A33D3D", lw=0.9, ls="--")
    ax_box.text(
        0.98,
        0.98,
        "Dashed line: 1% target",
        transform=ax_box.transAxes,
        color="#A33D3D",
        va="top",
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.86},
    )
    ax_box.set_ylabel("Cumulative mass error (%)")
    ax_box.set_xticks(range(1, len(ordered) + 1))
    ax_box.set_xticklabels([METHOD_LABELS[m] for m in ordered], rotation=0)
    ax_box.grid(axis="y", color="#E6E6E6", lw=0.6)
    draw_panel_label(ax_box, "a")

    method_summary = summary.set_index("method").loc[ordered].reset_index()
    y = np.arange(len(method_summary))
    colors = [PALETTE[m] for m in method_summary["method"]]
    improved = method_summary["paired_cum_improved_pct"].to_numpy(float)
    ax_delta.barh(y, improved, color=colors, alpha=0.84, height=0.66)
    ax_delta.axvline(50, color="#555555", lw=0.8, ls=":")
    ax_delta.set_xlim(0, 100)
    ax_delta.set_xlabel("Improved vs data only (%)")
    ax_delta.set_yticks(y)
    ax_delta.set_yticklabels([METHOD_LABELS[m].replace("\n", " ") for m in method_summary["method"]])
    ax_delta.invert_yaxis()
    ax_delta.grid(axis="x", color="#E6E6E6", lw=0.6)
    draw_panel_label(ax_delta, "b")

    success = method_summary["success_lt_1pct"].to_numpy(float)
    ax_success.barh(y, success, color=colors, alpha=0.86, height=0.66)
    ax_success.set_xlim(0, 100)
    ax_success.set_xlabel("Splits below 1% error (%)")
    ax_success.set_yticks(y)
    ax_success.set_yticklabels([METHOD_LABELS[m].replace("\n", " ") for m in method_summary["method"]])
    ax_success.invert_yaxis()
    ax_success.grid(axis="x", color="#E6E6E6", lw=0.6)
    draw_panel_label(ax_success, "c")

    fig.suptitle("Monte Carlo leave-one-site robustness of PG-TSC simulation", x=0.02, ha="left", fontsize=8.5)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize and plot Monte Carlo robustness experiment.")
    parser.add_argument("--input", default="results/sim_monte_carlo_robustness.csv")
    parser.add_argument("--summary", default="results/sim_monte_carlo_robustness_summary.csv")
    parser.add_argument("--out-base", default="paper/figures/fig11_monte_carlo_robustness")
    args = parser.parse_args()

    input_path = Path(args.input)
    summary_path = Path(args.summary)
    out_base = Path(args.out_base)
    if not input_path.is_absolute():
        input_path = ROOT / input_path
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    if not out_base.is_absolute():
        out_base = ROOT / out_base

    df = pd.read_csv(input_path)
    summary = summarize(df)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    plot_robustness(df, summary, out_base)
    meta = write_run_metadata(
        summary_path,
        script="experiments/analyze_monte_carlo_robustness.py",
        args=vars(args),
        inputs=[input_path],
        outputs=[summary_path, out_base.with_suffix(".svg"), out_base.with_suffix(".png"), out_base.with_suffix(".pdf")],
        notes="Summary and publication figure for Monte Carlo leave-one-site robustness experiment.",
    )
    print(f"wrote={summary_path}")
    print(f"wrote={out_base.with_suffix('.svg')}")
    print(f"wrote={out_base.with_suffix('.png')}")
    print(f"wrote={out_base.with_suffix('.pdf')}")
    print(f"wrote={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
