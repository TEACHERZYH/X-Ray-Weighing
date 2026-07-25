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
    "DataOnly-HGB",
    "Physics-Temporal-HGB",
    "PG-TSC-lite-HGB calibrate=20",
    "PG-TSC-lite-HGB calibrate=60",
    "PG-TSC-lite-HGB calibrate=180",
]
METHOD_LABELS = {
    "DataOnly-HGB": "Data only",
    "Physics-Temporal-HGB": "Physics-temporal",
    "PG-TSC-lite-HGB calibrate=20": "PG-TSC 20",
    "PG-TSC-lite-HGB calibrate=60": "PG-TSC 60",
    "PG-TSC-lite-HGB calibrate=180": "PG-TSC 180",
}
MISMATCH_LABELS = {
    "nominal": "Nominal",
    "mild": "Mild",
    "moderate": "Moderate",
    "strong": "Strong",
}
PALETTE = {
    "DataOnly-HGB": "#8F9399",
    "Physics-Temporal-HGB": "#315F9D",
    "PG-TSC-lite-HGB calibrate=20": "#72AE91",
    "PG-TSC-lite-HGB calibrate=60": "#2F806A",
    "PG-TSC-lite-HGB calibrate=180": "#1F6656",
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
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "legend.frameon": False,
        }
    )


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["replicate", "heldout_site", "mismatch"]
    base = df[df["method"].eq("Physics-Temporal-HGB")][[*keys, "cum_err_pct"]].rename(
        columns={"cum_err_pct": "temporal_hgb_cum_err_pct"}
    )
    paired = df.merge(base, on=keys, how="left")
    paired["recovery_vs_temporal_pct"] = (
        (paired["temporal_hgb_cum_err_pct"] - paired["cum_err_pct"])
        / paired["temporal_hgb_cum_err_pct"].clip(lower=1e-9)
        * 100
    )

    rows: list[dict[str, object]] = []
    for mismatch in sorted(paired["mismatch"].unique(), key=lambda x: int(paired[paired["mismatch"].eq(x)]["mismatch_order"].iloc[0])):
        for method in METHOD_ORDER:
            part = paired[paired["mismatch"].eq(mismatch) & paired["method"].eq(method)].copy()
            if part.empty:
                continue
            rows.append(
                {
                    "mismatch": mismatch,
                    "mismatch_order": int(part["mismatch_order"].iloc[0]),
                    "method": method,
                    "n_splits": int(len(part)),
                    "mae_median": float(part["mae"].median()),
                    "rmse_median": float(part["rmse"].median()),
                    "cum_err_median_pct": float(part["cum_err_pct"].median()),
                    "cum_err_q25_pct": float(part["cum_err_pct"].quantile(0.25)),
                    "cum_err_q75_pct": float(part["cum_err_pct"].quantile(0.75)),
                    "cum_err_p90_pct": float(part["cum_err_pct"].quantile(0.90)),
                    "success_lt_1pct": float((part["cum_err_pct"] < 1.0).mean() * 100),
                    "recovery_vs_temporal_median_pct": float(part["recovery_vs_temporal_pct"].median()),
                }
            )
    return pd.DataFrame(rows)


def draw_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.14, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left")


def plot_boundary(summary: pd.DataFrame, out_base: Path) -> None:
    configure_style()
    ordered_methods = [m for m in METHOD_ORDER if m in set(summary["method"])]
    mismatch_order = (
        summary[["mismatch", "mismatch_order"]]
        .drop_duplicates()
        .sort_values("mismatch_order")
        .reset_index(drop=True)
    )
    x = mismatch_order["mismatch_order"].to_numpy(int)
    xlabels = [MISMATCH_LABELS.get(m, m) for m in mismatch_order["mismatch"]]

    fig = plt.figure(figsize=(7.2, 4.25), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])
    ax_line = fig.add_subplot(gs[:, 0])
    ax_success = fig.add_subplot(gs[0, 1])
    ax_recovery = fig.add_subplot(gs[1, 1])

    for method in ordered_methods:
        part = summary[summary["method"].eq(method)].sort_values("mismatch_order")
        y = part["cum_err_median_pct"].to_numpy(float)
        ylo = part["cum_err_q25_pct"].to_numpy(float)
        yhi = part["cum_err_q75_pct"].to_numpy(float)
        ax_line.plot(x, y, marker="o", ms=4.0, lw=1.5, color=PALETTE[method], label=METHOD_LABELS[method])
        ax_line.fill_between(x, ylo, yhi, color=PALETTE[method], alpha=0.10, linewidth=0)
    ax_line.axhline(1.0, color="#A33D3D", lw=0.9, ls="--")
    ax_line.text(
        0.98,
        0.95,
        "1% target",
        transform=ax_line.transAxes,
        color="#A33D3D",
        va="top",
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0, "alpha": 0.88},
    )
    ax_line.set_xticks(x)
    ax_line.set_xticklabels(xlabels)
    ax_line.set_ylabel("Median cumulative mass error (%)")
    ax_line.set_xlabel("Target simulator mismatch level")
    ax_line.grid(axis="y", color="#E7E7E7", lw=0.6)
    ax_line.legend(loc="upper left", ncols=1, handlelength=1.8)
    draw_panel_label(ax_line, "a")

    focus = ["Physics-Temporal-HGB", "PG-TSC-lite-HGB calibrate=60", "PG-TSC-lite-HGB calibrate=180"]
    bar_width = 0.24
    offsets = np.linspace(-bar_width, bar_width, len(focus))
    for off, method in zip(offsets, focus):
        part = summary[summary["method"].eq(method)].sort_values("mismatch_order")
        ax_success.bar(
            x + off,
            part["success_lt_1pct"].to_numpy(float),
            width=bar_width,
            color=PALETTE[method],
            alpha=0.88,
            label=METHOD_LABELS[method],
        )
    ax_success.set_ylim(0, 105)
    ax_success.set_xticks(x)
    ax_success.set_xticklabels(xlabels, rotation=0)
    ax_success.set_ylabel("Splits below 1% (%)")
    ax_success.grid(axis="y", color="#E7E7E7", lw=0.6)
    draw_panel_label(ax_success, "b")

    recovery_methods = [
        "PG-TSC-lite-HGB calibrate=20",
        "PG-TSC-lite-HGB calibrate=60",
        "PG-TSC-lite-HGB calibrate=180",
    ]
    for method in recovery_methods:
        part = summary[summary["method"].eq(method)].sort_values("mismatch_order")
        ax_recovery.plot(
            x,
            part["recovery_vs_temporal_median_pct"].to_numpy(float),
            marker="o",
            ms=3.8,
            lw=1.4,
            color=PALETTE[method],
            label=METHOD_LABELS[method],
        )
    ax_recovery.axhline(0, color="#555555", lw=0.8, ls=":")
    ax_recovery.set_xticks(x)
    ax_recovery.set_xticklabels(xlabels, rotation=0)
    ax_recovery.set_ylabel("Median recovery vs\nuncalibrated (%)")
    ax_recovery.set_xlabel("Target simulator mismatch level")
    ax_recovery.grid(axis="y", color="#E7E7E7", lw=0.6)
    draw_panel_label(ax_recovery, "c")

    fig.suptitle("Simulator-mismatch boundary of PG-TSC few-sample calibration", x=0.02, ha="left", fontsize=8.5)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize and plot simulator-mismatch boundary experiment.")
    parser.add_argument("--input", default="results/sim_simulator_mismatch_boundary.csv")
    parser.add_argument("--summary", default="results/sim_simulator_mismatch_boundary_summary.csv")
    parser.add_argument("--out-base", default="paper/figures/fig13_simulator_mismatch_boundary")
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
    plot_boundary(summary, out_base)
    outputs = [
        summary_path,
        out_base.with_suffix(".svg"),
        out_base.with_suffix(".png"),
        out_base.with_suffix(".pdf"),
        out_base.with_suffix(".tiff"),
    ]
    meta = write_run_metadata(
        summary_path,
        script="experiments/analyze_simulator_mismatch_boundary.py",
        args=vars(args),
        inputs=[input_path],
        outputs=outputs,
        notes="Summary and publication figure for simulator-mismatch boundary experiment.",
    )
    print(f"wrote={summary_path}")
    for item in outputs[1:]:
        print(f"wrote={item}")
    print(f"wrote={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
