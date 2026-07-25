from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from experiment_versioning import write_run_metadata


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "paper" / "figures"


PALETTE = {
    "baseline": "#6F7686",
    "direct": "#9B6A5F",
    "calibrate": "#2E6F9E",
    "ratio": "#4C8B6B",
    "residual": "#7B6BAE",
    "mass": "#C46B37",
    "scratch": "#B54E4E",
    "soft_blue": "#D9E8F5",
    "soft_green": "#DDEDDD",
    "soft_orange": "#F2E0CF",
    "grid": "#D9DDE3",
    "text": "#222222",
}


TRANSFER_METHODS = [
    ("Pretrained direct", "Direct", "direct"),
    ("Pretrained install-calibrate=60", "Install\ncal.", "calibrate"),
    ("Pretrained install-ratio=180", "Ratio\ncal.", "ratio"),
    ("Pretrained residual-finetune", "Residual\nFT", "residual"),
    ("Pretrained residual-finetune mass=10", "Mass\nFT", "mass"),
    ("Scratch target-only", "Scratch", "scratch"),
]


ACTIVE_METHODS = [
    ("head-contiguous", "Head"),
    ("time-uniform", "Uniform"),
    ("random-mean", "Random"),
    ("prediction-stratified", "Pred.\nstrat."),
    ("feature-diverse", "Feature\ndiverse"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig. 9 transfer calibration and reliability summary.")
    parser.add_argument("--out-base", default="paper/figures/fig9_transfer_calibration_reliability")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.05, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def panel_transfer(ax: plt.Axes, transfer: pd.DataFrame) -> None:
    rows = []
    for method, label, color_key in TRANSFER_METHODS:
        hit = transfer[transfer["method"].eq(method)]
        if hit.empty:
            raise ValueError(f"missing transfer method: {method}")
        row = hit.iloc[0]
        rows.append({"label": label, "mae": float(row["mae"]), "cum": float(row["cum_err_pct"]), "color": PALETTE[color_key]})
    df = pd.DataFrame(rows)
    ax.scatter(df["mae"], df["cum"], s=58, c=df["color"], edgecolor="white", linewidth=0.9, zorder=3)
    for _, row in df.iterrows():
        dx = 0.07 if row["label"] != "Scratch" else -0.45
        ha = "left" if row["label"] != "Scratch" else "right"
        ax.text(row["mae"] + dx, row["cum"] + 0.04, row["label"], fontsize=6.6, ha=ha, va="center")
    ax.axhline(1.0, color=PALETTE["grid"], lw=0.8, ls="--", zorder=0)
    ax.set_xlabel("MAE/(kg/s)")
    ax.set_ylabel("累计误差/%")
    ax.set_title("目标域少样本适配折中", fontsize=8, fontweight="bold", pad=5)
    ax.grid(color=PALETTE["grid"], linewidth=0.55, alpha=0.9)
    ax.set_xlim(7.4, 13.7)
    ax.set_ylim(-0.05, 5.25)
    ax.annotate(
        "低MAE",
        xy=(8.07, 1.37),
        xytext=(7.65, 2.4),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": PALETTE["mass"]},
        color=PALETTE["mass"],
        fontsize=6.5,
    )
    ax.annotate(
        "低累计偏差",
        xy=(10.63, 0.85),
        xytext=(11.15, 0.2),
        arrowprops={"arrowstyle": "->", "lw": 0.8, "color": PALETTE["calibrate"]},
        color=PALETTE["calibrate"],
        fontsize=6.5,
    )
    add_panel_label(ax, "a")


def panel_active(ax: plt.Axes, active: pd.DataFrame) -> None:
    df = active[active["label_budget_per_install"].eq(20)].copy()
    rows = []
    for method, label in ACTIVE_METHODS:
        hit = df[df["method"].eq(method)]
        if hit.empty:
            raise ValueError(f"missing active method: {method}")
        row = hit.iloc[0]
        rows.append(
            {
                "label": label,
                "cum": float(row["cum_err_pct"]),
                "span": float(row["selected_span_ratio"]),
            }
        )
    plot_df = pd.DataFrame(rows)
    x = range(len(plot_df))
    colors = [PALETTE["baseline"], PALETTE["soft_blue"], PALETTE["soft_green"], PALETTE["calibrate"], PALETTE["soft_orange"]]
    ax.bar(x, plot_df["cum"], color=colors, edgecolor="#333333", linewidth=0.45, width=0.66)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["label"])
    ax.set_ylabel("累计误差/%")
    ax.set_ylim(0, 2.15)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    ax.set_title("20样本主动校准选样", fontsize=8, fontweight="bold", pad=5)
    for i, val in enumerate(plot_df["cum"]):
        ax.text(i, val + 0.06, f"{val:.3f}", ha="center", va="bottom", fontsize=5.8)
    ax2 = ax.twinx()
    ax2.plot(list(x), plot_df["span"], color=PALETTE["mass"], marker="o", lw=1.5, ms=4)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("覆盖率")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.8)
    ax2.tick_params(axis="y", width=0.7)
    ax2.text(3, 1.03, "覆盖完整", color=PALETTE["mass"], fontsize=6.2, ha="center", va="bottom")
    add_panel_label(ax, "b")


def panel_reliability(ax: plt.Axes, reliability: pd.DataFrame) -> None:
    df = reliability[
        reliability["strategy"].eq("prediction-stratified") & reliability["nominal_coverage_pct"].eq(90.0)
    ].copy()
    budgets = [20, 60, 180]
    interval_methods = [("global", "全局"), ("install", "安装"), ("flow-bin", "流量分层")]
    width = 0.22
    xs = list(range(len(budgets)))
    colors = [PALETTE["soft_blue"], PALETTE["soft_green"], PALETTE["soft_orange"]]
    for offset_idx, (interval, label) in enumerate(interval_methods):
        vals = []
        for budget in budgets:
            hit = df[df["label_budget_per_install"].eq(budget) & df["interval_method"].eq(interval)]
            if hit.empty:
                raise ValueError(f"missing reliability row: {budget}, {interval}")
            vals.append(float(hit.iloc[0]["coverage_pct"]))
        offset = (offset_idx - 1) * width
        ax.bar([x + offset for x in xs], vals, width=width, color=colors[offset_idx], edgecolor="#333333", linewidth=0.45, label=label)
    ax.axhline(90, color=PALETTE["grid"], lw=0.9, ls="--")
    ax.text(2.35, 90.8, "90%名义", fontsize=6.0, color=PALETTE["baseline"])
    top10 = []
    for budget in budgets:
        hit = df[df["label_budget_per_install"].eq(budget) & df["interval_method"].eq("global")]
        top10.append(float(hit.iloc[0]["risk_capture_top10_error_pct"]))
    ax2 = ax.twinx()
    ax2.plot(xs, top10, color=PALETTE["scratch"], marker="o", lw=1.5, ms=4, label="Top10捕获")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("标注样本/安装")
    ax.set_ylabel("实际覆盖率/%")
    ax2.set_ylabel("Top10捕获率/%")
    ax.set_ylim(0, 100)
    ax2.set_ylim(0, 100)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    ax.set_title("可靠性边界与风险触发", fontsize=8, fontweight="bold", pad=5)
    ax.legend(loc="upper left", fontsize=5.8, ncol=3, bbox_to_anchor=(0.0, 1.02))
    ax2.text(2, top10[-1] + 4, f"{top10[-1]:.1f}%", color=PALETTE["scratch"], fontsize=6.2, ha="center")
    add_panel_label(ax, "c")


def main() -> int:
    args = parse_args()
    setup_style()
    out_base = Path(args.out_base)
    if not out_base.is_absolute():
        out_base = ROOT / out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)

    transfer_path = RESULTS / "sim_transfer_finetune_results.csv"
    active_path = RESULTS / "sim_active_calibration_results.csv"
    reliability_path = RESULTS / "sim_prediction_reliability_results.csv"
    transfer = pd.read_csv(transfer_path)
    active = pd.read_csv(active_path)
    reliability = pd.read_csv(reliability_path)

    fig = plt.figure(figsize=(7.2, 5.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.52, wspace=0.36)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    panel_transfer(ax_a, transfer)
    panel_active(ax_b, active)
    panel_reliability(ax_c, reliability)
    fig.text(
        0.015,
        0.015,
        "数据源：表7-表9对应模拟结果；所有结果限定于目标域模拟实验。",
        fontsize=6.2,
        color=PALETTE["baseline"],
    )
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)

    meta = write_run_metadata(
        out_base.with_suffix(".svg"),
        script="experiments/plot_transfer_calibration_reliability.py",
        args={"out_base": out_base.relative_to(ROOT).as_posix()},
        inputs=[transfer_path, active_path, reliability_path],
        outputs=[out_base.with_suffix(".svg"), out_base.with_suffix(".png"), out_base.with_suffix(".pdf")],
        notes="Figure 9 summarizes target-domain transfer, active calibration, and reliability-trigger simulation evidence.",
    )
    print(f"wrote={out_base.with_suffix('.svg')}")
    print(f"wrote={out_base.with_suffix('.png')}")
    print(f"wrote={out_base.with_suffix('.pdf')}")
    print(f"wrote={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
