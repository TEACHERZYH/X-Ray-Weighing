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

SCENARIO_ORDER = ["clean", "mild", "moderate", "high", "severe"]
SCENARIO_LABELS = {
    "clean": "clean",
    "mild": "mild",
    "moderate": "moderate",
    "high": "high",
    "severe": "severe",
}
PALETTE = {
    20: "#8A6F5A",
    60: "#2E6F9E",
    180: "#4C8B6B",
    "target": "#B54E4E",
    "grid": "#D9DDE3",
    "text": "#222222",
    "muted": "#6F7686",
    "soft_red": "#F1D3D0",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig. 16 calibration label noise boundary.")
    parser.add_argument("--out-base", default="paper/figures/fig14_calibration_label_noise_boundary")
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


def strip_trailing_space(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def ordered_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["scenario"] = pd.Categorical(out["scenario"], categories=SCENARIO_ORDER, ordered=True)
    out = out.sort_values(["label_budget_per_install", "scenario"]).reset_index(drop=True)
    return out


def panel_error_distribution(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = list(range(len(SCENARIO_ORDER)))
    for budget, group in summary.groupby("label_budget_per_install"):
        group = group.sort_values("scenario")
        color = PALETTE[int(budget)]
        ax.plot(x, group["cum_err_median_pct"], marker="o", lw=1.5, ms=4.2, color=color, label=f"{int(budget)}样本")
        ax.fill_between(
            x,
            group["cum_err_q25_pct"],
            group["cum_err_q75_pct"],
            color=color,
            alpha=0.13,
            linewidth=0,
        )
    ax.axhline(1.0, color=PALETTE["target"], lw=0.9, ls="--")
    ax.text(4.05, 1.03, "1%目标", color=PALETTE["target"], fontsize=6.2, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[item] for item in SCENARIO_ORDER])
    ax.set_ylabel("累计误差/%")
    ax.set_title("累计误差分布随标签噪声扩张", fontsize=8, fontweight="bold", pad=5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    ax.set_ylim(0, 1.75)
    ax.legend(loc="upper left", ncol=3, fontsize=6.1)
    add_panel_label(ax, "a")


def panel_success_rate(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x = list(range(len(SCENARIO_ORDER)))
    tick_labels = []
    for scenario in SCENARIO_ORDER:
        mean_label_error = summary[summary["scenario"].astype(str).eq(scenario)]["label_abs_pct_error_mean"].mean()
        tick_labels.append(f"{SCENARIO_LABELS[scenario]}\n{mean_label_error:.2f}%")
    for budget, group in summary.groupby("label_budget_per_install"):
        group = group.sort_values("scenario")
        color = PALETTE[int(budget)]
        ax.plot(
            x,
            group["success_lt_1pct"],
            marker="o",
            lw=1.6,
            ms=4.4,
            color=color,
            label=f"{int(budget)}样本",
        )
    ax.axhspan(0, 60, color=PALETTE["soft_red"], alpha=0.35, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel("")
    ax.set_ylabel("低于1%目标比例/%")
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.1, 4.45)
    ax.set_title("参考标签质量限定校准通过率", fontsize=8, fontweight="bold", pad=5)
    ax.grid(color=PALETTE["grid"], linewidth=0.55)
    ax.text(0.05, 58, "低通过率区", color=PALETTE["target"], fontsize=6.2, ha="left", va="top")
    high = summary[summary["scenario"].astype(str).eq("high")]
    for _, row in high.iterrows():
        budget = int(row["label_budget_per_install"])
        x_offset, y_offset = {20: (-0.08, 3.2), 60: (-0.10, -7.0), 180: (0.12, 3.2)}[budget]
        ax.text(
            3 + x_offset,
            row["success_lt_1pct"] + y_offset,
            f"{row['success_lt_1pct']:.1f}",
            fontsize=5.7,
            color=PALETTE[budget],
            ha="center",
        )
    add_panel_label(ax, "b")


def panel_tail_risk(ax: plt.Axes, summary: pd.DataFrame) -> None:
    width = 0.22
    x = list(range(len(SCENARIO_ORDER)))
    budgets = [20, 60, 180]
    for idx, budget in enumerate(budgets):
        group = summary[summary["label_budget_per_install"].eq(budget)].sort_values("scenario")
        offset = (idx - 1) * width
        ax.bar(
            [item + offset for item in x],
            group["cum_err_p90_pct"],
            width=width,
            color=PALETTE[budget],
            edgecolor="#333333",
            linewidth=0.35,
            label=f"{budget}样本",
        )
    ax.axhline(1.0, color=PALETTE["target"], lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS[item] for item in SCENARIO_ORDER])
    ax.set_ylabel("90分位累计误差/%")
    ax.set_title("尾部累计偏差给出质控边界", fontsize=8, fontweight="bold", pad=5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    ax.set_ylim(0, 3.8)
    add_panel_label(ax, "c")


def main() -> int:
    args = parse_args()
    setup_style()
    out_base = Path(args.out_base)
    if not out_base.is_absolute():
        out_base = ROOT / out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS / "sim_calibration_label_noise_summary.csv"
    summary = ordered_summary(pd.read_csv(summary_path))

    fig = plt.figure(figsize=(7.2, 5.1))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.48, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_error_distribution(ax_a, summary)
    panel_success_rate(ax_b, summary)
    panel_tail_risk(ax_c, summary)
    fig.text(
        0.015,
        0.015,
        "数据源：表13对应模拟结果；标签噪声由系统偏差和随机误差叠加构成，结果不代表真实参考秤误差。",
        fontsize=6.2,
        color=PALETTE["muted"],
    )

    svg_path = out_base.with_suffix(".svg")
    fig.savefig(svg_path, bbox_inches="tight")
    strip_trailing_space(svg_path)
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    meta = write_run_metadata(
        out_base.with_suffix(".svg"),
        script="experiments/plot_calibration_label_noise_boundary.py",
        args={"out_base": out_base.relative_to(ROOT).as_posix()},
        inputs=[summary_path],
        outputs=[
            out_base.with_suffix(".svg"),
            out_base.with_suffix(".png"),
            out_base.with_suffix(".pdf"),
            out_base.with_suffix(".tiff"),
        ],
        notes="Figure 16 visualizes the calibration label noise boundary from the existing Table 13 simulation summary.",
    )
    print(f"wrote={out_base.with_suffix('.svg')}")
    print(f"wrote={out_base.with_suffix('.png')}")
    print(f"wrote={out_base.with_suffix('.pdf')}")
    print(f"wrote={out_base.with_suffix('.tiff')}")
    print(f"wrote={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
