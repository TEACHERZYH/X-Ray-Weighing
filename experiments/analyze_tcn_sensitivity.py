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
    "raw": "#8F5F4A",
    "cal": "#2F6F9F",
    "grid": "#D9DDE3",
    "text": "#222222",
    "muted": "#6F7686",
    "highlight": "#B54E4E",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot TCN sensitivity results.")
    parser.add_argument("--input", default="results/sim_tcn_sensitivity.csv")
    parser.add_argument("--summary", default="results/sim_tcn_sensitivity_summary.csv")
    parser.add_argument("--out-base", default="paper/figures/fig10_tcn_sensitivity")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 7.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
        }
    )


def calibration_samples(method: str) -> int:
    if "calibrate=" not in method:
        return 0
    return int(method.rsplit("calibrate=", 1)[1])


def config_label(config: str) -> str:
    return config.replace("win", "W").replace("_c", "\nC").replace("_l", " L")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["calibrate_samples"] = work["method"].map(calibration_samples)
    rows = []
    for config, group in work.groupby("config", sort=False):
        raw = group[group["calibrate_samples"].eq(0)].iloc[0]
        calibrated = group[group["calibrate_samples"].gt(0)].copy()
        best_cum = calibrated.sort_values(["cum_err_pct", "mae"]).iloc[0]
        best_mae = calibrated.sort_values(["mae", "cum_err_pct"]).iloc[0]
        rows.append(
            {
                "config": config,
                "seq_len": int(raw["seq_len"]),
                "channels": int(raw["channels"]),
                "levels": int(raw["levels"]),
                "tcn_params": int(raw["tcn_params"]),
                "raw_mae": float(raw["mae"]),
                "raw_rmse": float(raw["rmse"]),
                "raw_cum_err_pct": float(raw["cum_err_pct"]),
                "best_cum_calibrate_samples": int(best_cum["calibrate_samples"]),
                "best_cum_err_pct": float(best_cum["cum_err_pct"]),
                "best_cum_mae": float(best_cum["mae"]),
                "best_cum_rmse": float(best_cum["rmse"]),
                "best_mae_calibrate_samples": int(best_mae["calibrate_samples"]),
                "best_mae": float(best_mae["mae"]),
                "best_mae_cum_err_pct": float(best_mae["cum_err_pct"]),
            }
        )
    return pd.DataFrame(rows)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.06, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def plot_summary(summary: pd.DataFrame, out_base: Path) -> None:
    setup_style()
    labels = [config_label(c) for c in summary["config"]]
    x = list(range(len(summary)))
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.35), constrained_layout=True)

    ax = axes[0]
    ax.plot(x, summary["raw_mae"], color=PALETTE["raw"], marker="o", lw=1.6, ms=4.2, label="未校准")
    ax.plot(x, summary["best_mae"], color=PALETTE["cal"], marker="s", lw=1.6, ms=4.2, label="最佳MAE校准")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE/(kg/s)")
    ax.set_title("不同TCN配置的瞬时误差", fontsize=8.5, fontweight="bold", pad=6)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, 1.16), ncol=2, fontsize=6.2)
    best_idx = int(summary["best_mae"].idxmin())
    y_min = min(summary["raw_mae"].min(), summary["best_mae"].min()) - 0.18
    y_max = max(summary["raw_mae"].max(), summary["best_mae"].max()) + 0.22
    ax.set_ylim(y_min, y_max)
    ax.scatter([best_idx], [summary.loc[best_idx, "best_mae"]], s=34, color=PALETTE["highlight"], zorder=4)
    ax.text(best_idx, summary.loc[best_idx, "best_mae"] - 0.05, "最低", color=PALETTE["highlight"], ha="center", va="top", fontsize=6.0)
    add_panel_label(ax, "a")

    ax = axes[1]
    ax.plot(x, summary["raw_cum_err_pct"], color=PALETTE["raw"], marker="o", lw=1.6, ms=4.2, label="未校准")
    ax.plot(x, summary["best_cum_err_pct"], color=PALETTE["cal"], marker="s", lw=1.6, ms=4.2, label="最佳累计校准")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("累计误差/%")
    ax.set_title("不同TCN配置的累计偏差", fontsize=8.5, fontweight="bold", pad=6)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    ax.axhline(1.0, color=PALETTE["highlight"], lw=0.85, ls="--")
    ax.text(0.98, 1.0, "1%目标", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.8, color=PALETTE["highlight"])
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, 1.16), ncol=2, fontsize=6.2)
    best_idx = int(summary["best_cum_err_pct"].idxmin())
    ax.set_ylim(0, max(summary["raw_cum_err_pct"].max(), summary["best_cum_err_pct"].max()) + 0.12)
    ax.scatter([best_idx], [summary.loc[best_idx, "best_cum_err_pct"]], s=34, color=PALETTE["highlight"], zorder=4)
    ax.text(best_idx, summary.loc[best_idx, "best_cum_err_pct"] + 0.035, "最低", color=PALETTE["highlight"], ha="center", va="bottom", fontsize=6.0)
    add_panel_label(ax, "b")

    fig.text(
        0.01,
        -0.02,
        "注：每个配置均采用S01-S03训练、S04测试；参数量范围为11.6k-118.5k；最佳校准仅在20/60/180个代表性样本中选择。",
        fontsize=6.2,
        color=PALETTE["muted"],
    )
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".png"), dpi=450, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
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
    plot_summary(summary, out_base)
    meta = write_run_metadata(
        summary_path,
        script="experiments/analyze_tcn_sensitivity.py",
        args=vars(args),
        inputs=[input_path],
        outputs=[summary_path, out_base.with_suffix(".svg"), out_base.with_suffix(".png"), out_base.with_suffix(".pdf")],
        notes="Summary and figure for TCN sequence length and capacity sensitivity experiment.",
    )
    print(f"wrote={summary_path}")
    print(f"wrote={out_base.with_suffix('.svg')}")
    print(f"wrote={out_base.with_suffix('.png')}")
    print(f"wrote={out_base.with_suffix('.pdf')}")
    print(f"wrote={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
