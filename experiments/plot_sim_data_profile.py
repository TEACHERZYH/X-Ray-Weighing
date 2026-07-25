from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiment_versioning import write_run_metadata
from figure_style import PALETTE, add_panel_label, save_figure, setup_publication_style, soften_axes


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/simulated/xray_belt_sim.csv")
OUT_DIR = Path("paper/figures")


def setup_style() -> None:
    setup_publication_style(font_size=7.2)


def select_install(df: pd.DataFrame) -> pd.DataFrame:
    group = df[(df["site_id"].eq("S04")) & (df["device_id"].eq("D01")) & (df["install_id"].eq("I01"))].copy()
    if group.empty:
        group = df.groupby(["site_id", "device_id", "install_id"], sort=True).get_group(tuple(df[["site_id", "device_id", "install_id"]].iloc[0]))
    group = group.sort_values("timestamp_s").iloc[:1800].copy()
    group["minute"] = group["timestamp_s"] / 60.0
    return group


def plot_panel_a(ax: plt.Axes, group: pd.DataFrame) -> None:
    ax.plot(group["minute"], group["true_loading_kgpm"], color=PALETTE["temporal"], linewidth=1.25, label="载荷")
    empty = group["empty_flag"].astype(bool)
    if empty.any():
        ax.fill_between(group["minute"], 0, group["true_loading_kgpm"].max() * 1.08, where=empty, color="#DDEAF7", alpha=0.7, linewidth=0, label="空载片段")
    ax.set_ylabel("载荷/(kg/m)")
    ax.set_xlabel("时间/min")
    ax.set_title("典型安装段煤流与空载片段", loc="left")
    soften_axes(ax)
    add_panel_label(ax, "a")
    ax2 = ax.twinx()
    ax2.plot(group["minute"], group["rx_value"], color=PALETTE["physics"], linewidth=0.95, alpha=0.9, label="接收端射线值")
    ax2.set_ylabel("射线值")
    ax2.spines["top"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.05), fontsize=7)


def plot_panel_b(ax: plt.Axes, group: pd.DataFrame) -> None:
    ax.plot(group["minute"], group["env_temp_c"], color=PALETTE["risk"], linewidth=1.1, label="环境温度")
    ax.plot(group["minute"], group["source_temp_c"], color=PALETTE["data"], linewidth=1.1, label="源端温度")
    ax.set_ylabel("温度/degC")
    ax.set_xlabel("时间/min")
    ax.set_title("环境与源端温度漂移", loc="left")
    soften_axes(ax)
    add_panel_label(ax, "b")
    ax2 = ax.twinx()
    ax2.plot(group["minute"], group["env_humidity_pct"], color="#8B6FB3", linewidth=1.0, alpha=0.9, label="环境湿度")
    ax2.set_ylabel("湿度/%")
    ax2.spines["top"].set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, ncol=1, loc="upper right", fontsize=7)


def plot_panel_c(ax: plt.Axes, df: pd.DataFrame) -> None:
    stats = (
        df.groupby("coal_type", sort=True)
        .agg(samples=("coal_type", "size"), ash_missing_pct=("ash_missing", lambda s: 100 * s.astype(float).mean()))
        .reset_index()
    )
    x = range(len(stats))
    bars = ax.bar(x, stats["samples"] / 1000.0, color="#9AB884", width=0.62, edgecolor="#333333", linewidth=0.35)
    ax.set_xticks(list(x), stats["coal_type"], rotation=25, ha="right")
    ax.set_ylabel("样本数/千")
    ax.set_title("煤种覆盖与灰分缺失比例", loc="left")
    soften_axes(ax)
    add_panel_label(ax, "c")
    for bar, miss in zip(bars, stats["ash_missing_pct"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.35, f"缺失{miss:.1f}%", ha="center", va="bottom", fontsize=7)


def make_figure() -> None:
    setup_style()
    df = pd.read_csv(DATA)
    group = select_install(df)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7.2, 5.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.12, 1.0])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    plot_panel_a(ax_a, group)
    plot_panel_b(ax_b, group)
    plot_panel_c(ax_c, df)

    fig.text(0.01, 0.012, "模拟器覆盖煤流、空载、温湿度、源端温度、煤种和灰分缺失，用于受控扰动验证。", fontsize=6.2, color=PALETTE["muted"])
    svg = OUT_DIR / "fig3_sim_data_profile.svg"
    png = OUT_DIR / "fig3_sim_data_profile.png"
    save_figure(fig, OUT_DIR / "fig3_sim_data_profile", png_dpi=450, tiff_dpi=450, pdf=False)
    plt.close(fig)
    write_run_metadata(
        svg,
        script=Path(__file__).resolve().relative_to(ROOT).as_posix(),
        args={},
        inputs=[DATA],
        outputs=[svg, png],
        notes="Figure 3 summarizes simulated data coverage and drift patterns.",
    )
    print(f"wrote={svg}")


if __name__ == "__main__":
    make_figure()
