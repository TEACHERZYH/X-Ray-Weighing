from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from figure_style import METHOD_COLORS, PALETTE, add_target_line, save_figure, setup_publication_style, soften_axes


def setup_style() -> None:
    setup_publication_style(font_size=7.4)


def plot_error_bar(out_dir: Path) -> None:
    rows = []
    base = pd.read_csv("results/sim_baselines_v2.csv")
    tcn = pd.read_csv("results/sim_tcn_results_v2.csv")
    pick = {
        "Physics+I0": "物理+I0",
        "DataOnly-Ridge": "纯数据",
        "Physics-Ridge": "物理+岭回归",
        "Physics-Temporal-Ridge": "物理+时序",
        "PG-TSC-lite calibrate=180": "PG-TSC-lite（180样本）",
    }
    for method, label in pick.items():
        r = base[base["method"].eq(method)].iloc[0]
        rows.append({"method": label, "cum_err_pct": r["cum_err_pct"]})
    r = tcn[tcn["method"].eq("PG-TSC-TCN calibrate=20")].iloc[0]
    rows.append({"method": "PG-TSC-TCN（20样本）", "cum_err_pct": r["cum_err_pct"]})

    df = pd.DataFrame(rows)
    colors = [METHOD_COLORS.get(m, PALETTE["muted"]) for m in df["method"]]
    fig, ax = plt.subplots(figsize=(7.0, 3.05))
    y = range(len(df))
    ax.barh(y, df["cum_err_pct"], color=colors, height=0.58, edgecolor="#333333", linewidth=0.35, zorder=3)
    ax.set_yticks(list(y), df["method"])
    ax.invert_yaxis()
    ax.set_xlabel("累计误差/%")
    ax.set_title("跨现场累计质量误差比较", loc="left")
    ax.axvspan(0, 1.0, color="#EDF6EF", zorder=0)
    ax.axvline(1.0, color=PALETTE["target"], linewidth=0.9, linestyle=(0, (4, 2)), zorder=2)
    ax.text(
        1.03,
        0.06,
        "1%目标",
        transform=ax.get_xaxis_transform(),
        color=PALETTE["target"],
        fontsize=6.2,
        ha="left",
        va="bottom",
    )
    soften_axes(ax, grid_axis="x")
    for yi, v in zip(y, df["cum_err_pct"], strict=True):
        ax.text(v + 0.06, yi, f"{v:.3f}", ha="left", va="center", fontsize=6.8, color=PALETTE["text"])
    fig.text(0.01, 0.02, "绿色区域表示累计误差低于1%的仪表目标区间。", fontsize=6.2, color=PALETTE["muted"])
    save_figure(fig, out_dir / "fig4a_cumulative_error", png_dpi=450, tiff_dpi=450, pdf=False)
    plt.close(fig)


def plot_cumulative_curve(out_dir: Path) -> None:
    pred = pd.read_csv("results/sim_predictions_v2.csv")
    g = pred[(pred["device_id"].eq("D01")) & (pred["install_id"].eq("I01"))].copy()
    g = g.sort_values("timestamp_s")
    g = g.iloc[:1800]
    g["true_cum"] = g["true_flow_kgps"].cumsum()
    methods = [
        ("DataOnly-HGB", "纯数据HGB", METHOD_COLORS["纯数据HGB"]),
        ("Physics-Ridge", "物理+岭回归", METHOD_COLORS["物理+岭回归"]),
        ("Physics-Temporal-Ridge", "物理+时序", METHOD_COLORS["物理+时序"]),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(g["timestamp_s"], g["true_cum"], label="真实值", color="#222222", linewidth=1.9, zorder=4)
    end_labels = []
    for col, label, color in methods:
        if col in g:
            curve = g[col].clip(lower=0).cumsum()
            ax.plot(g["timestamp_s"], curve, label=label, color=color, linewidth=1.25, zorder=3)
            end_labels.append((float(curve.iloc[-1]), label, color))
    end_labels.append((float(g["true_cum"].iloc[-1]), "真实值", "#222222"))
    end_labels.sort(key=lambda item: item[0])
    min_gap = 18000.0
    adjusted = []
    for value, label, color in end_labels:
        y_text = value if not adjusted else max(value, adjusted[-1][0] + min_gap)
        adjusted.append((y_text, value, label, color))
    x_end = float(g["timestamp_s"].iloc[-1])
    for y_text, value, label, color in adjusted:
        ax.plot([x_end, x_end + 42], [value, y_text], color=color, linewidth=0.55, alpha=0.75)
        ax.text(x_end + 55, y_text, label, fontsize=5.9, va="center", color=color)
    ax.set_xlabel("时间/s")
    ax.set_ylabel("累计重量/kg")
    ax.set_title("典型安装段累计质量曲线", loc="left")
    ax.set_xlim(g["timestamp_s"].min(), g["timestamp_s"].max() + 350)
    ymax = max(y for y, *_ in adjusted)
    ax.set_ylim(bottom=min(0, float(g["true_cum"].min()) - 5000), top=ymax + 35000)
    soften_axes(ax, grid_axis="both")
    fig.text(0.01, 0.02, "曲线展示一个目标安装段，排名结论以表1和图6(a)为准。", fontsize=6.2, color=PALETTE["muted"])
    save_figure(fig, out_dir / "fig4b_cumulative_curve", png_dpi=450, tiff_dpi=450, pdf=False)
    plt.close(fig)


def main() -> None:
    setup_style()
    out_dir = Path("paper/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_error_bar(out_dir)
    plot_cumulative_curve(out_dir)
    print(f"wrote={out_dir / 'fig4a_cumulative_error.svg'}")
    print(f"wrote={out_dir / 'fig4b_cumulative_curve.svg'}")


if __name__ == "__main__":
    main()
