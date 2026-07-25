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

PALETTE = {
    "data": "#6F7686",
    "physics": "#2E6F9E",
    "temporal": "#4C8B6B",
    "calibration": "#C46B37",
    "target": "#B54E4E",
    "grid": "#D9DDE3",
    "muted": "#6F7686",
    "text": "#222222",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot pseudo-real X-ray experiment summary.")
    parser.add_argument("--out-base", default="paper/figures/pseudo_real_experiment_summary")
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Noto Sans SC", "SimHei", "Arial", "DejaVu Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.unicode_minus": False,
            "font.size": 7.3,
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


def method_color(method: str) -> str:
    if "DataOnly" in method:
        return PALETTE["data"]
    if "Physics-Ridge" in method:
        return PALETTE["physics"]
    if "Temporal" in method or "NoMissing" in method:
        return PALETTE["temporal"]
    if "PG-TSC" in method:
        return PALETTE["calibration"]
    return PALETTE["muted"]


def panel_overall(ax: plt.Axes, summary: pd.DataFrame) -> None:
    rows = summary[summary["method"].isin(["DataOnly-Ridge", "Physics-Ridge", "Physics-Temporal-Ridge", "PG-TSC-lite budget=20 strategy=time_uniform", "PG-TSC-lite budget=180 strategy=time_uniform"])].copy()
    labels = {
        "DataOnly-Ridge": "DataOnly",
        "Physics-Ridge": "Physics",
        "Physics-Temporal-Ridge": "Temporal",
        "PG-TSC-lite budget=20 strategy=time_uniform": "Cal-20",
        "PG-TSC-lite budget=180 strategy=time_uniform": "Cal-180",
    }
    ax.scatter(rows["mae"], rows["cum_err_pct"], s=58, c=[method_color(m) for m in rows["method"]], edgecolor="white", linewidth=0.8, zorder=3)
    for _, row in rows.iterrows():
        ax.text(row["mae"] + 0.22, row["cum_err_pct"] + 0.035, labels[row["method"]], fontsize=6.4, va="center")
    ax.axhline(1.0, color=PALETTE["target"], lw=0.85, ls="--")
    ax.text(rows["mae"].max() + 0.6, 1.03, "1%目标", color=PALETTE["target"], fontsize=6.2, ha="right", va="bottom")
    ax.set_xlabel("MAE/(kg/s)")
    ax.set_ylabel("累计误差/%")
    ax.set_title("未见站点总体误差", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(color=PALETTE["grid"], linewidth=0.55)
    add_panel_label(ax, "a")


def panel_calibration(ax: plt.Axes, calibration: pd.DataFrame) -> None:
    order = ["time_uniform", "prediction_stratified", "load_stratified", "random"]
    labels = {"time_uniform": "Time", "prediction_stratified": "Pred.", "load_stratified": "Load", "random": "Rand."}
    width = 0.18
    budgets = [5, 20, 60, 180]
    xs = list(range(len(budgets)))
    colors = ["#7A869A", "#2E6F9E", "#4C8B6B", "#C46B37"]
    for idx, strategy in enumerate(order):
        vals = []
        for budget in budgets:
            hit = calibration[(calibration["budget"].eq(budget)) & (calibration["strategy"].eq(strategy))]
            vals.append(float(hit.iloc[0]["cum_err_pct"]) if not hit.empty else float("nan"))
        offset = (idx - 1.5) * width
        ax.bar([x + offset for x in xs], vals, width=width, color=colors[idx], edgecolor="#333333", linewidth=0.35, label=labels[strategy])
    ax.axhline(1.0, color=PALETTE["target"], lw=0.85, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("校准样本/安装")
    ax.set_ylabel("累计误差/%")
    ax.set_title("校准预算与采样策略", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    ax.legend(loc="upper right", ncol=2, fontsize=5.9)
    add_panel_label(ax, "b")


def panel_stratified(ax: plt.Axes, stratified: pd.DataFrame) -> None:
    df = stratified[
        stratified["dimension"].isin(["env_temp_c", "env_humidity_pct", "site_id"])
        & stratified["method"].eq("PG-TSC-lite budget=180 strategy=time_uniform")
    ].copy()
    df["label"] = df["dimension"].map({"env_temp_c": "Temp", "env_humidity_pct": "Humidity", "site_id": "Site"}) + ":" + df["group"].astype(str)
    df = df.sort_values(["dimension", "group"])
    ax.bar(range(len(df)), df["cum_err_pct"], color=PALETTE["calibration"], edgecolor="#333333", linewidth=0.35, width=0.65)
    ax.axhline(1.0, color=PALETTE["target"], lw=0.85, ls="--")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=35, ha="right")
    ax.set_ylabel("累计误差/%")
    ax.set_title("校准后分层累计误差", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    add_panel_label(ax, "c")


def panel_install(ax: plt.Axes, install: pd.DataFrame) -> None:
    methods = ["Physics-Temporal-Ridge", "PG-TSC-lite budget=180 strategy=time_uniform"]
    data = [install[install["method"].eq(method)]["cum_err_pct"].astype(float).to_numpy() for method in methods]
    bp = ax.boxplot(data, patch_artist=True, widths=0.5, showfliers=True)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Temporal", "Cal-180"])
    colors = [PALETTE["temporal"], PALETTE["calibration"]]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#333333")
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.0)
    ax.axhline(1.0, color=PALETTE["target"], lw=0.85, ls="--")
    ax.set_ylabel("安装级累计误差/%")
    ax.set_title("安装级尾部风险", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    add_panel_label(ax, "d")


def main() -> int:
    args = parse_args()
    setup_style()
    out_base = Path(args.out_base)
    if not out_base.is_absolute():
        out_base = ROOT / out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS / "pseudo_real_experiment_summary.csv"
    calibration_path = RESULTS / "pseudo_real_calibration_strategy.csv"
    stratified_path = RESULTS / "pseudo_real_stratified_errors.csv"
    install_path = RESULTS / "pseudo_real_install_errors.csv"
    summary = pd.read_csv(summary_path)
    calibration = pd.read_csv(calibration_path)
    stratified = pd.read_csv(stratified_path)
    install = pd.read_csv(install_path)

    fig = plt.figure(figsize=(7.2, 5.3))
    gs = fig.add_gridspec(2, 2, hspace=0.48, wspace=0.36)
    panel_overall(fig.add_subplot(gs[0, 0]), summary)
    panel_calibration(fig.add_subplot(gs[0, 1]), calibration)
    panel_stratified(fig.add_subplot(gs[1, 0]), stratified)
    panel_install(fig.add_subplot(gs[1, 1]), install)
    fig.text(
        0.015,
        0.012,
        "数据源：类真实字段替换演练数据；结果用于流程验证，不代表真实工业现场验证。",
        fontsize=6.2,
        color=PALETTE["muted"],
    )

    svg_path = out_base.with_suffix(".svg")
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    fig.savefig(svg_path, bbox_inches="tight")
    strip_trailing_space(svg_path)
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    meta = write_run_metadata(
        svg_path,
        script=__file__,
        args={"out_base": out_base.relative_to(ROOT).as_posix()},
        inputs=[summary_path, calibration_path, stratified_path, install_path],
        outputs=[svg_path, png_path, pdf_path],
        notes="Pseudo-real experiment report figure. Not a main manuscript field-validation figure.",
    )
    print(f"wrote={svg_path}")
    print(f"wrote={png_path}")
    print(f"wrote={pdf_path}")
    print(f"wrote={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
