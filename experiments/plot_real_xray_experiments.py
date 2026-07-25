from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from experiment_versioning import write_run_metadata


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

PALETTE = {
    "data": "#6F7686",
    "physics": "#2E6F9E",
    "temporal": "#4C8B6B",
    "hgb": "#8A5A9E",
    "rf": "#8B7C3F",
    "pgt": "#C46B37",
    "target": "#B54E4E",
    "grid": "#D9DDE3",
    "muted": "#6F7686",
    "text": "#222222",
}

LABELS = {
    "Physics+I0": "Physics+I0",
    "DataOnly-Ridge": "DataOnly",
    "Physics-Ridge": "Physics",
    "Physics-Temporal-Ridge": "Temporal",
    "HGB": "HGB",
    "RandomForest": "RF",
    "PG-TSC-real-mass-scale budget=5 strategy=prediction_stratified": "PG-5",
    "PG-TSC-real-mass-scale budget=60 strategy=prediction_stratified": "PG-60",
    "PG-TSC-real-mass-scale budget=180 strategy=prediction_stratified": "PG-180",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot real X-ray belt experiment summary.")
    parser.add_argument("--out-base", default="paper/figures/real_xray_experiment_summary")
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
    ax.text(-0.11, 1.04, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def strip_trailing_space(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def method_color(method: str) -> str:
    if method == "DataOnly-Ridge":
        return PALETTE["data"]
    if method == "Physics-Ridge" or method == "Physics+I0":
        return PALETTE["physics"]
    if method == "Physics-Temporal-Ridge" or method == "NoMissingFlag-Ablation":
        return PALETTE["temporal"]
    if method == "HGB":
        return PALETTE["hgb"]
    if method == "RandomForest":
        return PALETTE["rf"]
    if "PG-TSC-real-mass-scale" in method:
        return PALETTE["pgt"]
    return PALETTE["muted"]


def panel_main(ax: plt.Axes, summary: pd.DataFrame) -> None:
    keep = list(LABELS)
    rows = summary[summary["method"].isin(keep)].copy()
    rows["label"] = rows["method"].map(LABELS)
    ax.scatter(
        rows["mae"],
        rows["cum_err_pct"],
        s=50,
        c=[method_color(m) for m in rows["method"]],
        edgecolor="white",
        linewidth=0.85,
        zorder=3,
    )
    for _, row in rows.iterrows():
        offsets = {
            "Temporal": (0.22, 0.11),
            "Physics": (0.22, -0.12),
            "HGB": (0.22, -0.12),
            "PG-5": (0.20, 0.08),
            "PG-60": (0.20, -0.09),
            "PG-180": (0.20, 0.08),
            "RF": (0.22, -0.02),
            "DataOnly": (0.22, 0.03),
            "Physics+I0": (0.22, 0.03),
        }
        dx, dy = offsets.get(str(row["label"]), (0.22, 0.025))
        ax.text(float(row["mae"]) + dx, float(row["cum_err_pct"]) + dy, row["label"], fontsize=6.0, va="center")
    ax.axhline(1.0, color=PALETTE["target"], lw=0.85, ls="--")
    ax.text(0.98, 1.0, "1%目标", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.8, color=PALETTE["target"])
    ax.set_xlabel("MAE/(kg/s)")
    ax.set_ylabel("累计误差/%")
    ax.set_title("目标现场总体误差", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(color=PALETTE["grid"], linewidth=0.55)
    add_panel_label(ax, "a")


def panel_calibration(ax: plt.Axes, calibration: pd.DataFrame) -> None:
    df = calibration[calibration["method"].str.contains("mass-scale", regex=False)].copy()
    strategies = ["time_uniform", "prediction_stratified", "load_stratified", "random"]
    labels = {"time_uniform": "Time", "prediction_stratified": "Pred.", "load_stratified": "Load", "random": "Rand."}
    colors = ["#7A869A", "#2E6F9E", "#4C8B6B", "#C46B37"]
    budgets = [5, 20, 60, 180]
    width = 0.18
    xs = list(range(len(budgets)))
    for idx, strategy in enumerate(strategies):
        vals = []
        for budget in budgets:
            hit = df[(df["budget"].eq(budget)) & (df["strategy"].eq(strategy))]
            vals.append(float(hit.iloc[0]["cum_err_pct"]) if not hit.empty else float("nan"))
        ax.bar([x + (idx - 1.5) * width for x in xs], vals, width=width, color=colors[idx], edgecolor="#333333", linewidth=0.35, label=labels[strategy])
    ax.axhline(1.0, color=PALETTE["target"], lw=0.85, ls="--")
    ax.text(0.98, 1.0, "1%目标", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.8, color=PALETTE["target"])
    ax.set_xticks(xs)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.set_xlabel("校准样本/安装")
    ax.set_ylabel("累计误差/%")
    ax.set_title("累计质量比例锚定策略", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    ax.legend(loc="upper right", ncol=2, fontsize=5.8)
    add_panel_label(ax, "b")


def panel_strata(ax: plt.Axes, stratified: pd.DataFrame) -> None:
    method = "PG-TSC-real-mass-scale budget=5 strategy=prediction_stratified"
    df = stratified[stratified["method"].eq(method)].copy()
    split = df["split"].astype(str).str.split("=", n=1, expand=True)
    df["dimension"] = split[0]
    df["group"] = split[1]
    df = df[df["dimension"].isin(["env_temp_c", "env_humidity_pct", "site_id", "ash_missing"])].copy()
    order = {"env_temp_c": 0, "env_humidity_pct": 1, "site_id": 2, "ash_missing": 3}
    short = {"env_temp_c": "Temp", "env_humidity_pct": "Hum.", "site_id": "Site", "ash_missing": "AshMiss"}
    df["sort"] = df["dimension"].map(order)
    df["label"] = df["dimension"].map(short) + ":" + df["group"].astype(str)
    df = df.sort_values(["sort", "group"])
    ax.bar(range(len(df)), df["cum_err_pct"], color=PALETTE["pgt"], edgecolor="#333333", linewidth=0.35, width=0.68)
    ax.axhline(1.0, color=PALETTE["target"], lw=0.85, ls="--")
    ax.text(0.98, 1.0, "1%目标", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.8, color=PALETTE["target"])
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["label"], rotation=42, ha="right")
    ax.set_ylabel("累计误差/%")
    ax.set_title("PG-5分层累计误差", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    add_panel_label(ax, "c")


def panel_install(ax: plt.Axes, install: pd.DataFrame) -> None:
    methods = [
        "Physics-Temporal-Ridge",
        "HGB",
        "PG-TSC-real-mass-scale budget=5 strategy=prediction_stratified",
        "PG-TSC-real-mass-scale budget=60 strategy=prediction_stratified",
    ]
    labels = ["Temporal", "HGB", "PG-5", "PG-60"]
    data = [install[install["method"].eq(method)]["cum_err_pct"].astype(float).to_numpy() for method in methods]
    bp = ax.boxplot(data, patch_artist=True, widths=0.54, showfliers=True)
    colors = [PALETTE["temporal"], PALETTE["hgb"], PALETTE["pgt"], "#D8945A"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.78)
        patch.set_edgecolor("#333333")
    for median in bp["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.0)
    ax.axhline(1.0, color=PALETTE["target"], lw=0.85, ls="--")
    ax.text(0.98, 1.0, "1%目标", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=5.8, color=PALETTE["target"])
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("安装级累计误差/%")
    ax.set_title("安装级误差分布", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    add_panel_label(ax, "d")


def panel_anomaly(ax: plt.Axes, qc: pd.DataFrame) -> None:
    keep = ["Physics-Temporal-Ridge", "HGB", "RandomForest"]
    rows = qc[qc["method"].isin(keep)].copy()
    rows["label"] = rows["method"].map({"Physics-Temporal-Ridge": "Temporal", "HGB": "HGB", "RandomForest": "RF"})
    ax.bar(rows["label"], rows["cum_err_pct"], color=[method_color(m) for m in rows["method"]], edgecolor="#333333", linewidth=0.45)
    ax.set_ylabel("异常留出累计误差/%")
    ax.set_title("异常样本风险边界", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    add_panel_label(ax, "e")


def panel_cleaning(ax: plt.Axes, cleaning: pd.DataFrame) -> None:
    lookup = dict(zip(cleaning["metric"], cleaning["value"]))
    labels = ["empty", "bad", "cont.", "excluded", "ash fill"]
    values = [
        float(lookup["empty_rows"]),
        float(lookup["bad_point_rows"]),
        float(lookup["continuous_anomaly_rows"]),
        float(lookup["train_excluded_rows"]),
        float(lookup["ash_filled_rows"]),
    ]
    ax.bar(labels, values, color=["#6F7686", "#B54E4E", "#C46B37", "#8A5A9E", "#4C8B6B"], edgecolor="#333333", linewidth=0.45)
    ax.set_ylabel("样本数")
    ax.set_title("清洗与质量控制", fontsize=8.2, fontweight="bold", pad=5)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.55)
    add_panel_label(ax, "f")


def save_tiff_from_png(png_path: Path, tiff_path: Path) -> None:
    with Image.open(png_path) as image:
        image.save(tiff_path, compression="tiff_lzw", dpi=(450, 450))


def main() -> int:
    args = parse_args()
    setup_style()
    out_base = Path(args.out_base)
    if not out_base.is_absolute():
        out_base = ROOT / out_base
    out_base.parent.mkdir(parents=True, exist_ok=True)

    summary_path = RESULTS / "real_xray_experiment_summary.csv"
    calibration_path = RESULTS / "real_xray_calibration_strategy.csv"
    stratified_path = RESULTS / "real_xray_stratified_errors.csv"
    install_path = RESULTS / "real_xray_install_errors.csv"
    qc_path = RESULTS / "real_xray_qc_holdout_errors.csv"
    cleaning_path = RESULTS / "real_data_cleaning_report.csv"

    summary = pd.read_csv(summary_path)
    calibration = pd.read_csv(calibration_path)
    stratified = pd.read_csv(stratified_path)
    install = pd.read_csv(install_path)
    qc = pd.read_csv(qc_path)
    cleaning = pd.read_csv(cleaning_path)

    fig = plt.figure(figsize=(8.2, 7.4))
    fig.suptitle("真实X射线皮带CSV补充验证：少样本累计质量锚定与风险边界", x=0.02, y=0.99, ha="left", fontsize=9.0, fontweight="bold")
    gs = fig.add_gridspec(3, 2, hspace=0.58, wspace=0.34)
    panel_main(fig.add_subplot(gs[0, 0]), summary)
    panel_calibration(fig.add_subplot(gs[0, 1]), calibration)
    panel_strata(fig.add_subplot(gs[1, 0]), stratified)
    panel_install(fig.add_subplot(gs[1, 1]), install)
    panel_anomaly(fig.add_subplot(gs[2, 0]), qc)
    panel_cleaning(fig.add_subplot(gs[2, 1]), cleaning)
    fig.text(
        0.015,
        0.010,
        "数据源：真实X射线煤炭传送带数据；原始CSV不进入投稿包，图中异常留出结果用于风险触发边界。",
        fontsize=6.2,
        color=PALETTE["muted"],
    )

    svg_path = out_base.with_suffix(".svg")
    png_path = out_base.with_suffix(".png")
    pdf_path = out_base.with_suffix(".pdf")
    tiff_path = out_base.with_suffix(".tiff")
    fig.savefig(svg_path, bbox_inches="tight")
    strip_trailing_space(svg_path)
    fig.savefig(png_path, dpi=450, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    save_tiff_from_png(png_path, tiff_path)

    meta = write_run_metadata(
        svg_path,
        script=__file__,
        args={"out_base": out_base.relative_to(ROOT).as_posix()},
        inputs=[summary_path, calibration_path, stratified_path, install_path, qc_path, cleaning_path],
        outputs=[svg_path, png_path, pdf_path, tiff_path],
        notes="Real X-ray belt experiment completion figure. Supporting material for real-data experiment report.",
    )
    print(f"wrote={svg_path}")
    print(f"wrote={png_path}")
    print(f"wrote={pdf_path}")
    print(f"wrote={tiff_path}")
    print(f"wrote={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
