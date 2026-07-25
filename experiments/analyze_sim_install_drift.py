from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiment_versioning import write_run_metadata
from figure_style import METHOD_COLORS, PALETTE, add_target_line, save_figure, setup_publication_style, soften_axes


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = Path("paper/figures")
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT).as_posix()


def setup_style() -> None:
    setup_publication_style(font_size=7.4)


def cumulative_error(g: pd.DataFrame, pred_col: str) -> float:
    y_true = g["true_flow_kgps"]
    y_pred = g[pred_col].clip(lower=0)
    return abs(y_pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-9) * 100


def make_install_table(pred: pd.DataFrame) -> None:
    methods = {
        "纯数据HGB": "DataOnly-HGB",
        "物理+岭回归": "Physics-Ridge",
        "物理+时序": "Physics-Temporal-Ridge",
    }
    rows: list[dict[str, object]] = []
    for (device_id, install_id), g in pred.groupby(["device_id", "install_id"], sort=True):
        row: dict[str, object] = {
            "设备": device_id,
            "安装": install_id,
            "样本数": len(g),
            "真实累计重量/kg": float(g["true_flow_kgps"].sum()),
        }
        for label, col in methods.items():
            row[f"{label}/%"] = cumulative_error(g, col)
        rows.append(row)
    out = Path("results/sim_install_errors.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    meta = write_run_metadata(
        out,
        script=SCRIPT_PATH,
        args={},
        inputs=["results/sim_predictions.csv"],
        outputs=[out],
        notes="Per-device and per-install cumulative-error table.",
    )
    print(f"wrote={out}")
    print(f"wrote={meta}")


def plot_drift_curve(pred: pd.DataFrame) -> None:
    g = pred[(pred["device_id"].eq("D01")) & (pred["install_id"].eq("I01"))].copy()
    g = g.sort_values("timestamp_s").iloc[:3600]
    g["true_cum"] = g["true_flow_kgps"].cumsum()
    methods = [
        ("DataOnly-HGB", "纯数据HGB", METHOD_COLORS["纯数据HGB"]),
        ("Physics-Ridge", "物理+岭回归", METHOD_COLORS["物理+岭回归"]),
        ("Physics-Temporal-Ridge", "物理+时序", METHOD_COLORS["物理+时序"]),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 3.25))
    valid = (g["true_cum"] > 50000) & (g["timestamp_s"] > 300)
    for col, label, color in methods:
        pred_cum = g[col].clip(lower=0).cumsum()
        err = (pred_cum - g["true_cum"]).abs() / g["true_cum"].clip(lower=1) * 100
        ax.plot(g.loc[valid, "timestamp_s"], err.loc[valid], label=label, color=color, linewidth=1.4)
        ax.text(
            float(g.loc[valid, "timestamp_s"].iloc[-1]) + 45,
            float(err.loc[valid].iloc[-1]),
            label,
            fontsize=6.4,
            va="center",
            color=color,
        )
    ax.set_xlabel("时间/s")
    ax.set_ylabel("累计相对误差/%")
    ax.set_title("长期累计误差随时间变化", loc="left")
    ax.set_ylim(0, 15)
    ax.set_xlim(float(g.loc[valid, "timestamp_s"].min()), float(g.loc[valid, "timestamp_s"].max()) + 520)
    ax.axhspan(0, 1.0, color="#EDF6EF", zorder=0)
    add_target_line(ax)
    soften_axes(ax, grid_axis="both")
    fig.text(0.01, 0.018, "案例曲线用于展示漂移趋势，整体排名以安装级表格和跨现场统计为准。", fontsize=6.2, color=PALETTE["muted"])
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    svg = FIG_DIR / "fig6_drift_curve.svg"
    save_figure(fig, FIG_DIR / "fig6_drift_curve", png_dpi=450, tiff_dpi=450, pdf=False)
    plt.close(fig)
    meta = write_run_metadata(
        svg,
        script=SCRIPT_PATH,
        args={},
        inputs=["results/sim_predictions.csv"],
        outputs=[svg, FIG_DIR / "fig6_drift_curve.png"],
        notes="Long-time cumulative relative error curve used as Figure 6.",
    )
    print(f"wrote={svg}")
    print(f"wrote={meta}")


def main() -> None:
    setup_style()
    pred = pd.read_csv("results/sim_predictions.csv")
    make_install_table(pred)
    plot_drift_curve(pred)


if __name__ == "__main__":
    main()
