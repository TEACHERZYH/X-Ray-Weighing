from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiment_versioning import write_run_metadata
from figure_style import METHOD_COLORS, PALETTE, add_target_line, save_figure, setup_publication_style, soften_axes


FIG_DIR = Path("paper/figures")


def setup_style() -> None:
    setup_publication_style(font_size=7.4)


def cumulative_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return abs(y_pred.clip(lower=0).sum() - y_true.sum()) / max(y_true.sum(), 1e-9) * 100


def make_stratified_results() -> None:
    raw_cols = [
        "timestamp_s",
        "site_id",
        "device_id",
        "install_id",
        "coal_type",
        "env_temp_c",
        "env_humidity_pct",
    ]
    raw = pd.read_csv("data/simulated/xray_belt_sim.csv", usecols=raw_cols)
    pred = pd.read_csv("results/sim_predictions.csv")
    df = pred.merge(raw, on=["timestamp_s", "site_id", "device_id", "install_id"], how="left")

    df["temp_bin"] = pd.cut(df["env_temp_c"], bins=[-100, 18, 26, 100], labels=["低温", "中温", "高温"])
    df["humidity_bin"] = pd.cut(df["env_humidity_pct"], bins=[0, 45, 65, 100], labels=["低湿", "中湿", "高湿"])

    methods = {
        "纯数据HGB": "DataOnly-HGB",
        "物理+岭回归": "Physics-Ridge",
        "物理+时序": "Physics-Temporal-Ridge",
    }
    rows: list[dict[str, object]] = []
    for group_name, col in [("温度区间", "temp_bin"), ("湿度区间", "humidity_bin"), ("煤种", "coal_type")]:
        for group_value, g in df.groupby(col, observed=True):
            row: dict[str, object] = {"分组": group_name, "类别": str(group_value), "样本数": len(g)}
            for label, pred_col in methods.items():
                row[label] = cumulative_error(g["true_flow_kgps"], g[pred_col])
            rows.append(row)
    out = Path("results/sim_stratified_errors.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    meta = write_run_metadata(
        out,
        script=__file__,
        args={},
        inputs=["data/simulated/xray_belt_sim.csv", "results/sim_predictions.csv"],
        outputs=[out],
        notes="Stratified cumulative-error analysis by temperature, humidity, and coal type.",
    )
    print(f"wrote={out}")
    print(f"wrote={meta}")


def plot_calibration_curve() -> None:
    base = pd.read_csv("results/sim_baselines.csv")
    tcn = pd.read_csv("results/sim_tcn_results.csv")

    rows = []
    for _, r in base.iterrows():
        name = str(r["method"])
        if name.startswith("PG-TSC-lite calibrate="):
            rows.append(
                {
                    "样本数": int(name.split("=")[1]),
                    "累计误差": float(r["cum_err_pct"]),
                    "方法": "PG-TSC-lite",
                }
            )
    for _, r in tcn.iterrows():
        name = str(r["method"])
        if name == "PG-TSC-TCN":
            rows.append(
                {
                    "样本数": 0,
                    "累计误差": float(r["cum_err_pct"]),
                    "方法": "PG-TSC-TCN",
                }
            )
        if name.startswith("PG-TSC-TCN calibrate="):
            rows.append(
                {
                    "样本数": int(name.split("=")[1]),
                    "累计误差": float(r["cum_err_pct"]),
                    "方法": "PG-TSC-TCN",
                }
            )
    df = pd.DataFrame(rows).sort_values(["方法", "样本数"])
    out = Path("results/sim_calibration_curve.csv")
    df.to_csv(out, index=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sample_order = [0, 5, 20, 60, 180]
    x_pos = {v: i for i, v in enumerate(sample_order)}

    fig, ax = plt.subplots(figsize=(6.5, 3.15))
    for method, g in df.groupby("方法"):
        g = g[g["样本数"].isin(sample_order)].sort_values("样本数")
        ax.plot(
            [x_pos[v] for v in g["样本数"]],
            g["累计误差"],
            marker="o",
            linewidth=1.65,
            markersize=4.2,
            label=method,
            color=METHOD_COLORS.get(method, PALETTE["muted"]),
        )
        last = g.iloc[-1]
        ax.text(x_pos[int(last["样本数"])] + 0.05, float(last["累计误差"]), method, fontsize=6.4, va="center")
    ax.set_xlabel("校准样本数")
    ax.set_ylabel("累计误差/%")
    ax.set_title("设备级少样本校准曲线", loc="left")
    ax.set_xticks(range(len(sample_order)))
    ax.set_xticklabels([str(v) for v in sample_order])
    ax.axhspan(0, 1.0, color="#EDF6EF", zorder=0)
    add_target_line(ax)
    soften_axes(ax, grid_axis="both")
    ax.legend(loc="upper right", fontsize=6.4)
    fig.text(0.01, 0.018, "样本数增加不保证单调改善，代表性覆盖仍是安装校准关键。", fontsize=6.2, color=PALETTE["muted"])
    save_figure(fig, FIG_DIR / "fig5_calibration_curve", png_dpi=450, tiff_dpi=450, pdf=False)
    plt.close(fig)
    meta = write_run_metadata(
        out,
        script=__file__,
        args={},
        inputs=["results/sim_baselines.csv", "results/sim_tcn_results.csv"],
        outputs=[out, FIG_DIR / "fig5_calibration_curve.svg", FIG_DIR / "fig5_calibration_curve.png"],
        notes="Few-sample calibration curve used as Figure 5.",
    )
    print(f"wrote={out}")
    print(f"wrote={meta}")
    print(f"wrote={FIG_DIR / 'fig5_calibration_curve.svg'}")


def main() -> None:
    setup_style()
    make_stratified_results()
    plot_calibration_curve()


if __name__ == "__main__":
    main()
