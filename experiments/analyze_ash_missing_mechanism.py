from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiment_versioning import write_run_metadata


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/simulated/xray_belt_sim.csv")
RESULTS = Path("results")
FIG_DIR = Path("paper/figures")
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT).as_posix()
COAL_ORDER = ["anthracite", "bituminous_A", "bituminous_B", "lignite"]
STRESS_COLUMNS = ["experiment", "scenario", "setting", "level", "mae", "rmse", "cum_err_pct"]
COAL_MEANS = {
    "anthracite": 9.0,
    "bituminous_A": 12.0,
    "bituminous_B": 18.0,
    "lignite": 24.0,
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Noto Sans SC",
            "font.sans-serif": ["Noto Sans SC", "SimHei", "Microsoft YaHei", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def normalize_svg(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    path.write_text("\n".join(line.rstrip() for line in text.splitlines()) + "\n", encoding="utf-8")


def add_online_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for _, idx in out.groupby(["site_id", "device_id", "install_id"], sort=False).groups.items():
        g = out.loc[idx].sort_values("timestamp_s")
        rx = g["rx_value"].astype(float).to_numpy()
        empty = g["empty_flag"].astype(int).to_numpy()
        empty_rx = rx[empty == 1]
        i0 = float(np.median(empty_rx)) if len(empty_rx) else float(np.percentile(rx, 95))
        i0_values: list[float] = []
        attenuation: list[float] = []
        for value, is_empty in zip(rx, empty, strict=True):
            if is_empty:
                i0 = 0.985 * i0 + 0.015 * float(value)
            att = max(0.0, np.log(max(i0, value + 1e-6) / max(value, 1.0)))
            i0_values.append(i0)
            attenuation.append(float(att))
        s_att = pd.Series(attenuation, index=g.index)
        s_rx = pd.Series(rx, index=g.index)
        out.loc[g.index, "i0_online"] = i0_values
        out.loc[g.index, "attenuation"] = attenuation
        out.loc[g.index, "attenuation_mean30"] = s_att.rolling(30, min_periods=1).mean()
        out.loc[g.index, "attenuation_std30"] = s_att.rolling(30, min_periods=1).std(ddof=0)
        out.loc[g.index, "rx_mean30"] = s_rx.rolling(30, min_periods=1).mean()
        out.loc[g.index, "rx_std30"] = s_rx.rolling(30, min_periods=1).std(ddof=0)
    out["phys_flow_feature"] = out["attenuation"].astype(float) * out["belt_speed_mps"].astype(float)
    return out


def feature_frame(df: pd.DataFrame, *, include_missing: bool) -> pd.DataFrame:
    cols = [
        "attenuation",
        "phys_flow_feature",
        "i0_online",
        "tube_voltage_kv",
        "tube_current_ma",
        "source_temp_c",
        "env_temp_c",
        "env_humidity_pct",
        "belt_speed_mps",
        "ash_pct",
    ]
    x = df[cols].astype(float).copy()
    coal = pd.get_dummies(df["coal_type"], prefix="coal", dtype=float)
    coal = coal.reindex(columns=[f"coal_{name}" for name in COAL_ORDER], fill_value=0.0)
    x = pd.concat([x, coal], axis=1)
    if include_missing:
        x["ash_missing"] = df["ash_missing"].astype(float)
    temporal = df[["attenuation_mean30", "attenuation_std30", "rx_mean30", "rx_std30"]].astype(float)
    return pd.concat([x, temporal], axis=1)


def metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    pred = np.maximum(y_pred, 0.0)
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(mean_squared_error(y_true, pred) ** 0.5),
        "cum_err_pct": float(abs(pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-9) * 100),
    }


def apply_missing(df: pd.DataFrame, *, mechanism: str, rate: float, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(seed)
    high_ash = out["coal_type"].isin(["lignite", "bituminous_B"]).to_numpy()
    site_s04 = out["site_id"].eq("S04").to_numpy()
    if mechanism == "random":
        p = np.full(len(out), rate, dtype=float)
    elif mechanism == "coal_biased":
        p = rate * np.where(high_ash, 1.8, 0.55)
    elif mechanism == "site_biased":
        p = rate * np.where(site_s04, 1.9, 0.45)
    elif mechanism == "coal_site_biased":
        p = rate * np.select(
            [site_s04 & high_ash, site_s04 | high_ash],
            [2.2, 0.95],
            default=0.25,
        )
    else:
        raise ValueError(f"unknown mechanism: {mechanism}")
    missing = rng.random(len(out)) < np.clip(p, 0.0, 0.97)
    means = out["coal_type"].map(COAL_MEANS).astype(float).to_numpy()
    fill_bias = np.where(high_ash, -2.4 * rate, 1.1 * rate)
    if mechanism == "coal_site_biased":
        fill_bias = np.where(site_s04, fill_bias * 1.35, fill_bias)
    out["ash_pct"] = np.where(missing, means + fill_bias, out["ash_pct"].astype(float).to_numpy())
    out["ash_missing"] = missing.astype(float)
    return out


def run_ash_experiment(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for mechanism in ["random", "coal_biased", "site_biased", "coal_site_biased"]:
        for rate in [0.2, 0.4, 0.6]:
            shifted = apply_missing(df, mechanism=mechanism, rate=rate, seed=20260623)
            train = shifted[shifted["site_id"].isin(["S01", "S02", "S03"])].copy()
            test = shifted[shifted["site_id"].eq("S04")].copy()
            y_train = train["true_flow_kgps"].astype(float).to_numpy()
            y_test = test["true_flow_kgps"].astype(float).to_numpy()
            for include_missing, setting in [(True, "with_missing_flag"), (False, "without_missing_flag")]:
                x_train = feature_frame(train, include_missing=include_missing)
                x_test = feature_frame(test, include_missing=include_missing).reindex(columns=x_train.columns, fill_value=0.0)
                model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                rows.append(
                    {
                        "experiment": "ash_missing_mechanism",
                        "scenario": mechanism,
                        "setting": setting,
                        "level": rate,
                        **metric_row(y_test, pred),
                    }
                )
    return pd.DataFrame(rows)


def write_results(ash_rows: pd.DataFrame) -> Path:
    out = RESULTS / "sim_stress_tests.csv"
    previous = pd.read_csv(out)
    kept = previous[~previous["experiment"].eq("ash_missing_mechanism")].copy()
    merged = pd.concat([kept, ash_rows], ignore_index=True, sort=False)
    merged = merged.reindex(columns=STRESS_COLUMNS)
    merged.to_csv(out, index=False)
    write_run_metadata(
        out,
        script=SCRIPT_PATH,
        args={},
        inputs=[DATA],
        outputs=[out],
        notes="Fast controlled ash-missingness supplement; non-ash stress-test rows are preserved from the prior stress-test table.",
    )
    return out


def plot_results(df: pd.DataFrame) -> Path:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7), constrained_layout=True)

    empty = df[df["experiment"].eq("empty_gate_error")].sort_values("level")
    axes[0].plot(empty["level"] * 100, empty["cum_err_pct"], marker="o", color="#4f81bd", linewidth=1.5)
    axes[0].set_title("(a) 空载误判压力", loc="left", fontsize=9)
    axes[0].set_xlabel("有载误判为空载比例/%")
    axes[0].set_ylabel("累计误差/%")
    axes[0].grid(axis="y", color="#dddddd", linewidth=0.5)

    ash = df[(df["experiment"].eq("ash_missing_mechanism")) & (df["scenario"].eq("coal_site_biased"))].copy()
    for setting, label, color in [
        ("with_missing_flag", "使用缺失标记", "#3a8f6b"),
        ("without_missing_flag", "不使用缺失标记", "#b36b5e"),
    ]:
        g = ash[ash["setting"].eq(setting)].sort_values("level")
        axes[1].plot(g["level"] * 100, g["cum_err_pct"], marker="o", label=label, color=color, linewidth=1.5)
    axes[1].set_title("(b) 灰分非随机缺失", loc="left", fontsize=9)
    axes[1].set_xlabel("目标缺失率/%")
    axes[1].grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[1].legend(fontsize=7)

    hidden = df[df["experiment"].eq("hidden_moisture_particle")].copy()
    for setting, label, color in [
        ("Physics-Ridge", "物理+岭回归", "#7c8798"),
        ("Physics-Temporal-Ridge", "物理+时序", "#6b5fb5"),
    ]:
        g = hidden[hidden["setting"].eq(setting)].sort_values("level")
        axes[2].plot(g["level"], g["cum_err_pct"], marker="o", label=label, color=color, linewidth=1.5)
    axes[2].set_title("(c) 隐藏水分/粒度扰动", loc="left", fontsize=9)
    axes[2].set_xlabel("隐藏扰动强度")
    axes[2].grid(axis="y", color="#dddddd", linewidth=0.5)
    axes[2].legend(fontsize=7)

    svg = FIG_DIR / "fig7_stress_tests.svg"
    png = FIG_DIR / "fig7_stress_tests.png"
    fig.savefig(svg)
    normalize_svg(svg)
    fig.savefig(png, dpi=300)
    plt.close(fig)
    write_run_metadata(
        svg,
        script=SCRIPT_PATH,
        args={},
        inputs=[RESULTS / "sim_stress_tests.csv"],
        outputs=[svg, png],
        notes="Figure 7 updated with a controlled non-random ash-missingness stress-test panel.",
    )
    return svg


def main() -> None:
    df = pd.read_csv(DATA)
    df = add_online_features(df)
    ash_rows = run_ash_experiment(df)
    out = write_results(ash_rows)
    full = pd.read_csv(out)
    fig = plot_results(full)
    focus = ash_rows[(ash_rows["scenario"].eq("coal_site_biased")) & (ash_rows["level"].eq(0.6))]
    print(f"wrote={out}")
    print(f"wrote={fig}")
    for _, row in focus.sort_values("setting").iterrows():
        print(f"{row['setting']}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, CumErr={row['cum_err_pct']:.3f}%")


if __name__ == "__main__":
    main()
