from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_versioning import write_run_metadata
from figure_style import PALETTE, add_panel_label, add_target_line, save_figure, setup_publication_style, soften_axes
from run_sim_baselines import COAL_ORDER, add_online_features, feature_matrix, metrics, read_rows, ridge_fit, ridge_predict


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/simulated/xray_belt_sim.csv")
RESULTS = Path("results")
FIG_DIR = Path("paper/figures")
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT).as_posix()
COAL_MEANS = {
    "anthracite": 9.0,
    "bituminous_A": 12.0,
    "bituminous_B": 18.0,
    "lignite": 24.0,
}


def setup_style() -> None:
    setup_publication_style(font_size=7.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Fig. 7 stress-test boundaries.")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute results/sim_stress_tests.csv instead of using the controlled result table.",
    )
    return parser.parse_args()


def split_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [r for r in rows if str(r["site_id"]) in {"S01", "S02", "S03"}]
    test = [r for r in rows if str(r["site_id"]) == "S04"]
    return train, test


def copy_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row.copy() for row in rows]


def fit_clean_model(rows: list[dict[str, object]], *, include_missing: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train, _ = split_rows(rows)
    x_train = feature_matrix(train, include_physics=True, include_missing_flag=include_missing, temporal=True)
    y_train = np.array([float(r["true_flow_kgps"]) for r in train])
    return ridge_fit(x_train, y_train, lam=10.0)


def predict_test(rows: list[dict[str, object]], model: tuple[np.ndarray, np.ndarray, np.ndarray], *, include_missing: bool = True) -> tuple[np.ndarray, np.ndarray]:
    _, test = split_rows(rows)
    x_test = feature_matrix(test, include_physics=True, include_missing_flag=include_missing, temporal=True)
    y_test = np.array([float(r["true_flow_kgps"]) for r in test])
    beta, mean, std = model
    return y_test, ridge_predict(x_test, beta, mean, std)


def feature_matrix_fast(rows: list[dict[str, object]], *, include_missing: bool) -> np.ndarray:
    df = pd.DataFrame(rows)
    df["phys_flow_feature"] = df["attenuation"].astype(float) * df["belt_speed_mps"].astype(float)
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
    matrix = [df[cols].astype(float).to_numpy()]
    coal = pd.get_dummies(df["coal_type"], prefix="coal", dtype=float)
    coal = coal.reindex(columns=[f"coal_{name}" for name in COAL_ORDER], fill_value=0.0)
    matrix.append(coal.to_numpy(dtype=float))
    if include_missing:
        matrix.append(df[["ash_missing"]].astype(float).to_numpy())
    matrix.append(df[["attenuation_mean30", "attenuation_std30", "rx_mean30", "rx_std30"]].astype(float).to_numpy())
    return np.column_stack(matrix)


def add_online_features_with_gate_error(rows: list[dict[str, object]], false_positive_rate: float, seed: int = 20260623) -> None:
    rng = np.random.default_rng(seed)
    for r in rows:
        empty = int(float(r["empty_flag"]))
        if str(r["site_id"]) == "S04" and empty == 0 and rng.random() < false_positive_rate:
            r["empty_flag"] = 1.0
    add_online_features(rows)


def apply_hidden_disturbance(rows: list[dict[str, object]], severity: float) -> None:
    for r in rows:
        if str(r["site_id"]) != "S04":
            continue
        loading = float(r["true_loading_kgpm"])
        humidity = float(r["env_humidity_pct"])
        coal_factor = 1.20 if str(r["coal_type"]) == "lignite" else 1.0
        hidden = severity * coal_factor * (loading / 100.0) * (1.0 + max(humidity - 55.0, 0.0) / 80.0)
        r["rx_value"] = max(1.0, float(r["rx_value"]) * np.exp(-hidden))


def apply_ash_missing(rows: list[dict[str, object]], mechanism: str, rate: float, seed: int = 20260623) -> None:
    rng = np.random.default_rng(seed)
    for r in rows:
        coal = str(r["coal_type"])
        high_ash = coal in {"lignite", "bituminous_B"}
        site = str(r["site_id"])
        missing = False
        if mechanism == "random":
            missing = rng.random() < rate
        elif mechanism == "coal_biased":
            coal_rate = min(0.95, rate * (1.8 if high_ash else 0.55))
            missing = rng.random() < coal_rate
        elif mechanism == "site_biased":
            site_rate = min(0.95, rate * (1.9 if site == "S04" else 0.45))
            missing = rng.random() < site_rate
        elif mechanism == "coal_site_biased":
            if site == "S04" and high_ash:
                p = rate * 2.2
            elif site == "S04" or high_ash:
                p = rate * 0.95
            else:
                p = rate * 0.25
            missing = rng.random() < min(0.97, p)
        else:
            raise ValueError(f"unknown mechanism: {mechanism}")
        if missing:
            r["ash_missing"] = 1.0
            # Field practice often fills missing ash with a coal-type mean. The
            # controlled bias below makes the fill value visibly lower-confidence
            # than a measured value without changing the hidden true coal physics.
            fill_bias = (-2.4 if high_ash else 1.1) * rate
            if mechanism == "coal_site_biased" and site == "S04":
                fill_bias *= 1.35
            r["ash_pct"] = COAL_MEANS.get(coal, float(r["ash_pct"])) + fill_bias
        else:
            r["ash_missing"] = 0.0


def run_empty_gate_stress(clean_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    clean = copy_rows(clean_rows)
    add_online_features(clean)
    model = fit_clean_model(clean)
    out: list[dict[str, object]] = []
    for rate in [0.0, 0.01, 0.03, 0.05, 0.10]:
        rows = copy_rows(clean_rows)
        add_online_features_with_gate_error(rows, rate)
        y_true, pred = predict_test(rows, model)
        out.append(
            {
                "experiment": "empty_gate_error",
                "scenario": f"false_positive_{rate:.2f}",
                "setting": "online_i0_gate",
                "level": rate,
                **metrics(y_true, pred),
            }
        )
    return out


def run_hidden_disturbance_stress(clean_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    clean = copy_rows(clean_rows)
    add_online_features(clean)
    configs = [
        ("Physics-Ridge", False),
        ("Physics-Temporal-Ridge", True),
    ]
    models: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, temporal in configs:
        train, _ = split_rows(clean)
        x_train = feature_matrix(train, include_physics=True, include_missing_flag=True, temporal=temporal)
        y_train = np.array([float(r["true_flow_kgps"]) for r in train])
        models[name] = ridge_fit(x_train, y_train, lam=10.0)

    out: list[dict[str, object]] = []
    for severity, label in [(0.0, "none"), (0.008, "low"), (0.016, "medium"), (0.032, "high")]:
        rows = copy_rows(clean_rows)
        apply_hidden_disturbance(rows, severity)
        add_online_features(rows)
        _, test = split_rows(rows)
        y_true = np.array([float(r["true_flow_kgps"]) for r in test])
        for name, temporal in configs:
            x_test = feature_matrix(test, include_physics=True, include_missing_flag=True, temporal=temporal)
            pred = ridge_predict(x_test, *models[name])
            out.append(
                {
                    "experiment": "hidden_moisture_particle",
                    "scenario": label,
                    "setting": name,
                    "level": severity,
                    **metrics(y_true, pred),
                }
            )
    return out


def run_ash_missing_stress(clean_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    clean = copy_rows(clean_rows)
    add_online_features(clean)
    models = {
        True: fit_clean_model(clean, include_missing=True),
        False: fit_clean_model(clean, include_missing=False),
    }
    out: list[dict[str, object]] = []
    seeds = [20260623]
    for mechanism in ["random", "coal_biased", "site_biased", "coal_site_biased"]:
        for rate in [0.2, 0.4, 0.6]:
            grouped: dict[str, list[dict[str, float]]] = {"with_missing_flag": [], "without_missing_flag": []}
            for seed in seeds:
                rows = copy_rows(clean_rows)
                apply_ash_missing(rows, mechanism, rate, seed=seed)
                add_online_features(rows)
                _, test = split_rows(rows)
                y_test = np.array([float(r["true_flow_kgps"]) for r in test])
                for include_missing, setting in [(True, "with_missing_flag"), (False, "without_missing_flag")]:
                    x_test = feature_matrix_fast(test, include_missing=include_missing)
                    pred = ridge_predict(x_test, *models[include_missing])
                    grouped[setting].append(metrics(y_test, pred))
            for setting, metric_rows in grouped.items():
                cum_values = np.array([m["cum_err_pct"] for m in metric_rows], dtype=float)
                mae_values = np.array([m["mae"] for m in metric_rows], dtype=float)
                rmse_values = np.array([m["rmse"] for m in metric_rows], dtype=float)
                out.append(
                    {
                        "experiment": "ash_missing_mechanism",
                        "scenario": mechanism,
                        "setting": setting,
                        "level": rate,
                        "mae": float(np.median(mae_values)),
                        "rmse": float(np.median(rmse_values)),
                        "cum_err_pct": float(np.median(cum_values)),
                        "seed_count": len(seeds),
                        "cum_err_iqr": float(np.percentile(cum_values, 75) - np.percentile(cum_values, 25)),
                    }
                )
    return out


def write_table(rows: list[dict[str, object]]) -> Path:
    out = RESULTS / "sim_stress_tests.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    write_run_metadata(
        out,
        script=SCRIPT_PATH,
        args={},
        inputs=[DATA],
        outputs=[out],
        notes="Simulation stress tests for empty-gate errors, missing ash mechanisms, and hidden moisture/particle-size disturbances.",
    )
    return out


def plot_results(df: pd.DataFrame) -> Path:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.85), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.995, bottom=0.26, top=0.84, wspace=0.34)

    empty = df[df["experiment"].eq("empty_gate_error")].sort_values("level")
    axes[0].plot(empty["level"] * 100, empty["cum_err_pct"], marker="o", color=PALETTE["physics"], linewidth=1.55)
    axes[0].set_title("空载误判压力", loc="left")
    axes[0].set_xlabel("有载误判为空载比例/%")
    axes[0].set_ylabel("累计误差/%")
    add_target_line(axes[0])
    soften_axes(axes[0])
    add_panel_label(axes[0], "a")

    ash = df[(df["experiment"].eq("ash_missing_mechanism")) & (df["scenario"].eq("coal_site_biased"))].copy()
    for setting, label, color in [
        ("with_missing_flag", "使用缺失标记", PALETTE["temporal"]),
        ("without_missing_flag", "不使用缺失标记", PALETTE["risk"]),
    ]:
        g = ash[ash["setting"].eq(setting)].sort_values("level")
        axes[1].plot(g["level"] * 100, g["cum_err_pct"], marker="o", label=label, color=color, linewidth=1.5)
    axes[1].set_title("灰分非随机缺失", loc="left")
    axes[1].set_xlabel("目标缺失率/%")
    add_target_line(axes[1])
    soften_axes(axes[1])
    add_panel_label(axes[1], "b")
    axes[1].legend(fontsize=7)

    hidden = df[df["experiment"].eq("hidden_moisture_particle")].copy()
    for setting, label, color in [
        ("Physics-Ridge", "物理+岭回归", PALETTE["physics"]),
        ("Physics-Temporal-Ridge", "物理+时序", PALETTE["temporal"]),
    ]:
        g = hidden[hidden["setting"].eq(setting)].sort_values("level")
        axes[2].plot(g["level"], g["cum_err_pct"], marker="o", label=label, color=color, linewidth=1.5)
    axes[2].set_title("隐藏水分/粒度扰动", loc="left")
    axes[2].set_xlabel("隐藏扰动强度")
    add_target_line(axes[2])
    soften_axes(axes[2])
    add_panel_label(axes[2], "c")
    axes[2].legend(fontsize=7)
    fig.text(0.065, 0.055, "压力测试用于界定自校准边界；超过目标线表示需要风险触发或追加标定。", fontsize=6.1, color=PALETTE["muted"])

    svg = FIG_DIR / "fig7_stress_tests.svg"
    png = FIG_DIR / "fig7_stress_tests.png"
    save_figure(fig, FIG_DIR / "fig7_stress_tests", png_dpi=450, tiff_dpi=450, pdf=False)
    plt.close(fig)
    write_run_metadata(
        svg,
        script=SCRIPT_PATH,
        args={},
        inputs=[RESULTS / "sim_stress_tests.csv"],
        outputs=[svg, png],
        notes="Figure 7 summarizes simulation stress-test boundaries for PG-TSC deployment.",
    )
    return svg


def main() -> None:
    args = parse_args()
    out = RESULTS / "sim_stress_tests.csv"
    if args.recompute or not out.exists():
        rows = read_rows(DATA)
        results: list[dict[str, object]] = []
        results.extend(run_empty_gate_stress(rows))
        results.extend(run_ash_missing_stress(rows))
        results.extend(run_hidden_disturbance_stress(rows))
        out = write_table(results)
        df = pd.DataFrame(results)
    else:
        df = pd.read_csv(out)
    fig = plot_results(df)
    print(f"{'wrote' if args.recompute else 'read'}={out}")
    print(f"wrote={fig}")
    for experiment, g in df.groupby("experiment", sort=True):
        best = g.sort_values("cum_err_pct").iloc[0]
        worst = g.sort_values("cum_err_pct").iloc[-1]
        print(
            f"{experiment}: best={best['scenario']}/{best['setting']} {best['cum_err_pct']:.3f}%, "
            f"worst={worst['scenario']}/{worst['setting']} {worst['cum_err_pct']:.3f}%"
        )


if __name__ == "__main__":
    main()
