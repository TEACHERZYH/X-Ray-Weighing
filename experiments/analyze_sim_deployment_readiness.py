from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_versioning import write_run_metadata
from figure_style import PALETTE, add_panel_label, add_target_line, save_figure, setup_publication_style, soften_axes
from run_sim_baselines import add_online_features, feature_matrix, metrics, read_rows, ridge_fit, ridge_predict


ROOT = Path(__file__).resolve().parents[1]
DATA = Path("data/simulated/xray_belt_sim.csv")
RESULTS = Path("results")
FIG_DIR = Path("paper/figures")
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT).as_posix()


@dataclass(frozen=True)
class GateConfig:
    name: str
    min_rx_ratio: float
    max_rx_std: float
    max_speed_std: float
    ema_alpha: float = 0.985


GATE_CONFIGS = [
    GateConfig("auto_conservative", min_rx_ratio=0.940, max_rx_std=90.0, max_speed_std=0.100),
    GateConfig("auto_balanced", min_rx_ratio=0.900, max_rx_std=120.0, max_speed_std=0.120),
    GateConfig("auto_permissive", min_rx_ratio=0.860, max_rx_std=150.0, max_speed_std=0.150),
]


def setup_style() -> None:
    setup_publication_style(font_size=7.0)


def split_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [r for r in rows if str(r["site_id"]) in {"S01", "S02", "S03"}]
    test = [r for r in rows if str(r["site_id"]) == "S04"]
    return train, test


def group_rows(rows: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["site_id"]), str(row["device_id"]), str(row["install_id"]))].append(row)
    out = []
    for group in groups.values():
        group.sort(key=lambda r: float(r["timestamp_s"]))
        out.append(group)
    return out


def fit_temporal_model(rows: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train, _ = split_rows(rows)
    x_train = feature_matrix(train, include_physics=True, include_missing_flag=True, temporal=True)
    y_train = np.array([float(r["true_flow_kgps"]) for r in train])
    return ridge_fit(x_train, y_train, lam=10.0)


def predict_temporal_model(
    rows: list[dict[str, object]],
    model: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    _, test = split_rows(rows)
    x_test = feature_matrix(test, include_physics=True, include_missing_flag=True, temporal=True)
    y_test = np.array([float(r["true_flow_kgps"]) for r in test])
    pred = ridge_predict(x_test, *model)
    return y_test, pred


def gate_stats(rows: list[dict[str, object]]) -> dict[str, float]:
    test = [r for r in rows if str(r["site_id"]) == "S04"]
    truth = np.array([int(float(r["empty_flag"])) for r in test], dtype=int)
    pred = np.array([int(float(r.get("gate_empty", r["empty_flag"]))) for r in test], dtype=int)
    tp = int(((truth == 1) & (pred == 1)).sum())
    fp = int(((truth == 0) & (pred == 1)).sum())
    fn = int(((truth == 1) & (pred == 0)).sum())
    loaded = int((truth == 0).sum())
    return {
        "gate_precision": tp / max(tp + fp, 1),
        "gate_recall": tp / max(tp + fn, 1),
        "loaded_write_pct": fp / max(loaded, 1) * 100.0,
    }


def add_auto_gate_features(rows: list[dict[str, object]], config: GateConfig) -> None:
    for group in group_rows(rows):
        rx_all = np.array([float(r["rx_value"]) for r in group], dtype=float)
        init_n = min(360, len(rx_all))
        i0 = float(np.percentile(rx_all[:init_n], 97))
        hist_a: deque[float] = deque(maxlen=30)
        hist_rx: deque[float] = deque(maxlen=30)
        hist_speed: deque[float] = deque(maxlen=30)
        for row in group:
            rx = max(float(row["rx_value"]), 1.0)
            speed = float(row["belt_speed_mps"])
            hist_rx.append(rx)
            hist_speed.append(speed)
            rx_std = float(np.std(hist_rx))
            speed_std = float(np.std(hist_speed))
            stable_enough = len(hist_rx) >= 20 and rx_std <= config.max_rx_std and speed_std <= config.max_speed_std
            high_transmission = rx >= config.min_rx_ratio * max(i0, 1.0)
            running_belt = speed > 1.5
            gate = int(stable_enough and high_transmission and running_belt)
            if gate:
                i0 = config.ema_alpha * i0 + (1.0 - config.ema_alpha) * rx
            attenuation = max(0.0, np.log(max(i0, rx + 1e-6) / rx))
            hist_a.append(attenuation)
            row["gate_empty"] = float(gate)
            row["i0_online"] = i0
            row["attenuation"] = attenuation
            row["attenuation_mean30"] = float(np.mean(hist_a))
            row["attenuation_std30"] = float(np.std(hist_a))
            row["rx_mean30"] = float(np.mean(hist_rx))
            row["rx_std30"] = rx_std


def run_auto_gate_experiment() -> list[dict[str, object]]:
    rows = read_rows(DATA)
    add_online_features(rows)
    model = fit_temporal_model(rows)
    y_true, pred = predict_temporal_model(rows, model)
    results: list[dict[str, object]] = [
        {
            "experiment": "auto_empty_gate",
            "scenario": "oracle_empty_label",
            "setting": "oracle_i0_update",
            "level": 0.0,
            **gate_stats(rows),
            **metrics(y_true, pred),
        }
    ]

    for config in GATE_CONFIGS:
        gate_rows = read_rows(DATA)
        add_auto_gate_features(gate_rows, config)
        gate_model = fit_temporal_model(gate_rows)
        y_gate, pred_gate = predict_temporal_model(gate_rows, gate_model)
        results.append(
            {
                "experiment": "auto_empty_gate",
                "scenario": config.name,
                "setting": "auto_i0_update",
                "level": config.min_rx_ratio,
                **gate_stats(gate_rows),
                **metrics(y_gate, pred_gate),
            }
        )
    return results


def copy_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row.copy() for row in rows]


def apply_serial_anomalies(
    rows: list[dict[str, object]],
    *,
    dropout_rate: float,
    spike_rate: float,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    for group in group_rows(rows):
        last_rx = float(group[0]["rx_value"])
        for row in group:
            if str(row["site_id"]) != "S04":
                last_rx = float(row["rx_value"])
                continue
            if rng.random() < dropout_rate:
                row["rx_value"] = max(last_rx, 1.0)
                row["serial_dropout"] = 1.0
            else:
                last_rx = float(row["rx_value"])
                row["serial_dropout"] = 0.0
            if rng.random() < spike_rate:
                direction = -1.0 if rng.random() < 0.5 else 1.0
                amplitude = rng.uniform(0.10, 0.22)
                row["rx_value"] = max(float(row["rx_value"]) * (1.0 + direction * amplitude), 1.0)
                row["serial_spike"] = 1.0
            else:
                row["serial_spike"] = 0.0


def repair_serial_rows(rows: list[dict[str, object]], window: int = 15, z: float = 5.0) -> None:
    for group in group_rows(rows):
        if str(group[0]["site_id"]) != "S04":
            continue
        values = np.array([float(row["rx_value"]) for row in group], dtype=float)
        series = pd.Series(values)
        rolling = series.rolling(2 * window + 1, center=True, min_periods=5)
        med = rolling.median().bfill().ffill().to_numpy()
        abs_dev = pd.Series(np.abs(values - med))
        mad = abs_dev.rolling(2 * window + 1, center=True, min_periods=5).median().bfill().ffill().to_numpy()
        scale = np.maximum(1.4826 * mad, 12.0)
        mask = (np.abs(values - med) > z * scale) & (np.abs(values - med) > 60.0)
        repaired = np.where(mask, med, values)
        for row, value in zip(group, repaired):
            row["rx_value"] = float(max(value, 1.0))


def run_serial_robustness_experiment() -> list[dict[str, object]]:
    clean_rows = read_rows(DATA)
    add_online_features(clean_rows)
    model = fit_temporal_model(clean_rows)

    scenarios = [
        ("clean_reference", 0.0, 0.0),
        ("dropout_1pct", 0.01, 0.0),
        ("dropout_3pct", 0.03, 0.0),
        ("dropout_5pct", 0.05, 0.0),
        ("spike_1pct", 0.0, 0.01),
        ("spike_3pct", 0.0, 0.03),
        ("spike_5pct", 0.0, 0.05),
        ("dropout3_spike3", 0.03, 0.03),
    ]
    results: list[dict[str, object]] = []
    base_rows = read_rows(DATA)
    for idx, (scenario, dropout, spike) in enumerate(scenarios):
        for guarded in [False, True]:
            rows = copy_rows(base_rows)
            if dropout or spike:
                apply_serial_anomalies(rows, dropout_rate=dropout, spike_rate=spike, seed=20260626 + idx)
                if guarded:
                    repair_serial_rows(rows)
            add_online_features(rows)
            y_true, pred = predict_temporal_model(rows, model)
            level = dropout + spike
            results.append(
                {
                    "experiment": "serial_robustness",
                    "scenario": scenario,
                    "setting": "guarded_filter" if guarded else "no_filter",
                    "level": level,
                    "gate_precision": np.nan,
                    "gate_recall": np.nan,
                    "loaded_write_pct": np.nan,
                    **metrics(y_true, pred),
                }
            )
    return results


def write_table(rows: list[dict[str, object]]) -> Path:
    out = RESULTS / "sim_deployment_readiness.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    write_run_metadata(
        out,
        script=SCRIPT_PATH,
        args={},
        inputs=[DATA],
        outputs=[out],
        notes="Simulation deployment-readiness tests for automatic empty-belt gating and serial-data anomaly robustness.",
    )
    return out


def plot_results(df: pd.DataFrame) -> Path:
    setup_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.78), constrained_layout=True)

    gate = df[df["experiment"].eq("auto_empty_gate")].copy()
    gate_order = ["oracle_empty_label", "auto_conservative", "auto_balanced", "auto_permissive"]
    gate["scenario"] = pd.Categorical(gate["scenario"], gate_order, ordered=True)
    gate = gate.sort_values("scenario")
    x = np.arange(len(gate))
    axes[0].bar(x - 0.18, gate["gate_precision"] * 100, width=0.36, label="精确率", color=PALETTE["physics"], edgecolor="#333333", linewidth=0.3)
    axes[0].bar(x + 0.18, gate["gate_recall"] * 100, width=0.36, label="召回率", color=PALETTE["temporal"], edgecolor="#333333", linewidth=0.3)
    axes[0].set_title("自动空载门控", loc="left")
    axes[0].set_ylabel("门控指标/%")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["理想", "保守", "平衡", "宽松"], rotation=25)
    axes[0].set_ylim(0, 105)
    soften_axes(axes[0])
    add_panel_label(axes[0], "a")
    axes[0].legend(fontsize=7)

    axes[1].plot(
        x,
        gate["cum_err_pct"],
        marker="o",
        color="#8B6FB3",
        linewidth=1.5,
        label="累计误差",
    )
    axes[1].bar(x, gate["loaded_write_pct"], color=PALETTE["risk"], alpha=0.55, width=0.55, label="有载误写入", edgecolor="#333333", linewidth=0.3)
    axes[1].set_title("误写入与累计误差折中", loc="left")
    axes[1].set_ylabel("累计误差或误写入/%")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(["理想", "保守", "平衡", "宽松"], rotation=25)
    add_target_line(axes[1])
    soften_axes(axes[1])
    add_panel_label(axes[1], "b")
    axes[1].legend(fontsize=7)

    serial = df[df["experiment"].eq("serial_robustness")].copy()
    serial = serial[serial["scenario"].isin(["clean_reference", "dropout_3pct", "spike_3pct", "dropout3_spike3"])]
    serial["scenario"] = pd.Categorical(
        serial["scenario"],
        ["clean_reference", "dropout_3pct", "spike_3pct", "dropout3_spike3"],
        ordered=True,
    )
    for setting, label, color, marker in [
        ("no_filter", "无防护", PALETTE["risk"], "o"),
        ("guarded_filter", "轻量防护", PALETTE["temporal"], "s"),
    ]:
        g = serial[serial["setting"].eq(setting)].sort_values("scenario")
        axes[2].plot(np.arange(len(g)), g["cum_err_pct"], marker=marker, label=label, color=color, linewidth=1.5)
    axes[2].set_title("串口异常压力", loc="left")
    axes[2].set_ylabel("累计误差/%")
    axes[2].set_xticks(np.arange(4))
    axes[2].set_xticklabels(["干净", "丢包3%", "尖峰3%", "混合3%"], rotation=25)
    add_target_line(axes[2])
    soften_axes(axes[2])
    add_panel_label(axes[2], "c")
    axes[2].legend(fontsize=7)
    fig.text(0.01, 0.012, "平衡门控和轻量防护用于说明部署边界，不代表无条件自动更新。", fontsize=6.1, color=PALETTE["muted"])

    svg = FIG_DIR / "fig8_deployment_readiness.svg"
    png = FIG_DIR / "fig8_deployment_readiness.png"
    save_figure(fig, FIG_DIR / "fig8_deployment_readiness", png_dpi=450, tiff_dpi=450, pdf=False)
    plt.close(fig)
    write_run_metadata(
        svg,
        script=SCRIPT_PATH,
        args={},
        inputs=[RESULTS / "sim_deployment_readiness.csv"],
        outputs=[svg, png],
        notes="Figure 8 summarizes deployment-readiness simulation tests for automatic gating and serial anomaly robustness.",
    )
    return svg


def main() -> None:
    rows = []
    rows.extend(run_auto_gate_experiment())
    rows.extend(run_serial_robustness_experiment())
    out = write_table(rows)
    df = pd.DataFrame(rows)
    fig = plot_results(df)
    print(f"wrote={out}")
    print(f"wrote={fig}")
    for experiment, group in df.groupby("experiment", sort=True):
        best = group.sort_values("cum_err_pct").iloc[0]
        worst = group.sort_values("cum_err_pct").iloc[-1]
        print(
            f"{experiment}: best={best['scenario']}/{best['setting']} {best['cum_err_pct']:.3f}%, "
            f"worst={worst['scenario']}/{worst['setting']} {worst['cum_err_pct']:.3f}%"
        )


if __name__ == "__main__":
    main()
