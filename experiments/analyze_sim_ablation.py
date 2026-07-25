from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiment_versioning import write_run_metadata
from run_sim_baselines import add_online_features, feature_matrix, metrics, read_rows, ridge_fit, ridge_predict


def add_fixed_i0_features(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for r in rows:
        key = (str(r["site_id"]), str(r["device_id"]), str(r["install_id"]))
        groups.setdefault(key, []).append(r)

    for group_rows in groups.values():
        group_rows.sort(key=lambda x: float(x["timestamp_s"]))
        initial_empty = [float(r["rx_value"]) for r in group_rows if int(float(r["empty_flag"])) == 1][:120]
        if initial_empty:
            i0 = float(np.median(initial_empty))
        else:
            i0 = float(np.percentile([float(r["rx_value"]) for r in group_rows[:300]], 95))
        hist_a: list[float] = []
        hist_rx: list[float] = []
        for r in group_rows:
            rx = max(float(r["rx_value"]), 1.0)
            attenuation = max(0.0, np.log(max(i0, rx + 1e-6) / rx))
            hist_a.append(attenuation)
            hist_rx.append(rx)
            a_win = hist_a[-30:]
            rx_win = hist_rx[-30:]
            r["i0_online"] = i0
            r["attenuation"] = attenuation
            r["attenuation_mean30"] = float(np.mean(a_win))
            r["attenuation_std30"] = float(np.std(a_win))
            r["rx_mean30"] = float(np.mean(rx_win))
            r["rx_std30"] = float(np.std(rx_win))


def split_rows(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train_rows = [r for r in rows if str(r["site_id"]) in {"S01", "S02", "S03"}]
    test_rows = [r for r in rows if str(r["site_id"]) == "S04"]
    return train_rows, test_rows


def ridge_variant(rows: list[dict[str, object]], *, include_missing: bool = True) -> tuple[np.ndarray, np.ndarray]:
    train_rows, test_rows = split_rows(rows)
    y_train = np.array([float(r["true_flow_kgps"]) for r in train_rows])
    y_test = np.array([float(r["true_flow_kgps"]) for r in test_rows])
    x_train = feature_matrix(train_rows, include_physics=True, include_missing_flag=include_missing, temporal=True)
    x_test = feature_matrix(test_rows, include_physics=True, include_missing_flag=include_missing, temporal=True)
    beta, mean, std = ridge_fit(x_train, y_train, lam=10.0)
    return y_test, ridge_predict(x_test, beta, mean, std)


def selected_indices(y_pred: np.ndarray, n: int, strategy: str) -> np.ndarray:
    n = min(n, len(y_pred))
    if n <= 0:
        return np.array([], dtype=int)
    if strategy == "first":
        return np.arange(n)
    if strategy == "uniform_time":
        return np.linspace(0, len(y_pred) - 1, n).astype(int)
    if strategy == "load_stratified":
        order = np.argsort(y_pred)
        qs = np.linspace(0, len(order) - 1, n).astype(int)
        return np.sort(order[qs])
    raise ValueError(f"unknown strategy: {strategy}")


def calibrate_with_strategy(y_pred: np.ndarray, y_true: np.ndarray, n: int, strategy: str) -> np.ndarray:
    idx = selected_indices(y_pred, n, strategy)
    if len(idx) == 0:
        return y_pred
    x = y_pred[idx]
    y = y_true[idx]
    a, b = np.linalg.lstsq(np.column_stack([x, np.ones_like(x)]), y, rcond=None)[0]
    return a * y_pred + b


def main() -> None:
    input_path = Path("data/simulated/xray_belt_sim.csv")
    out = Path("results/sim_ablation_summary.csv")

    online_rows = read_rows(input_path)
    add_online_features(online_rows)
    y_test, online_pred = ridge_variant(online_rows, include_missing=True)

    fixed_rows = read_rows(input_path)
    add_fixed_i0_features(fixed_rows)
    _, fixed_pred = ridge_variant(fixed_rows, include_missing=True)
    _, no_missing_pred = ridge_variant(online_rows, include_missing=False)

    rows: list[dict[str, object]] = []
    rows.append({"group": "i0_update", "setting": "online_i0", "n_samples": 0, **metrics(y_test, online_pred)})
    rows.append({"group": "i0_update", "setting": "fixed_initial_i0", "n_samples": 0, **metrics(y_test, fixed_pred)})
    rows.append({"group": "ash_missing", "setting": "with_missing_flag", "n_samples": 0, **metrics(y_test, online_pred)})
    rows.append({"group": "ash_missing", "setting": "without_missing_flag", "n_samples": 0, **metrics(y_test, no_missing_pred)})

    for strategy in ["first", "uniform_time", "load_stratified"]:
        for n in [5, 20, 60, 180]:
            pred = calibrate_with_strategy(online_pred, y_test, n, strategy)
            rows.append({"group": "calibration_sampling", "setting": strategy, "n_samples": n, **metrics(y_test, pred)})

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    meta = write_run_metadata(
        out,
        script=__file__,
        args={},
        inputs=[input_path],
        outputs=[out],
        notes="Supplementary simulation ablations for I0 update, missing-ash flag, and calibration sample selection.",
    )
    print(f"wrote={out}")
    print(f"wrote={meta}")
    for row in rows:
        print(f"{row['group']}/{row['setting']}/n={row['n_samples']}: CumErr={row['cum_err_pct']:.3f}%")


if __name__ == "__main__":
    main()
