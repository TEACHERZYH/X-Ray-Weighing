from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_versioning import write_run_metadata


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT).as_posix()

METHODS = {
    "DataOnly-HGB": "DataOnly-HGB",
    "Physics-Ridge": "Physics-Ridge",
    "Physics-Temporal-Ridge": "Physics-Temporal-Ridge",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze cross-install bootstrap uncertainty from point predictions.")
    parser.add_argument("--predictions", type=Path, default=RESULTS / "sim_predictions.csv")
    parser.add_argument("--out", type=Path, default=RESULTS / "sim_cross_install_uncertainty.csv")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260628)
    return parser.parse_args()


def metric_from_sums(abs_err: float, sq_err: float, n: int, true_sum: float, pred_sum: float, metric: str) -> float:
    if metric == "mae":
        return abs_err / max(n, 1)
    if metric == "rmse":
        return math.sqrt(sq_err / max(n, 1))
    if metric == "cum_err_pct":
        return abs(pred_sum - true_sum) / max(true_sum, 1e-9) * 100.0
    raise ValueError(f"unknown metric: {metric}")


def install_summaries(pred: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["site_id", "device_id", "install_id"]
    for keys, group in pred.groupby(group_cols, sort=True):
        site_id, device_id, install_id = keys
        y_true = group["true_flow_kgps"].to_numpy(dtype=float)
        for method, col in METHODS.items():
            y_pred = group[col].clip(lower=0).to_numpy(dtype=float)
            err = y_pred - y_true
            abs_err = float(np.abs(err).sum())
            sq_err = float(np.square(err).sum())
            true_sum = float(y_true.sum())
            pred_sum = float(y_pred.sum())
            rows.append(
                {
                    "site_id": site_id,
                    "device_id": device_id,
                    "install_id": install_id,
                    "method": method,
                    "n": len(group),
                    "abs_err_sum": abs_err,
                    "sq_err_sum": sq_err,
                    "true_sum": true_sum,
                    "pred_sum": pred_sum,
                    "mae": metric_from_sums(abs_err, sq_err, len(group), true_sum, pred_sum, "mae"),
                    "rmse": metric_from_sums(abs_err, sq_err, len(group), true_sum, pred_sum, "rmse"),
                    "cum_err_pct": metric_from_sums(abs_err, sq_err, len(group), true_sum, pred_sum, "cum_err_pct"),
                }
            )
    return pd.DataFrame(rows)


def bootstrap_metric(summary: pd.DataFrame, metric: str, n_boot: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    records = summary[["abs_err_sum", "sq_err_sum", "n", "true_sum", "pred_sum"]].to_numpy(dtype=float)
    n_units = len(records)
    values = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_idx = rng.integers(0, n_units, size=n_units)
        sample = records[sample_idx]
        abs_err = float(sample[:, 0].sum())
        sq_err = float(sample[:, 1].sum())
        n = int(sample[:, 2].sum())
        true_sum = float(sample[:, 3].sum())
        pred_sum = float(sample[:, 4].sum())
        values[i] = metric_from_sums(abs_err, sq_err, n, true_sum, pred_sum, metric)
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def summarize(pred: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    install = install_summaries(pred)
    rows: list[dict[str, object]] = []
    for method, group in install.groupby("method", sort=False):
        for metric, unit in [("mae", "kg/s"), ("rmse", "kg/s"), ("cum_err_pct", "%")]:
            point = metric_from_sums(
                float(group["abs_err_sum"].sum()),
                float(group["sq_err_sum"].sum()),
                int(group["n"].sum()),
                float(group["true_sum"].sum()),
                float(group["pred_sum"].sum()),
                metric,
            )
            ci_low, ci_high = bootstrap_metric(group, metric, n_boot=n_boot, seed=seed)
            values = group[metric].astype(float)
            rows.append(
                {
                    "method": method,
                    "metric": metric,
                    "unit": unit,
                    "point_estimate": point,
                    "install_mean": float(values.mean()),
                    "install_sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "install_min": float(values.min()),
                    "install_max": float(values.max()),
                    "bootstrap_ci95_low": ci_low,
                    "bootstrap_ci95_high": ci_high,
                    "n_install": int(group.shape[0]),
                    "n_boot": n_boot,
                    "seed": seed,
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    pred = pd.read_csv(args.predictions)
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    result = summarize(pred, n_boot=args.n_boot, seed=args.seed)
    result.to_csv(out, index=False)
    meta = write_run_metadata(
        out,
        script=SCRIPT_PATH,
        args={
            "predictions": args.predictions.relative_to(ROOT).as_posix() if args.predictions.is_absolute() else args.predictions.as_posix(),
            "out": out.relative_to(ROOT).as_posix() if out.is_absolute() else out.as_posix(),
            "n_boot": args.n_boot,
            "seed": args.seed,
        },
        inputs=[args.predictions],
        outputs=[out],
        notes="Cross-install bootstrap uncertainty analysis from existing point predictions; no model retraining.",
    )
    print(f"wrote={out}")
    print(f"wrote={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
