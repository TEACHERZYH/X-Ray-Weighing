from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent))

from experiment_versioning import write_run_metadata  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]

PROTOCOL_COLUMNS = [
    "timestamp_s",
    "site_id",
    "device_id",
    "install_id",
    "rx_value",
    "tube_voltage_kv",
    "tube_current_ma",
    "source_temp_c",
    "env_temp_c",
    "env_humidity_pct",
    "belt_speed_mps",
    "empty_flag",
    "coal_type",
    "ash_pct",
    "ash_missing",
    "true_loading_kgpm",
    "true_flow_kgps",
    "true_cum_mass_kg",
]

SENSOR_NUMERIC = [
    "rx_value",
    "tube_voltage_kv",
    "tube_current_ma",
    "source_temp_c",
    "env_temp_c",
    "env_humidity_pct",
    "belt_speed_mps",
]

LABEL_NUMERIC = ["true_loading_kgpm", "true_flow_kgps", "true_cum_mass_kg"]
FLAG_COLUMNS = ["empty_flag", "ash_missing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean real X-ray belt weighing CSV.")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "real_xray_belt.csv")
    parser.add_argument("--clean-output", type=Path, default=ROOT / "data" / "real" / "real_xray_belt_clean.csv")
    parser.add_argument("--qc-output", type=Path, default=ROOT / "data" / "real" / "real_xray_belt_clean_qc.csv")
    parser.add_argument("--report", type=Path, default=ROOT / "results" / "real_data_cleaning_report.csv")
    parser.add_argument("--profile", type=Path, default=ROOT / "results" / "real_data_profile.csv")
    parser.add_argument("--train-sites", default="S01,S02,S03")
    return parser.parse_args()


def metric(rows: list[dict[str, Any]], name: str, value: Any, note: str = "") -> None:
    rows.append({"metric": name, "value": value, "note": note})


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    missing_before: dict[str, int] = {}
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        missing_before[col] = int(df[col].isna().sum())
    return missing_before


def fill_sensor_columns(df: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, dict[str, int]]:
    out = df.copy()
    filled: dict[str, int] = {}
    train_medians = {col: float(out.loc[train_mask, col].median()) for col in SENSOR_NUMERIC}
    for col in SENSOR_NUMERIC:
        before = out[col].isna()
        out[col] = out.groupby(["site_id", "device_id", "install_id"], group_keys=False)[col].transform(
            lambda s: s.ffill()
        )
        out[col] = out[col].fillna(train_medians[col])
        filled[col] = int(before.sum())
    return out, filled


def fill_ash(df: pd.DataFrame, train_mask: pd.Series) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    ash_was_missing = out["ash_pct"].isna()
    if "ash_pct_missing_injected" in out:
        ash_was_missing = ash_was_missing | pd.to_numeric(out["ash_pct_missing_injected"], errors="coerce").fillna(0).eq(1)
    out["ash_missing"] = (
        pd.to_numeric(out["ash_missing"], errors="coerce").fillna(0).astype(int).eq(1) | ash_was_missing
    ).astype(int)
    train = out.loc[train_mask, ["coal_type", "ash_pct"]]
    coal_median = train.groupby("coal_type")["ash_pct"].median().to_dict()
    global_median = float(train["ash_pct"].median())
    out["ash_pct"] = out["ash_pct"].fillna(out["coal_type"].map(coal_median)).fillna(global_median)
    return out, int(ash_was_missing.sum())


def monotonic_violations(df: pd.DataFrame, column: str) -> int:
    violations = 0
    for _, g in df.groupby(["site_id", "device_id", "install_id"], sort=False):
        values = g[column].to_numpy(float)
        violations += int(np.sum(np.diff(values) < -1e-9))
    return violations


def write_key_value_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    raw = pd.read_csv(args.input)
    missing_required = [col for col in PROTOCOL_COLUMNS if col not in raw.columns]
    if missing_required:
        raise ValueError("missing required columns: " + ", ".join(missing_required))

    df = raw.copy()
    numeric_missing_before = coerce_numeric(df, ["timestamp_s", *SENSOR_NUMERIC, "ash_pct", *LABEL_NUMERIC])
    for col in FLAG_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    for col in ["bad_point", "continuous_anomaly", "missing_any_injected"]:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    df["site_id"] = df["site_id"].astype(str)
    df["device_id"] = df["device_id"].astype(str)
    df["install_id"] = df["install_id"].astype(str)
    df["coal_type"] = df["coal_type"].astype(str).replace({"": "unknown", "nan": "unknown"})
    df = df.sort_values(["site_id", "device_id", "install_id", "timestamp_s"], kind="mergesort").reset_index(drop=True)
    train_sites = {value.strip() for value in args.train_sites.split(",") if value.strip()}
    train_mask = df["site_id"].isin(train_sites)
    if not train_mask.any():
        raise ValueError(f"none of the requested training sites are present: {sorted(train_sites)}")

    df, sensor_filled = fill_sensor_columns(df, train_mask)
    df, ash_filled = fill_ash(df, train_mask)

    invalid_label = df[LABEL_NUMERIC].isna().any(axis=1) | df["true_flow_kgps"].lt(0) | df["true_cum_mass_kg"].isna()
    train_excluded = df["bad_point"].eq(1) | df["continuous_anomaly"].eq(1) | invalid_label
    df["train_excluded"] = train_excluded.astype(int)
    df["eval_excluded"] = invalid_label.astype(int)

    qc_columns = [
        "bad_point",
        "bad_point_type",
        "continuous_anomaly",
        "continuous_anomaly_type",
        "continuous_segment_id",
        "missing_any_injected",
        "missing_event_type",
        "missing_event_id",
        "train_excluded",
        "eval_excluded",
    ]
    existing_qc = [col for col in qc_columns if col in df.columns]
    clean = df[PROTOCOL_COLUMNS].copy()
    qc = df[[*PROTOCOL_COLUMNS, *existing_qc]].copy()

    args.clean_output.parent.mkdir(parents=True, exist_ok=True)
    args.qc_output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(args.clean_output, index=False)
    qc.to_csv(args.qc_output, index=False)

    report: list[dict[str, Any]] = []
    metric(report, "input_rows", len(raw))
    metric(report, "train_sites_for_imputation", ",".join(sorted(train_sites)))
    metric(report, "output_rows", len(clean))
    metric(report, "sites", df["site_id"].nunique())
    metric(report, "devices", df["device_id"].nunique())
    metric(report, "installs", df["install_id"].nunique())
    metric(report, "coal_types", df["coal_type"].nunique(), ",".join(sorted(df["coal_type"].unique())))
    metric(report, "empty_rows", int(df["empty_flag"].eq(1).sum()))
    metric(report, "loaded_rows", int(df["true_flow_kgps"].gt(0).sum()))
    metric(report, "bad_point_rows", int(df["bad_point"].sum()))
    metric(report, "continuous_anomaly_rows", int(df["continuous_anomaly"].sum()))
    metric(report, "train_excluded_rows", int(df["train_excluded"].sum()))
    metric(report, "eval_excluded_rows", int(df["eval_excluded"].sum()))
    metric(report, "ash_filled_rows", ash_filled)
    metric(report, "time_monotonic_violations", monotonic_violations(df, "timestamp_s"))
    metric(report, "cum_mass_monotonic_violations", monotonic_violations(df, "true_cum_mass_kg"))
    for col, count in numeric_missing_before.items():
        metric(report, f"missing_before_{col}", count)
    for col, count in sensor_filled.items():
        metric(report, f"sensor_filled_{col}", count)
    write_key_value_csv(args.report, report)

    profile_rows = []
    for key_cols, name in [
        (["site_id"], "site"),
        (["site_id", "device_id"], "device"),
        (["site_id", "device_id", "install_id"], "install"),
        (["coal_type"], "coal_type"),
    ]:
        grouped = df.groupby(key_cols, sort=True)
        for key, g in grouped:
            if not isinstance(key, tuple):
                key = (key,)
            profile_rows.append(
                {
                    "level": name,
                    "key": "|".join(map(str, key)),
                    "n": len(g),
                    "empty_rows": int(g["empty_flag"].eq(1).sum()),
                    "train_excluded_rows": int(g["train_excluded"].sum()),
                    "flow_mean_kgps": float(g["true_flow_kgps"].mean()),
                    "flow_sum_kg": float(g["true_flow_kgps"].sum()),
                    "env_temp_min": float(g["env_temp_c"].min()),
                    "env_temp_max": float(g["env_temp_c"].max()),
                    "humidity_min": float(g["env_humidity_pct"].min()),
                    "humidity_max": float(g["env_humidity_pct"].max()),
                }
            )
    args.profile.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(profile_rows).to_csv(args.profile, index=False)

    for output in [args.clean_output, args.qc_output, args.report, args.profile]:
        meta = write_run_metadata(
            output,
            script=__file__,
            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            inputs=[args.input],
            outputs=[output],
            notes="Real X-ray belt data cleaning with source-domain-only causal imputation.",
        )
        print(f"wrote={output}")
        print(f"wrote={meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
