from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean

from experiment_versioning import write_run_metadata


NUMERIC_COLUMNS = [
    "timestamp_s",
    "rx_value",
    "tube_voltage_kv",
    "tube_current_ma",
    "source_temp_c",
    "env_temp_c",
    "env_humidity_pct",
    "belt_speed_mps",
    "empty_flag",
    "ash_pct",
    "ash_missing",
    "true_loading_kgpm",
    "true_flow_kgps",
    "true_cum_mass_kg",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit simulated X-ray belt weighing data plausibility.")
    parser.add_argument("--input", default="data/simulated/xray_belt_sim.csv")
    parser.add_argument("--output", default="results/sim_simulator_plausibility.csv")
    return parser.parse_args()


def quantile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = fmean(xs)
    my = fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def status_between(value: float, lower: float | None, upper: float | None) -> str:
    if math.isnan(value):
        return "BLOCK"
    if lower is not None and value < lower:
        return "BLOCK"
    if upper is not None and value > upper:
        return "BLOCK"
    return "PASS"


def status_at_least(value: float, lower: float) -> str:
    if math.isnan(value):
        return "BLOCK"
    return "PASS" if value >= lower else "BLOCK"


def status_at_most(value: float, upper: float) -> str:
    if math.isnan(value):
        return "BLOCK"
    return "PASS" if value <= upper else "BLOCK"


def add_check(
    rows: list[dict[str, object]],
    metric: str,
    value: float | int | str,
    unit: str,
    status: str,
    lower_bound: float | str = "",
    upper_bound: float | str = "",
    note: str = "",
) -> None:
    rows.append(
        {
            "metric": metric,
            "value": value,
            "unit": unit,
            "status": status,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "note": note,
        }
    )


def read_rows(path: Path) -> tuple[list[dict[str, str]], dict[str, list[float]]]:
    records: list[dict[str, str]] = []
    numeric: dict[str, list[float]] = {name: [] for name in NUMERIC_COLUMNS}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = [name for name in NUMERIC_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"missing numeric columns: {missing}")
        for row in reader:
            records.append(row)
            for name in NUMERIC_COLUMNS:
                numeric[name].append(float(row[name]))
    return records, numeric


def add_numeric_range_checks(rows: list[dict[str, object]], numeric: dict[str, list[float]]) -> None:
    range_specs = {
        "rx_value": (1.0, None, "detector counts"),
        "tube_voltage_kv": (70.0, 90.0, "kV"),
        "tube_current_ma": (0.5, 5.0, "mA"),
        "source_temp_c": (-10.0, 90.0, "degC"),
        "env_temp_c": (-30.0, 60.0, "degC"),
        "env_humidity_pct": (0.0, 100.0, "%"),
        "belt_speed_mps": (1.0, 5.0, "m/s"),
        "ash_pct": (0.0, 50.0, "%"),
        "true_loading_kgpm": (0.0, 250.0, "kg/m"),
        "true_flow_kgps": (0.0, 600.0, "kg/s"),
    }
    for name, (lower, upper, unit) in range_specs.items():
        values = numeric[name]
        v_min = min(values)
        v_max = max(values)
        status = "PASS"
        if lower is not None and v_min < lower:
            status = "BLOCK"
        if upper is not None and v_max > upper:
            status = "BLOCK"
        add_check(
            rows,
            f"{name}_range",
            f"{v_min:.3f} to {v_max:.3f}",
            unit,
            status,
            lower if lower is not None else "",
            upper if upper is not None else "",
            f"mean={fmean(values):.3f}; p05={quantile(values, 0.05):.3f}; p95={quantile(values, 0.95):.3f}",
        )


def add_sequence_checks(rows: list[dict[str, object]], records: list[dict[str, str]]) -> None:
    by_install: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for record in records:
        key = (record["site_id"], record["device_id"], record["install_id"])
        by_install[key].append(record)

    monotonic_failures = 0
    sum_mismatch_failures = 0
    timestamp_failures = 0
    final_masses: list[float] = []
    for key, items in by_install.items():
        timestamps = [int(float(item["timestamp_s"])) for item in items]
        cums = [float(item["true_cum_mass_kg"]) for item in items]
        flows = [float(item["true_flow_kgps"]) for item in items]
        if timestamps != list(range(len(timestamps))):
            timestamp_failures += 1
        if any(b + 1e-6 < a for a, b in zip(cums, cums[1:])):
            monotonic_failures += 1
        summed = sum(flows)
        final = cums[-1] if cums else float("nan")
        final_masses.append(final)
        if math.isnan(final) or abs(final - summed) > max(1.0, 1e-6 * max(1.0, summed)):
            sum_mismatch_failures += 1

    add_check(rows, "timestamp_integrity_failures", timestamp_failures, "installations", "PASS" if timestamp_failures == 0 else "BLOCK", 0, 0, "timestamp_s should restart at zero and increase by one within each installation")
    add_check(rows, "cumulative_mass_monotonic_failures", monotonic_failures, "installations", "PASS" if monotonic_failures == 0 else "BLOCK", 0, 0, "true_cum_mass_kg should be monotonic within each installation")
    add_check(rows, "cumulative_mass_sum_mismatch_failures", sum_mismatch_failures, "installations", "PASS" if sum_mismatch_failures == 0 else "BLOCK", 0, 0, "final cumulative mass should match summed flow samples")
    add_check(rows, "install_final_mass_range", f"{min(final_masses):.1f} to {max(final_masses):.1f}", "kg", "PASS", "", "", "range of total simulated mass per installation")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records, numeric = read_rows(input_path)
    if not records:
        raise ValueError("input dataset is empty")

    rows: list[dict[str, object]] = []
    site_ids = {row["site_id"] for row in records}
    device_ids = {(row["site_id"], row["device_id"]) for row in records}
    install_ids = {(row["site_id"], row["device_id"], row["install_id"]) for row in records}
    coal_counts = Counter(row["coal_type"] for row in records)
    empty_ratio = fmean(numeric["empty_flag"])
    ash_missing_ratio = fmean(numeric["ash_missing"])

    add_check(rows, "row_count", len(records), "samples", status_at_least(len(records), 10_000), 10000, "", "small benchmark dataset should still cover many operating states")
    add_check(rows, "site_count", len(site_ids), "sites", status_at_least(len(site_ids), 4), 4, "", "cross-site split requires multiple simulated sites")
    add_check(rows, "device_count", len(device_ids), "site-device pairs", status_at_least(len(device_ids), 8), 8, "", "device variation is required for installation robustness tests")
    add_check(rows, "install_count", len(install_ids), "installations", status_at_least(len(install_ids), 16), 16, "", "installation variation is required for self-calibration tests")
    add_check(rows, "coal_type_count", len(coal_counts), "coal types", status_at_least(len(coal_counts), 4), 4, "", "; ".join(f"{k}={v}" for k, v in sorted(coal_counts.items())))
    add_check(rows, "empty_ratio", f"{empty_ratio:.6f}", "ratio", status_between(empty_ratio, 0.05, 0.35), 0.05, 0.35, "empty samples are needed for online I0 estimation but should not dominate")
    add_check(rows, "ash_missing_ratio", f"{ash_missing_ratio:.6f}", "ratio", status_between(ash_missing_ratio, 0.20, 0.60), 0.20, 0.60, "partial ash availability matches the current data-access assumption")

    add_numeric_range_checks(rows, numeric)

    env_span = max(numeric["env_temp_c"]) - min(numeric["env_temp_c"])
    humidity_span = max(numeric["env_humidity_pct"]) - min(numeric["env_humidity_pct"])
    rx_span = max(numeric["rx_value"]) - min(numeric["rx_value"])
    source_env_delta = fmean(numeric["source_temp_c"]) - fmean(numeric["env_temp_c"])
    add_check(rows, "env_temp_span", f"{env_span:.3f}", "degC", status_at_least(env_span, 10.0), 10.0, "", "environmental drift should be visible in the simulator")
    add_check(rows, "humidity_span", f"{humidity_span:.3f}", "%", status_at_least(humidity_span, 30.0), 30.0, "", "humidity drift should be visible in the simulator")
    add_check(rows, "rx_value_span", f"{rx_span:.3f}", "detector counts", status_at_least(rx_span, 1000.0), 1000.0, "", "detector signal should vary with load and source/detector drift")
    add_check(rows, "source_minus_env_mean", f"{source_env_delta:.3f}", "degC", status_between(source_env_delta, 10.0, 25.0), 10.0, 25.0, "source tube should be warmer than ambient in the synthetic device model")

    rx_loading_corr = corr(numeric["rx_value"], numeric["true_loading_kgpm"])
    loading_flow_corr = corr(numeric["true_loading_kgpm"], numeric["true_flow_kgps"])
    speed_flow_corr = corr(numeric["belt_speed_mps"], numeric["true_flow_kgps"])
    add_check(rows, "corr_rx_loading", f"{rx_loading_corr:.6f}", "Pearson r", "PASS" if rx_loading_corr <= -0.70 else "BLOCK", "", -0.70, "X-ray attenuation should make detector counts decrease as loading increases")
    add_check(rows, "corr_loading_flow", f"{loading_flow_corr:.6f}", "Pearson r", "PASS" if loading_flow_corr >= 0.95 else "BLOCK", 0.95, "", "mass flow is generated from loading and belt speed")
    add_check(rows, "corr_speed_flow", f"{speed_flow_corr:.6f}", "Pearson r", "PASS" if speed_flow_corr >= 0.05 else "WARN", 0.05, "", "belt speed has a secondary positive contribution because loading dominates flow variance")

    add_sequence_checks(rows, records)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["metric", "value", "unit", "status", "lower_bound", "upper_bound", "note"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_run_metadata(
        output_path,
        script="experiments/analyze_simulator_plausibility.py",
        args=vars(args),
        inputs=[input_path],
        outputs=[output_path],
        notes="Plausibility audit for the synthetic X-ray belt weighing simulator and manuscript data-mouth alignment.",
    )

    counts = Counter(str(row["status"]) for row in rows)
    print(f"wrote={output_path}")
    print(f"checks={len(rows)} PASS={counts.get('PASS', 0)} WARN={counts.get('WARN', 0)} BLOCK={counts.get('BLOCK', 0)}")
    for row in rows:
        if row["status"] != "PASS":
            print(f"{row['status']}: {row['metric']}={row['value']} {row['note']}")
    return 0 if counts.get("BLOCK", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
