from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from experiment_versioning import write_run_metadata
from simulate_xray_belt import simulate_install


NUMERIC_FIELDS = [
    "rx_value",
    "tube_voltage_kv",
    "tube_current_ma",
    "source_temp_c",
    "env_temp_c",
    "env_humidity_pct",
    "ash_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a shifted target-domain simulation dataset for transfer validation.")
    parser.add_argument("--output", default="data/simulated/xray_belt_transfer_target.csv")
    parser.add_argument("--summary", default="results/sim_transfer_target_summary.csv")
    parser.add_argument("--sites", type=int, default=8)
    parser.add_argument("--devices-per-site", type=int, default=2)
    parser.add_argument("--installs-per-device", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--severity", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260627)
    return parser.parse_args()


def apply_target_shift(rows: list[dict[str, Any]], severity: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = len(rows)
    phase = float(rng.uniform(0, 2 * math.pi))
    rx_gain = float(rng.normal(1.0, 0.018 * severity))
    rx_bias = float(rng.normal(0.0, 28.0 * severity))
    rx_quad = float(rng.normal(-0.8e-6 * severity, 0.25e-6 * severity))
    voltage_offset = float(rng.normal(0.0, 0.45 * severity))
    current_gain = float(rng.normal(1.0, 0.007 * severity))
    source_temp_offset = float(rng.normal(0.0, 1.8 * severity))
    env_temp_offset = float(rng.normal(0.0, 1.2 * severity))
    humidity_offset = float(rng.normal(0.0, 4.0 * severity))
    ash_bias = float(rng.normal(0.0, 1.2 * severity))

    for row in rows:
        t = float(row["timestamp_s"])
        progress = t / max(n - 1, 1)
        slow_drift = 1.0 + severity * (0.012 * math.sin(2 * math.pi * progress + phase) + 0.006 * (progress - 0.5))
        rx = float(row["rx_value"])
        centered = (rx - 8500.0) / 1000.0
        rx_shifted = rx * rx_gain * slow_drift + rx_bias + rx_quad * centered * centered * 1000.0
        rx_shifted += float(rng.normal(0.0, 9.0 * severity))
        row["rx_value"] = round(max(rx_shifted, 1.0), 6)
        row["tube_voltage_kv"] = round(float(row["tube_voltage_kv"]) + voltage_offset + 0.08 * math.sin(2 * math.pi * progress + phase), 6)
        row["tube_current_ma"] = round(max(float(row["tube_current_ma"]) * current_gain, 0.1), 6)
        row["source_temp_c"] = round(float(row["source_temp_c"]) + source_temp_offset, 6)
        row["env_temp_c"] = round(float(row["env_temp_c"]) + env_temp_offset, 6)
        row["env_humidity_pct"] = round(float(np.clip(float(row["env_humidity_pct"]) + humidity_offset, 5.0, 100.0)), 6)
        row["ash_pct"] = round(max(float(row["ash_pct"]) + ash_bias, 0.0), 6)


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    summary = Path(args.summary)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    total_mass_kg = 0.0
    empty_count = 0
    ash_missing_count = 0
    coal_counts: Counter[str] = Counter()
    run_seed = args.seed
    fieldnames: list[str] | None = None

    with out.open("w", newline="", encoding="utf-8") as f:
        writer: csv.DictWriter[str] | None = None
        for site_idx in range(1, args.sites + 1):
            for device_idx in range(1, args.devices_per_site + 1):
                for install_idx in range(1, args.installs_per_device + 1):
                    rows = simulate_install(100 + site_idx, device_idx, install_idx, args.seconds, run_seed)
                    for row in rows:
                        row["site_id"] = f"T{site_idx:02d}"
                    apply_target_shift(rows, args.severity, run_seed + 100_000)
                    run_seed += 1
                    if writer is None:
                        fieldnames = list(rows[0].keys())
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                    writer.writerows(rows)
                    rows_written += len(rows)
                    total_mass_kg += sum(float(r["true_flow_kgps"]) for r in rows)
                    empty_count += sum(int(r["empty_flag"]) for r in rows)
                    ash_missing_count += sum(int(r["ash_missing"]) for r in rows)
                    coal_counts.update(str(r["coal_type"]) for r in rows)

    if rows_written == 0 or fieldnames is None:
        raise RuntimeError("target-domain simulation generated no rows")

    summary_row: dict[str, object] = {
        "dataset": f"{out.parent.name}/{out.name}",
        "rows": rows_written,
        "sites": args.sites,
        "devices_per_site": args.devices_per_site,
        "installs_per_device": args.installs_per_device,
        "seconds_per_install": args.seconds,
        "install_count": args.sites * args.devices_per_site * args.installs_per_device,
        "severity": args.severity,
        "total_mass_kg": round(total_mass_kg, 6),
        "empty_ratio": round(empty_count / rows_written, 6),
        "ash_missing_ratio": round(ash_missing_count / rows_written, 6),
        "seed": args.seed,
        "coal_anthracite": coal_counts.get("anthracite", 0),
        "coal_bituminous_A": coal_counts.get("bituminous_A", 0),
        "coal_bituminous_B": coal_counts.get("bituminous_B", 0),
        "coal_lignite": coal_counts.get("lignite", 0),
    }
    with summary.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_row.keys()))
        writer.writeheader()
        writer.writerow(summary_row)

    meta = write_run_metadata(
        summary,
        script=__file__,
        args=vars(args),
        inputs=[],
        outputs=[out, summary],
        notes="Shifted target-domain simulation dataset for pretraining transfer validation.",
    )
    print(f"wrote={out}")
    print(f"wrote={summary}")
    print(f"wrote={meta}")
    print(f"rows={rows_written}")
    print(f"install_count={summary_row['install_count']}")
    print(f"empty_ratio={summary_row['empty_ratio']}")
    print(f"ash_missing_ratio={summary_row['ash_missing_ratio']}")


if __name__ == "__main__":
    main()
