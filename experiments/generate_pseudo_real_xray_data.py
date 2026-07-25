from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ is None:
    sys.path.append(str(Path(__file__).resolve().parent))

from experiment_versioning import write_run_metadata  # noqa: E402
from simulate_xray_belt import simulate_install  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pseudo-real X-ray belt weighing field data.")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "real_like" / "pseudo_real_xray_belt.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "results" / "pseudo_real_data_profile.csv")
    parser.add_argument("--sites", type=int, default=5)
    parser.add_argument("--devices-per-site", type=int, default=2)
    parser.add_argument("--installs-per-device", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260629)
    parser.add_argument("--inter-install-gap-s", type=int, default=300)
    return parser.parse_args()


def quantize(row: dict[str, object], rng: np.random.Generator) -> dict[str, object]:
    out = dict(row)
    rx = float(out["rx_value"])
    if rng.random() < 0.002:
        rx *= rng.uniform(0.96, 1.04)
    out["rx_value"] = int(round(max(rx, 1.0)))
    out["tube_voltage_kv"] = round(float(out["tube_voltage_kv"]), 2)
    out["tube_current_ma"] = round(float(out["tube_current_ma"]), 3)
    out["source_temp_c"] = round(float(out["source_temp_c"]), 1)
    out["env_temp_c"] = round(float(out["env_temp_c"]), 1)
    out["env_humidity_pct"] = round(float(out["env_humidity_pct"]), 1)
    out["belt_speed_mps"] = round(float(out["belt_speed_mps"]), 3)
    out["ash_pct"] = round(float(out["ash_pct"]), 2)
    out["true_loading_kgpm"] = round(float(out["true_loading_kgpm"]), 3)
    out["true_flow_kgps"] = round(float(out["true_flow_kgps"]), 3)
    out["true_cum_mass_kg"] = round(float(out["true_cum_mass_kg"]), 3)
    return out


def generate_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    global_time = 0
    run_seed = args.seed
    rng = np.random.default_rng(args.seed)
    for site_idx in range(1, args.sites + 1):
        for device_idx in range(1, args.devices_per_site + 1):
            for install_idx in range(1, args.installs_per_device + 1):
                install_rows = simulate_install(site_idx, device_idx, install_idx, args.seconds, run_seed)
                site_id = f"S{site_idx:02d}"
                device_id = f"XRAY-{site_idx:02d}{device_idx:02d}"
                install_id = f"{site_id}_{device_id}_SETUP{install_idx:02d}"
                for row in install_rows:
                    row = quantize(row, rng)
                    row["timestamp_s"] = int(global_time + int(row["timestamp_s"]))
                    row["site_id"] = site_id
                    row["device_id"] = device_id
                    row["install_id"] = install_id
                    rows.append(row)
                global_time += args.seconds + args.inter_install_gap_s
                run_seed += 1
    return rows


def corr(rows: list[dict[str, object]], a: str, b: str) -> float:
    xs = np.asarray([float(row[a]) for row in rows], dtype=float)
    ys = np.asarray([float(row[b]) for row in rows], dtype=float)
    if xs.size < 2 or np.std(xs) == 0 or np.std(ys) == 0:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


def span(rows: list[dict[str, object]], key: str) -> tuple[float, float]:
    values = [float(row[key]) for row in rows]
    return min(values), max(values)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sites = {str(row["site_id"]) for row in rows}
    devices = {str(row["device_id"]) for row in rows}
    installs = {str(row["install_id"]) for row in rows}
    coal_counts = Counter(str(row["coal_type"]) for row in rows)
    empty_count = sum(1 for row in rows if int(row["empty_flag"]) == 1)
    ash_missing = sum(1 for row in rows if int(row["ash_missing"]) == 1)
    loaded = sum(1 for row in rows if float(row["true_flow_kgps"]) > 0)
    total_mass = sum(float(row["true_flow_kgps"]) for row in rows)
    rx_loading_corr = corr(rows, "rx_value", "true_loading_kgpm")
    loading_flow_corr = corr(rows, "true_loading_kgpm", "true_flow_kgps")
    env_lo, env_hi = span(rows, "env_temp_c")
    hum_lo, hum_hi = span(rows, "env_humidity_pct")
    rx_lo, rx_hi = span(rows, "rx_value")
    summary_rows = [
        ("rows", len(rows), "samples", "total pseudo-real canonical rows"),
        ("site_count", len(sites), "sites", ",".join(sorted(sites))),
        ("device_count", len(devices), "devices", "unique anonymized X-ray devices"),
        ("install_count", len(installs), "installations", "globally unique install_id values"),
        ("seconds_per_install", args.seconds, "s", "nominal duration before inter-install gap"),
        ("empty_ratio", empty_count / len(rows), "ratio", f"{empty_count} empty samples"),
        ("loaded_rows", loaded, "samples", "true_flow_kgps > 0"),
        ("ash_missing_ratio", ash_missing / len(rows), "ratio", f"{ash_missing} ash_missing samples"),
        ("coal_type_count", len(coal_counts), "classes", "; ".join(f"{k}={v}" for k, v in sorted(coal_counts.items()))),
        ("total_mass", total_mass, "kg", "sum of per-second true_flow_kgps samples"),
        ("env_temp_span", env_hi - env_lo, "degC", f"{env_lo:.1f} to {env_hi:.1f}"),
        ("humidity_span", hum_hi - hum_lo, "%", f"{hum_lo:.1f} to {hum_hi:.1f}"),
        ("rx_value_span", rx_hi - rx_lo, "counts", f"{rx_lo:.0f} to {rx_hi:.0f}"),
        ("corr_rx_loading", rx_loading_corr, "Pearson r", "should be strongly negative"),
        ("corr_loading_flow", loading_flow_corr, "Pearson r", "should be strongly positive"),
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value", "unit", "note"])
        for row in summary_rows:
            writer.writerow(row)


def main() -> int:
    args = parse_args()
    meta_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    rows = generate_rows(args)
    if not rows:
        raise RuntimeError("no pseudo-real rows generated")
    write_csv(args.output, rows)
    write_summary(args.summary, rows, args)
    data_meta = write_run_metadata(
        args.output,
        script=__file__,
        args=meta_args,
        inputs=[],
        outputs=[args.output],
        notes="Pseudo-real canonical X-ray belt weighing data for real-data replacement rehearsal; not field evidence.",
    )
    summary_meta = write_run_metadata(
        args.summary,
        script=__file__,
        args=meta_args,
        inputs=[args.output],
        outputs=[args.summary],
        notes="Profile summary for pseudo-real canonical X-ray belt weighing data.",
    )
    print(f"wrote={args.output}")
    print(f"wrote={data_meta}")
    print(f"wrote={args.summary}")
    print(f"wrote={summary_meta}")
    print(f"rows={len(rows)}")
    print(f"installs={args.sites * args.devices_per_site * args.installs_per_device}")
    print(f"corr_rx_loading={corr(rows, 'rx_value', 'true_loading_kgpm'):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
