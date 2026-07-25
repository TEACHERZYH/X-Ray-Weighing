from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiment_versioning import write_run_metadata


@dataclass(frozen=True)
class CoalType:
    name: str
    mu_base: float
    ash_mean: float
    moisture_mean: float
    particle_noise: float


COAL_TYPES = [
    CoalType("bituminous_A", 0.00380, 12.0, 7.5, 0.010),
    CoalType("bituminous_B", 0.00405, 18.0, 9.0, 0.014),
    CoalType("anthracite", 0.00355, 9.0, 5.5, 0.008),
    CoalType("lignite", 0.00435, 24.0, 13.0, 0.018),
]


def smooth_random_walk(n: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    steps = rng.normal(0.0, scale, size=n)
    y = np.cumsum(steps)
    y -= y.mean()
    return y


def make_loading_profile(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    loading = np.zeros(n)
    empty = np.ones(n, dtype=int)
    i = 0
    while i < n:
        empty_len = int(rng.integers(60, 240))
        i += empty_len
        if i >= n:
            break
        run_len = int(rng.integers(600, 1800))
        j = min(n, i + run_len)
        base = rng.uniform(45, 130)
        wave = 0.18 * base * np.sin(np.linspace(0, rng.uniform(3, 8) * math.pi, j - i))
        noise = rng.normal(0, 0.06 * base, size=j - i)
        bursts = np.zeros(j - i)
        for _ in range(int(rng.integers(2, 6))):
            center = int(rng.integers(0, j - i))
            width = int(rng.integers(20, 120))
            amp = rng.uniform(-0.25, 0.35) * base
            x = np.arange(j - i)
            bursts += amp * np.exp(-((x - center) ** 2) / (2 * width**2))
        loading[i:j] = np.maximum(0, base + wave + noise + bursts)
        empty[i:j] = 0
        i = j
    return loading, empty


def simulate_install(site_idx: int, device_idx: int, install_idx: int, n: int, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    site_id = f"S{site_idx:02d}"
    device_id = f"D{device_idx:02d}"
    install_id = f"I{install_idx:02d}"

    coal = random.choice(COAL_TYPES)
    t = np.arange(n, dtype=float)
    loading, empty_flag = make_loading_profile(n, rng)

    env_temp = 18 + 8 * np.sin(2 * math.pi * t / max(n, 1) + rng.uniform(0, math.pi)) + smooth_random_walk(n, 0.015, rng)
    env_humidity = 55 + 18 * np.sin(2 * math.pi * t / max(n * 0.8, 1) + rng.uniform(0, math.pi)) + smooth_random_walk(n, 0.03, rng)
    env_humidity = np.clip(env_humidity, 20, 95)

    tube_voltage = 80 + rng.normal(0, 0.35, n) + smooth_random_walk(n, 0.002, rng)
    tube_current = 1.5 + rng.normal(0, 0.015, n) + smooth_random_walk(n, 0.0002, rng)
    source_temp = env_temp + 15 + 0.015 * loading + smooth_random_walk(n, 0.01, rng)

    belt_speed = 2.4 + rng.normal(0, 0.025, n) + 0.04 * np.sin(2 * math.pi * t / 900 + rng.uniform(0, math.pi))
    belt_speed = np.clip(belt_speed, 1.8, 3.0)

    ash_true = coal.ash_mean + rng.normal(0, 1.8, n) + 0.4 * np.sin(2 * math.pi * t / 1200)
    moisture_hidden = coal.moisture_mean + rng.normal(0, 1.2, n) + 0.12 * (env_humidity - 55)
    particle_hidden = rng.normal(0, coal.particle_noise, n)

    ash_missing = rng.random(n) < rng.uniform(0.25, 0.55)
    ash_estimate = coal.ash_mean + rng.normal(0, 0.7, n)
    ash_input = np.where(ash_missing, ash_estimate, ash_true)

    device_gain = rng.normal(1.0, 0.035)
    install_gap_effect = rng.normal(1.0, 0.025)
    detector_bias = rng.normal(0.0, 12.0)
    base_i0 = rng.uniform(7800, 9600)

    i0 = (
        base_i0
        * device_gain
        * install_gap_effect
        * (tube_current / 1.5)
        * (tube_voltage / 80) ** 1.7
        * (1 - 0.0018 * (source_temp - 35))
        * (1 - 0.0004 * (env_humidity - 55))
    )
    i0 *= 1 + smooth_random_walk(n, 0.000018, rng)

    mu_eff = (
        coal.mu_base
        * (1 + 0.010 * (ash_true - coal.ash_mean))
        * (1 + 0.006 * (moisture_hidden - coal.moisture_mean))
        * (1 - 0.003 * (tube_voltage - 80))
        * (1 + particle_hidden)
    )

    rx_value = i0 * np.exp(-mu_eff * loading) + detector_bias + rng.normal(0, 18.0, n)
    rx_value = np.maximum(rx_value, 1.0)

    true_flow = loading * belt_speed
    true_cum = np.cumsum(true_flow)

    rows: list[dict[str, object]] = []
    for k in range(n):
        rows.append(
            {
                "timestamp_s": int(k),
                "site_id": site_id,
                "device_id": device_id,
                "install_id": install_id,
                "rx_value": round(float(rx_value[k]), 6),
                "tube_voltage_kv": round(float(tube_voltage[k]), 6),
                "tube_current_ma": round(float(tube_current[k]), 6),
                "source_temp_c": round(float(source_temp[k]), 6),
                "env_temp_c": round(float(env_temp[k]), 6),
                "env_humidity_pct": round(float(env_humidity[k]), 6),
                "belt_speed_mps": round(float(belt_speed[k]), 6),
                "empty_flag": int(empty_flag[k]),
                "coal_type": coal.name,
                "ash_pct": round(float(ash_input[k]), 6),
                "ash_missing": int(ash_missing[k]),
                "true_loading_kgpm": round(float(loading[k]), 6),
                "true_flow_kgps": round(float(true_flow[k]), 6),
                "true_cum_mass_kg": round(float(true_cum[k]), 6),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/simulated/xray_belt_sim.csv")
    parser.add_argument("--sites", type=int, default=4)
    parser.add_argument("--devices-per-site", type=int, default=2)
    parser.add_argument("--installs-per-device", type=int, default=2)
    parser.add_argument("--seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=20260623)
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    run_seed = args.seed
    for s in range(1, args.sites + 1):
        for d in range(1, args.devices_per_site + 1):
            for i in range(1, args.installs_per_device + 1):
                all_rows.extend(simulate_install(s, d, i, args.seconds, run_seed))
                run_seed += 1

    fieldnames = list(all_rows[0].keys())
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    total_mass = sum(float(r["true_flow_kgps"]) for r in all_rows)
    empty_ratio = sum(int(r["empty_flag"]) for r in all_rows) / len(all_rows)
    ash_missing_ratio = sum(int(r["ash_missing"]) for r in all_rows) / len(all_rows)
    meta = write_run_metadata(
        out,
        script=__file__,
        args=vars(args),
        inputs=[],
        outputs=[out],
        notes="Simulated X-ray belt weighing dataset. CSV is intentionally git-ignored; commit this metadata with downstream result tables.",
    )
    print(f"wrote={out}")
    print(f"wrote={meta}")
    print(f"rows={len(all_rows)}")
    print(f"duration_per_install_s={args.seconds}")
    print(f"total_mass_kg={total_mass:.2f}")
    print(f"empty_ratio={empty_ratio:.3f}")
    print(f"ash_missing_ratio={ash_missing_ratio:.3f}")


if __name__ == "__main__":
    main()
