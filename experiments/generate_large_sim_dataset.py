from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from experiment_versioning import write_run_metadata
from simulate_xray_belt import simulate_install


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a large streamed simulation dataset for PG-TSC pretraining.")
    parser.add_argument("--output", default="data/simulated/xray_belt_pretrain_large.csv")
    parser.add_argument("--summary", default="results/sim_pretrain_dataset_summary.csv")
    parser.add_argument("--sites", type=int, default=16)
    parser.add_argument("--devices-per-site", type=int, default=3)
    parser.add_argument("--installs-per-device", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=7200)
    parser.add_argument("--seed", type=int, default=20260626)
    return parser.parse_args()


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
    fieldnames: list[str] | None = None
    run_seed = args.seed

    with out.open("w", newline="", encoding="utf-8") as f:
        writer: csv.DictWriter[str] | None = None
        for site_idx in range(1, args.sites + 1):
            for device_idx in range(1, args.devices_per_site + 1):
                for install_idx in range(1, args.installs_per_device + 1):
                    rows = simulate_install(site_idx, device_idx, install_idx, args.seconds, run_seed)
                    run_seed += 1
                    if not rows:
                        continue
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
        raise RuntimeError("simulation generated no rows")

    summary_row: dict[str, object] = {
        "dataset": f"{out.parent.name}/{out.name}",
        "rows": rows_written,
        "sites": args.sites,
        "devices_per_site": args.devices_per_site,
        "installs_per_device": args.installs_per_device,
        "seconds_per_install": args.seconds,
        "install_count": args.sites * args.devices_per_site * args.installs_per_device,
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
        notes="Large streamed simulation dataset for simulation-pretrained PG-TSC.",
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
