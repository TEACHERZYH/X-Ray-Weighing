from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import torch

from experiment_versioning import write_run_metadata
from run_sim_tcn import FEATURE_COLS, TCNRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark PG-TSC TCN inference cost on synthetic online inputs.")
    parser.add_argument("--output", default="results/sim_deployment_efficiency.csv")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=120)
    parser.add_argument("--batches", default="1,32,256")
    parser.add_argument("--seed", type=int, default=20260628)
    return parser.parse_args()


def receptive_field(levels: int, kernel_size: int = 3, convs_per_block: int = 2) -> int:
    return 1 + convs_per_block * (kernel_size - 1) * sum(2**i for i in range(levels))


def macs_per_sequence(n_features: int, seq_len: int, channels: int, levels: int, kernel_size: int = 3) -> int:
    macs = 0
    in_ch = n_features
    for _ in range(levels):
        macs += seq_len * channels * in_ch * kernel_size
        macs += seq_len * channels * channels * kernel_size
        if in_ch != channels:
            macs += seq_len * channels * in_ch
        in_ch = channels
    macs += channels * 32 + 32
    return int(macs)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def benchmark(model: torch.nn.Module, batch: int, seq_len: int, n_features: int, warmup: int, repeats: int) -> tuple[float, float]:
    x = torch.randn(batch, seq_len, n_features)
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(x)
        started = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        elapsed = time.perf_counter() - started
    mean_batch_ms = elapsed * 1000.0 / repeats
    mean_sample_ms = mean_batch_ms / batch
    return mean_batch_ms, mean_sample_ms


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    batches = [int(item) for item in args.batches.split(",") if item.strip()]
    n_features = len(FEATURE_COLS)
    configs = [
        ("PG-TSC-TCN-small", 24, 3),
        ("PG-TSC-TCN-pretrain", 48, 4),
    ]
    rows: list[dict[str, object]] = []
    for model_name, channels, levels in configs:
        model = TCNRegressor(n_features=n_features, channels=channels, levels=levels, dropout=0.0)
        params = count_parameters(model)
        rf = receptive_field(levels)
        macs = macs_per_sequence(n_features, args.seq_len, channels, levels)
        for batch in batches:
            mean_batch_ms, mean_sample_ms = benchmark(model, batch, args.seq_len, n_features, args.warmup, args.repeats)
            throughput = 1000.0 / mean_sample_ms if mean_sample_ms > 0 else 0.0
            rows.append(
                {
                    "model": model_name,
                    "seq_len": args.seq_len,
                    "n_features": n_features,
                    "channels": channels,
                    "levels": levels,
                    "receptive_field_samples": rf,
                    "parameters": params,
                    "macs_per_sequence": macs,
                    "batch_size": batch,
                    "mean_batch_latency_ms": mean_batch_ms,
                    "mean_sample_latency_ms": mean_sample_ms,
                    "throughput_samples_per_s": throughput,
                    "torch_threads": args.threads,
                    "device": "cpu",
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "platform": platform.platform(),
                }
            )
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    write_run_metadata(
        output,
        script="experiments/analyze_deployment_efficiency.py",
        args=vars(args),
        inputs=[],
        outputs=[output],
        notes="Module-level CPU inference benchmark for PG-TSC TCN deployment feasibility using synthetic online inputs.",
    )
    print(f"wrote={output}")
    for row in rows:
        if row["model"] == "PG-TSC-TCN-pretrain" and row["batch_size"] == 1:
            print(
                "pretrain_batch1="
                f"{float(row['mean_sample_latency_ms']):.4f} ms/sample, "
                f"params={int(row['parameters'])}, macs={int(row['macs_per_sequence'])}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
