from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from evaluation_protocol import PROTOCOL_VERSION, evaluate_installation_holdout, protocol_fields
from experiment_versioning import write_run_metadata
from run_sim_tcn import FEATURE_COLS, SequenceDataset, TCNRegressor, add_features, build_groups, make_frame, metric_row, predict_frame


BASE_COLS = [
    "attenuation",
    "i0_online",
    "phys_flow_feature",
    "tube_voltage_kv",
    "tube_current_ma",
    "source_temp_c",
    "env_temp_c",
    "env_humidity_pct",
    "belt_speed_mps",
    "ash_pct",
    "ash_missing",
    "coal_anthracite",
    "coal_bituminous_A",
    "coal_bituminous_B",
    "coal_lignite",
]


@dataclass(frozen=True)
class TcnConfig:
    seq_len: int
    channels: int
    levels: int

    @property
    def name(self) -> str:
        return f"win{self.seq_len}_c{self.channels}_l{self.levels}"


def parse_configs(raw: str) -> list[TcnConfig]:
    configs: list[TcnConfig] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"invalid config '{item}', expected seq_len:channels:levels")
        configs.append(TcnConfig(seq_len=int(parts[0]), channels=int(parts[1]), levels=int(parts[2])))
    if not configs:
        raise ValueError("at least one TCN config is required")
    return configs


def parse_samples(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def limit_rows_per_group(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0:
        return df
    return (
        df.sort_values(["site_id", "device_id", "install_id", "timestamp_s"])
        .groupby(["site_id", "device_id", "install_id"], group_keys=False)
        .head(max_rows)
        .copy()
    )


def prepare_frame(input_path: Path, max_rows_per_group: int, initial_empty_samples: int) -> pd.DataFrame:
    df = add_features(
        make_frame(
            input_path,
            initial_empty_samples=initial_empty_samples,
            max_rows_per_install=max_rows_per_group,
        )
    )
    train_df = df[df["site_id"].isin(["S01", "S02", "S03"])].copy()
    base_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    base_model.fit(train_df[BASE_COLS], train_df["true_flow_kgps"])
    df["base_flow_pred"] = base_model.predict(df[BASE_COLS])
    df["residual_flow"] = df["true_flow_kgps"] - df["base_flow_pred"]
    return df


def evaluate(model: nn.Module, loader: DataLoader, y_mean: float, y_std: float, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    trues: list[np.ndarray] = []
    with torch.no_grad():
        for x, _residual, true, base in loader:
            x = x.to(device)
            residual_pred = model(x).cpu().numpy() * y_std + y_mean
            pred = base.numpy() + residual_pred
            preds.append(pred)
            trues.append(true.numpy())
    return np.concatenate(trues), np.maximum(np.concatenate(preds), 0)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def train_config(
    df: pd.DataFrame,
    config: TcnConfig,
    *,
    epochs: int,
    batch_size: int,
    train_stride: int,
    num_workers: int,
    device: torch.device,
    seed: int,
    calibrate_samples: list[int],
    adapt_fraction: float,
) -> list[dict[str, float | str | int]]:
    set_seed(seed)
    train_df = df[df["site_id"].isin(["S01", "S02", "S03"])].copy()
    test_df = df[df["site_id"].eq("S04")].copy()
    mean = train_df[FEATURE_COLS].mean()
    std = train_df[FEATURE_COLS].std().replace(0, 1)
    y_mean = float(train_df["residual_flow"].mean())
    y_std = float(train_df["residual_flow"].std()) or 1.0

    train_ds = SequenceDataset(build_groups(train_df, mean, std, y_mean, y_std), config.seq_len, stride=train_stride)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = TCNRegressor(n_features=len(FEATURE_COLS), channels=config.channels, levels=config.levels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    started = time.perf_counter()
    loss_history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for x, y, _true, _base in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.item()))
        epoch_loss = float(np.mean(losses)) if losses else float("nan")
        loss_history.append(epoch_loss)
        if epoch == 1 or epoch == epochs or epoch % 4 == 0:
            print(f"config={config.name} epoch={epoch} train_loss={epoch_loss:.6f}", flush=True)

    predictions = predict_frame(
        model,
        test_df,
        mean,
        std,
        y_mean,
        y_std,
        config.seq_len,
        batch_size,
        device,
    )
    elapsed = time.perf_counter() - started
    params = count_params(model)

    rows: list[dict[str, float | str | int]] = []
    for n in [0, *calibrate_samples]:
        evaluation = evaluate_installation_holdout(
            predictions,
            calibration_samples=n,
            adapt_fraction=adapt_fraction,
        )
        method = f"PG-TSC-TCN {config.name}" if n == 0 else f"PG-TSC-TCN {config.name} calibrate={n}"
        row = metric_row(method, evaluation.y_true, evaluation.y_pred)
        row["calibration_samples_per_install"] = n
        row.update(protocol_fields(evaluation, adapt_fraction))
        rows.append(row)

    for row in rows:
        row["config"] = config.name
        row["seq_len"] = config.seq_len
        row["channels"] = config.channels
        row["levels"] = config.levels
        row["epochs"] = epochs
        row["batch_size"] = batch_size
        row["train_stride"] = train_stride
        row["train_sequences"] = len(train_ds)
        row["test_sequences"] = len(predictions)
        row["tcn_params"] = params
        row["final_train_loss"] = float(loss_history[-1])
        row["elapsed_sec"] = float(elapsed)
        row["device"] = str(device)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TCN window/capacity sensitivity experiments for PG-TSC.")
    parser.add_argument("--input", default="data/simulated/xray_belt_sim.csv")
    parser.add_argument("--output", default="results/sim_tcn_sensitivity_v2.csv")
    parser.add_argument("--configs", default="30:24:3,60:24:3,60:48:4,120:48:4,60:64:4,120:64:5")
    parser.add_argument("--calibrate-samples", default="20,60,180")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-stride", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    parser.add_argument("--initial-empty-samples", type=int, default=60)
    parser.add_argument("--max-rows-per-group", type=int, default=0, help="Limit rows per install group for smoke tests.")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    configs = parse_configs(args.configs)
    calibrate_samples = parse_samples(args.calibrate_samples)
    df = prepare_frame(Path(args.input), args.max_rows_per_group, args.initial_empty_samples)

    all_rows: list[dict[str, float | str | int]] = []
    for idx, config in enumerate(configs):
        rows = train_config(
            df,
            config,
            epochs=args.epochs,
            batch_size=args.batch_size,
            train_stride=args.train_stride,
            num_workers=args.num_workers,
            device=device,
            seed=args.seed + idx,
            calibrate_samples=calibrate_samples,
            adapt_fraction=args.adapt_fraction,
        )
        all_rows.extend(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    meta = write_run_metadata(
        out,
        script="experiments/run_tcn_sensitivity.py",
        args=vars(args),
        inputs=[args.input],
        outputs=[out],
        notes=(
            "TCN sequence length, channel count, and depth sensitivity with chronological "
            f"installation holdout protocol {PROTOCOL_VERSION}."
        ),
    )
    print(f"wrote={out}")
    print(f"wrote={meta}")
    for row in all_rows:
        print(
            f"{row['method']}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, "
            f"CumErr={row['cum_err_pct']:.3f}%, params={row['tcn_params']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
