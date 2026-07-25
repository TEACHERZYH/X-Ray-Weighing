from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset

from evaluation_protocol import (
    PROTOCOL_VERSION,
    evaluate_installation_holdout,
    protocol_fields,
)
from experiment_versioning import write_run_metadata
from run_sim_baselines import add_online_features, limit_rows_per_install, read_rows


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_frame(path: Path, *, initial_empty_samples: int = 60, max_rows_per_install: int = 0) -> pd.DataFrame:
    rows = read_rows(path)
    rows = limit_rows_per_install(rows, max_rows_per_install)
    add_online_features(rows, initial_empty_samples=initial_empty_samples)
    df = pd.DataFrame(rows)
    df["phys_flow_feature"] = df["attenuation"] * df["belt_speed_mps"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    coal = pd.get_dummies(df["coal_type"], prefix="coal", dtype=float)
    df = pd.concat([df.reset_index(drop=True), coal.reset_index(drop=True)], axis=1)
    for name in ["coal_anthracite", "coal_bituminous_A", "coal_bituminous_B", "coal_lignite"]:
        if name not in df:
            df[name] = 0.0
    return df


FEATURE_COLS = [
    "rx_value",
    "tube_voltage_kv",
    "tube_current_ma",
    "source_temp_c",
    "env_temp_c",
    "env_humidity_pct",
    "belt_speed_mps",
    "ash_pct",
    "ash_missing",
    "i0_online",
    "attenuation",
    "phys_flow_feature",
    "attenuation_mean30",
    "attenuation_std30",
    "rx_mean30",
    "rx_std30",
    "coal_anthracite",
    "coal_bituminous_A",
    "coal_bituminous_B",
    "coal_lignite",
]


class SequenceDataset(Dataset):
    def __init__(self, groups: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], seq_len: int, stride: int = 1):
        self.groups = groups
        self.seq_len = seq_len
        self.index: list[tuple[int, int]] = []
        for gi, (x, _residual, _true, _base) in enumerate(groups):
            for end in range(seq_len - 1, len(x), stride):
                self.index.append((gi, end))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gi, end = self.index[idx]
        x, residual, true, base = self.groups[gi]
        start = end - self.seq_len + 1
        return (
            torch.from_numpy(x[start : end + 1].copy()).float(),
            torch.tensor(residual[end]).float(),
            torch.tensor(true[end]).float(),
            torch.tensor(base[end]).float(),
        )


class Chomp1d(nn.Module):
    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp] if self.chomp > 0 else x


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.net(x) + self.downsample(x))


class TCNRegressor(nn.Module):
    def __init__(self, n_features: int, channels: int = 48, levels: int = 4, dropout: float = 0.05):
        super().__init__()
        blocks = []
        in_ch = n_features
        for i in range(levels):
            blocks.append(TemporalBlock(in_ch, channels, kernel_size=3, dilation=2**i, dropout=dropout))
            in_ch = channels
        self.tcn = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.tcn(x)[:, :, -1]
        return self.head(h).squeeze(-1)


def build_groups(df: pd.DataFrame, mean: pd.Series, std: pd.Series, y_mean: float, y_std: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    groups = []
    for _, g in df.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = g.sort_values("timestamp_s")
        x = ((g[FEATURE_COLS] - mean) / std).to_numpy(np.float32)
        residual = ((g["residual_flow"].to_numpy(np.float32) - y_mean) / y_std).astype(np.float32)
        true = g["true_flow_kgps"].to_numpy(np.float32)
        base = g["base_flow_pred"].to_numpy(np.float32)
        groups.append((x, residual, true, base))
    return groups


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    y_mean: float,
    y_std: float,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    device = device or next(model.parameters()).device
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for x, _residual, true, base in loader:
            residual_pred = model(x.to(device)).cpu() * y_std + y_mean
            pred = base + residual_pred
            preds.append(pred.numpy())
            trues.append(true.numpy())
    yp = np.concatenate(preds)
    yt = np.concatenate(trues)
    yp = np.maximum(yp, 0)
    return yt, yp


def metric_row(method: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    return {
        "method": method,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "cum_err_pct": float(abs(y_pred.sum() - y_true.sum()) / max(y_true.sum(), 1e-9) * 100),
    }


def predict_frame(
    model: nn.Module,
    df: pd.DataFrame,
    mean: pd.Series,
    std: pd.Series,
    y_mean: float,
    y_std: float,
    seq_len: int,
    batch_size: int,
    device: torch.device,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for key, group in df.groupby(["site_id", "device_id", "install_id"], sort=False):
        ordered = group.sort_values("timestamp_s", kind="mergesort").reset_index(drop=True)
        dataset = SequenceDataset(build_groups(ordered, mean, std, y_mean, y_std), seq_len, stride=1)
        if len(dataset) == 0:
            continue
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        y_true, y_pred = evaluate(model, loader, y_mean, y_std, device)
        aligned = ordered.iloc[seq_len - 1 :].copy()
        if len(aligned) != len(y_true):
            raise RuntimeError("TCN prediction alignment failed")
        frame = pd.DataFrame(
            {
                "site_id": key[0],
                "device_id": key[1],
                "install_id": key[2],
                "timestamp_s": aligned["timestamp_s"].to_numpy(float),
                "y_true": y_true,
                "y_pred": y_pred,
            }
        )
        frames.append(frame)
    if not frames:
        raise ValueError("no target installation is long enough for the requested sequence length")
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/simulated/xray_belt_sim.csv")
    parser.add_argument("--output", default="results/sim_tcn_results_v2.csv")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=24)
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--train-stride", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-samples", default="20,60,180")
    parser.add_argument("--initial-empty-samples", type=int, default=60)
    parser.add_argument("--max-rows-per-install", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    df = add_features(
        make_frame(
            Path(args.input),
            initial_empty_samples=args.initial_empty_samples,
            max_rows_per_install=args.max_rows_per_install,
        )
    )
    train_df = df[df["site_id"].isin(["S01", "S02", "S03"])].copy()
    test_df = df[df["site_id"].eq("S04")].copy()

    base_cols = [
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
    base_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    base_model.fit(train_df[base_cols], train_df["true_flow_kgps"])
    df["base_flow_pred"] = base_model.predict(df[base_cols])
    df["residual_flow"] = df["true_flow_kgps"] - df["base_flow_pred"]
    train_df = df[df["site_id"].isin(["S01", "S02", "S03"])].copy()
    test_df = df[df["site_id"].eq("S04")].copy()

    mean = train_df[FEATURE_COLS].mean()
    std = train_df[FEATURE_COLS].std().replace(0, 1)
    y_mean = float(train_df["residual_flow"].mean())
    y_std = float(train_df["residual_flow"].std())

    train_ds = SequenceDataset(build_groups(train_df, mean, std, y_mean, y_std), args.seq_len, stride=args.train_stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    model = TCNRegressor(n_features=len(FEATURE_COLS), channels=args.channels, levels=args.levels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, _true, _base in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch == args.epochs or epoch % 4 == 0:
            print(f"epoch={epoch} train_loss={np.mean(losses):.5f}", flush=True)

    predictions = predict_frame(
        model,
        test_df,
        mean,
        std,
        y_mean,
        y_std,
        args.seq_len,
        args.batch_size,
        device,
    )
    rows: list[dict[str, object]] = []
    sample_budgets = [0, *[int(value.strip()) for value in args.calibration_samples.split(",") if value.strip()]]
    for n in sample_budgets:
        evaluation = evaluate_installation_holdout(
            predictions,
            calibration_samples=n,
            adapt_fraction=args.adapt_fraction,
        )
        method = "PG-TSC-TCN" if n == 0 else f"PG-TSC-TCN calibrate={n}"
        rows.append(
            {
                **metric_row(method, evaluation.y_true, evaluation.y_pred),
                "calibration_samples_per_install": n,
                **protocol_fields(evaluation, args.adapt_fraction),
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    meta = write_run_metadata(
        out,
        script=__file__,
        args=vars(args),
        inputs=[args.input],
        outputs=[out],
        notes=(
            "TCN residual experiment with causal I0 features and chronological installation "
            f"holdout protocol {PROTOCOL_VERSION}."
        ),
    )
    print(f"device={device}")
    print(f"wrote={out}")
    print(f"wrote={meta}")
    for row in rows:
        print(f"{row['method']}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, CumErr={row['cum_err_pct']:.3f}%")


if __name__ == "__main__":
    main()
