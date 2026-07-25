from __future__ import annotations

import argparse
import pickle
import time
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain a PG-TSC TCN residual model on a large simulated dataset.")
    parser.add_argument("--input", default="data/simulated/xray_belt_pretrain_large.csv")
    parser.add_argument("--output", default="results/sim_pretrain_tcn_results_v2.csv")
    parser.add_argument("--checkpoint", default="models/pgt_sc_tcn_pretrained_v2.pt")
    parser.add_argument("--base-model", default="models/pgt_sc_base_model_v2.pkl")
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=48)
    parser.add_argument("--levels", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--train-stride", type=int, default=4)
    parser.add_argument("--holdout-site-count", type=int, default=4)
    parser.add_argument("--holdout-sites", default="", help="Comma-separated site IDs. Overrides --holdout-site-count.")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-samples", default="20,60,180,600")
    parser.add_argument("--initial-empty-samples", type=int, default=60)
    return parser.parse_args()


def site_key(site: str) -> tuple[int, str]:
    digits = "".join(ch for ch in site if ch.isdigit())
    return (int(digits) if digits else 0, site)


def split_sites(df: pd.DataFrame, holdout_sites: str, holdout_site_count: int) -> tuple[list[str], list[str]]:
    sites = sorted([str(s) for s in df["site_id"].unique()], key=site_key)
    if holdout_sites.strip():
        holdout = [s.strip() for s in holdout_sites.split(",") if s.strip()]
    else:
        n = max(1, min(holdout_site_count, len(sites) - 1))
        holdout = sites[-n:]
    train = [s for s in sites if s not in holdout]
    if not train or not holdout:
        raise ValueError(f"invalid split: train={train}, holdout={holdout}")
    return train, holdout


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
    yp = np.maximum(np.concatenate(preds), 0)
    yt = np.concatenate(trues)
    return yt, yp


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)

    started = time.perf_counter()
    df = add_features(make_frame(Path(args.input), initial_empty_samples=args.initial_empty_samples))
    train_sites, holdout_sites = split_sites(df, args.holdout_sites, args.holdout_site_count)
    train_df = df[df["site_id"].isin(train_sites)].copy()
    holdout_df = df[df["site_id"].isin(holdout_sites)].copy()

    base_model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    base_model.fit(train_df[BASE_COLS], train_df["true_flow_kgps"])
    df["base_flow_pred"] = base_model.predict(df[BASE_COLS])
    df["residual_flow"] = df["true_flow_kgps"] - df["base_flow_pred"]
    train_df = df[df["site_id"].isin(train_sites)].copy()
    holdout_df = df[df["site_id"].isin(holdout_sites)].copy()

    mean = train_df[FEATURE_COLS].mean()
    std = train_df[FEATURE_COLS].std().replace(0, 1)
    y_mean = float(train_df["residual_flow"].mean())
    y_std = float(train_df["residual_flow"].std()) or 1.0

    train_ds = SequenceDataset(build_groups(train_df, mean, std, y_mean, y_std), args.seq_len, stride=args.train_stride)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = TCNRegressor(
        n_features=len(FEATURE_COLS),
        channels=args.channels,
        levels=args.levels,
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    loss_history: list[float] = []

    print(
        f"device={device} rows={len(df)} train_rows={len(train_df)} holdout_rows={len(holdout_df)} "
        f"train_sequences={len(train_ds)}",
        flush=True,
    )
    print(f"train_sites={','.join(train_sites)} holdout_sites={','.join(holdout_sites)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
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
        if epoch == 1 or epoch == args.epochs or epoch % 4 == 0:
            print(f"epoch={epoch} train_loss={epoch_loss:.6f}", flush=True)

    predictions = predict_frame(
        model,
        holdout_df,
        mean,
        std,
        y_mean,
        y_std,
        args.seq_len,
        args.batch_size,
        device,
    )
    rows: list[dict[str, float | str | int]] = []
    budgets = [0, *[int(value.strip()) for value in args.calibration_samples.split(",") if value.strip()]]
    for n in budgets:
        evaluation = evaluate_installation_holdout(
            predictions,
            calibration_samples=n,
            adapt_fraction=args.adapt_fraction,
        )
        method = "PG-TSC-TCN-pretrain" if n == 0 else f"PG-TSC-TCN-pretrain calibrate={n}"
        row = metric_row(method, evaluation.y_true, evaluation.y_pred)
        row["calibration_samples_per_install"] = n
        row.update(protocol_fields(evaluation, args.adapt_fraction))
        rows.append(row)
    for row in rows:
        row["train_rows"] = int(len(train_df))
        row["holdout_rows"] = int(len(holdout_df))
        row["train_sequences"] = int(len(train_ds))
        row["holdout_sequences"] = int(len(predictions))
        row["epochs"] = int(args.epochs)
        row["final_train_loss"] = float(loss_history[-1])
        row["train_sites"] = ",".join(train_sites)
        row["holdout_sites"] = ",".join(holdout_sites)

    out = Path(args.output)
    checkpoint = Path(args.checkpoint)
    base_model_path = Path(args.base_model)
    out.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    base_model_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    with base_model_path.open("wb") as f:
        pickle.dump(base_model, f)

    model_cpu = model.to("cpu")
    torch.save(
        {
            "model_state_dict": model_cpu.state_dict(),
            "model_config": {
                "n_features": len(FEATURE_COLS),
                "channels": args.channels,
                "levels": args.levels,
                "dropout": args.dropout,
                "seq_len": args.seq_len,
            },
            "feature_cols": FEATURE_COLS,
            "base_cols": BASE_COLS,
            "feature_mean": mean.to_dict(),
            "feature_std": std.to_dict(),
            "residual_mean": y_mean,
            "residual_std": y_std,
            "base_model_path": base_model_path.as_posix(),
            "train_sites": train_sites,
            "holdout_sites": holdout_sites,
            "loss_history": loss_history,
            "args": vars(args),
        },
        checkpoint,
    )

    meta = write_run_metadata(
        out,
        script=__file__,
        args=vars(args),
        inputs=[args.input],
        outputs=[out, checkpoint, base_model_path],
        notes=(
            "Large-simulation pretraining run with causal I0 features and chronological "
            f"installation holdout protocol {PROTOCOL_VERSION}."
        ),
    )
    elapsed = time.perf_counter() - started
    print(f"wrote={out}")
    print(f"wrote={checkpoint}")
    print(f"wrote={base_model_path}")
    print(f"wrote={meta}")
    print(f"elapsed_s={elapsed:.1f}")
    for row in rows:
        print(f"{row['method']}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, CumErr={row['cum_err_pct']:.3f}%")


if __name__ == "__main__":
    main()
