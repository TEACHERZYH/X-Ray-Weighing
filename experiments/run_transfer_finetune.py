from __future__ import annotations

import argparse
import copy
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from experiment_versioning import write_run_metadata
from run_pretrain_tcn import BASE_COLS
from run_sim_tcn import FEATURE_COLS, SequenceDataset, TCNRegressor, add_features, build_groups, make_frame, metric_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate simulation-pretrained PG-TSC transfer on a shifted target domain.")
    parser.add_argument("--target-input", default="data/simulated/xray_belt_transfer_target.csv")
    parser.add_argument("--pretrained-checkpoint", default="models/pgt_sc_tcn_pretrained_v2.pt")
    parser.add_argument("--base-model", default="models/pgt_sc_base_model_v2.pkl")
    parser.add_argument("--output", default="results/sim_transfer_finetune_results_v2.csv")
    parser.add_argument("--finetuned-checkpoint", default="models/pgt_sc_tcn_transfer_finetuned_v2.pt")
    parser.add_argument("--adapt-fraction", type=float, default=0.2)
    parser.add_argument("--calibration-samples", default="20,60,180")
    parser.add_argument("--calibration-min-flow", type=float, default=5.0)
    parser.add_argument("--finetune-samples-per-install", type=int, default=180)
    parser.add_argument("--finetune-epochs", type=int, default=8)
    parser.add_argument("--scratch-epochs", type=int, default=8)
    parser.add_argument("--mass-loss-weights", default="0.1,0.5,1.0")
    parser.add_argument("--post-finetune-calibration-samples", default="60")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scratch-lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, object]:
    return torch.load(path, map_location="cpu")


def make_model(checkpoint: dict[str, object]) -> TCNRegressor:
    cfg = checkpoint["model_config"]
    model = TCNRegressor(
        n_features=int(cfg["n_features"]),
        channels=int(cfg["channels"]),
        levels=int(cfg["levels"]),
        dropout=float(cfg.get("dropout", 0.05)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def prepare_target_frame(path: Path, base_model_path: Path) -> pd.DataFrame:
    with base_model_path.open("rb") as f:
        base_model = pickle.load(f)
    df = add_features(make_frame(path))
    df["base_flow_pred"] = base_model.predict(df[BASE_COLS])
    df["residual_flow"] = df["true_flow_kgps"] - df["base_flow_pred"]
    return df


def split_adapt_eval(df: pd.DataFrame, adapt_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    adapt_parts = []
    eval_parts = []
    for _, g in df.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = g.sort_values("timestamp_s")
        cutoff = float(g["timestamp_s"].min()) + adapt_fraction * (float(g["timestamp_s"].max()) - float(g["timestamp_s"].min()))
        adapt_parts.append(g[g["timestamp_s"] <= cutoff])
        eval_parts.append(g[g["timestamp_s"] > cutoff])
    return pd.concat(adapt_parts, ignore_index=True), pd.concat(eval_parts, ignore_index=True)


def tensor_predict(model: nn.Module, x: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[start : start + batch_size]).float().to(device)
            preds.append(model(batch).cpu().numpy())
    return np.concatenate(preds)


def predict_rows(
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
    rows = []
    for (site, device_id, install), g in df.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = g.sort_values("timestamp_s").reset_index(drop=True)
        if len(g) < seq_len:
            continue
        x_norm = ((g[FEATURE_COLS] - mean) / std).to_numpy(np.float32)
        sequences = np.stack([x_norm[end - seq_len + 1 : end + 1] for end in range(seq_len - 1, len(g))])
        residual = tensor_predict(model, sequences, batch_size, device) * y_std + y_mean
        base = g["base_flow_pred"].to_numpy(np.float32)[seq_len - 1 :]
        true = g["true_flow_kgps"].to_numpy(np.float32)[seq_len - 1 :]
        timestamp = g["timestamp_s"].to_numpy(np.float32)[seq_len - 1 :]
        pred = np.maximum(base + residual, 0.0)
        rows.append(
            pd.DataFrame(
                {
                    "site_id": site,
                    "device_id": device_id,
                    "install_id": install,
                    "timestamp_s": timestamp,
                    "y_true": true,
                    "y_pred": pred,
                }
            )
        )
    if not rows:
        raise ValueError("no prediction rows generated; check seq_len and target dataset length")
    return pd.concat(rows, ignore_index=True)


def install_calibrate(pred_df: pd.DataFrame, n_samples: int, adapt_fraction: float, min_flow: float) -> pd.DataFrame:
    calibrated = []
    for _, g in pred_df.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = g.sort_values("timestamp_s").copy()
        cutoff = float(g["timestamp_s"].min()) + adapt_fraction * (float(g["timestamp_s"].max()) - float(g["timestamp_s"].min()))
        adapt = g[g["timestamp_s"] <= cutoff]
        loaded = adapt[adapt["y_pred"] >= min_flow]
        if len(loaded) >= 2:
            adapt = loaded
        if len(adapt) >= 2 and n_samples > 0:
            idx = np.linspace(0, len(adapt) - 1, min(n_samples, len(adapt))).astype(int)
            sample = adapt.iloc[idx]
            a, b = np.linalg.lstsq(
                np.column_stack([sample["y_pred"].to_numpy(float), np.ones(len(sample))]),
                sample["y_true"].to_numpy(float),
                rcond=None,
            )[0]
        else:
            a, b = 1.0, 0.0
        g["y_pred"] = np.maximum(a * g["y_pred"].to_numpy(float) + b, 0.0)
        calibrated.append(g)
    return pd.concat(calibrated, ignore_index=True)


def install_mass_ratio_calibrate(pred_df: pd.DataFrame, n_samples: int, adapt_fraction: float, min_flow: float) -> pd.DataFrame:
    calibrated = []
    for _, g in pred_df.groupby(["site_id", "device_id", "install_id"], sort=False):
        g = g.sort_values("timestamp_s").copy()
        cutoff = float(g["timestamp_s"].min()) + adapt_fraction * (float(g["timestamp_s"].max()) - float(g["timestamp_s"].min()))
        adapt = g[g["timestamp_s"] <= cutoff]
        loaded = adapt[adapt["y_pred"] >= min_flow]
        if len(loaded) >= 2:
            adapt = loaded
        if len(adapt) >= 2 and n_samples > 0:
            idx = np.linspace(0, len(adapt) - 1, min(n_samples, len(adapt))).astype(int)
            sample = adapt.iloc[idx]
            denom = float(sample["y_pred"].sum())
            scale = float(sample["y_true"].sum()) / denom if abs(denom) > 1e-9 else 1.0
        else:
            scale = 1.0
        g["y_pred"] = np.maximum(scale * g["y_pred"].to_numpy(float), 0.0)
        calibrated.append(g)
    return pd.concat(calibrated, ignore_index=True)


def eval_region(pred_df: pd.DataFrame, adapt_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    parts = []
    for _, g in pred_df.groupby(["site_id", "device_id", "install_id"], sort=False):
        cutoff = float(g["timestamp_s"].min()) + adapt_fraction * (float(g["timestamp_s"].max()) - float(g["timestamp_s"].min()))
        parts.append(g[g["timestamp_s"] > cutoff])
    eval_df = pd.concat(parts, ignore_index=True)
    return eval_df["y_true"].to_numpy(float), eval_df["y_pred"].to_numpy(float)


def subset_evenly_by_group(dataset: SequenceDataset, max_per_group: int) -> Subset:
    by_group: dict[int, list[int]] = {}
    for idx, (gi, _end) in enumerate(dataset.index):
        by_group.setdefault(gi, []).append(idx)
    selected: list[int] = []
    for indices in by_group.values():
        if len(indices) <= max_per_group:
            selected.extend(indices)
        else:
            take = np.linspace(0, len(indices) - 1, max_per_group).astype(int)
            selected.extend([indices[i] for i in take])
    return Subset(dataset, selected)


def train_residual_model(
    model: nn.Module,
    dataset: SequenceDataset,
    max_per_group: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    num_workers: int,
    y_mean: float = 0.0,
    y_std: float = 1.0,
    mass_loss_weight: float = 0.0,
) -> float:
    train_ds = subset_evenly_by_group(dataset, max_per_group)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model.to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    last_loss = float("nan")
    for _epoch in range(epochs):
        losses = []
        for x, y, true, base in loader:
            x = x.to(device)
            y = y.to(device)
            true = true.to(device)
            base = base.to(device)
            opt.zero_grad()
            residual_norm = model(x)
            point_loss = loss_fn(residual_norm, y)
            if mass_loss_weight > 0:
                pred_flow = torch.clamp(base + residual_norm * y_std + y_mean, min=0.0)
                mass_loss = torch.abs(pred_flow.sum() - true.sum()) / torch.clamp(torch.abs(true.sum()), min=1e-6)
                loss = point_loss + mass_loss_weight * mass_loss
            else:
                loss = point_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.item()))
        last_loss = float(np.mean(losses)) if losses else float("nan")
    return last_loss


def append_metric(rows: list[dict[str, object]], method: str, y_true: np.ndarray, y_pred: np.ndarray, **extra: object) -> None:
    row = metric_row(method, y_true, y_pred)
    row.update(extra)
    rows.append(row)


def parse_float_list(value: str) -> list[float]:
    weights = []
    for item in value.split(","):
        item = item.strip()
        if item:
            weights.append(float(item))
    return weights


def parse_int_list(value: str) -> list[int]:
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


def append_post_calibrated_metrics(
    rows: list[dict[str, object]],
    method: str,
    pred_df: pd.DataFrame,
    budgets: list[int],
    args: argparse.Namespace,
    common: dict[str, object],
    **extra: object,
) -> None:
    for budget in budgets:
        cal_pred = install_calibrate(pred_df, budget, args.adapt_fraction, args.calibration_min_flow)
        y_true, y_pred = eval_region(cal_pred, args.adapt_fraction)
        append_metric(
            rows,
            f"{method} install-calibrate={budget}",
            y_true,
            y_pred,
            label_budget_per_install=max(int(extra.get("label_budget_per_install", 0)), budget),
            finetune_epochs=extra.get("finetune_epochs", 0),
            final_train_loss=extra.get("final_train_loss", ""),
            post_calibration_samples=budget,
            **{k: v for k, v in extra.items() if k not in {"label_budget_per_install", "finetune_epochs", "final_train_loss"}},
            **common,
        )


def append_post_mass_ratio_metrics(
    rows: list[dict[str, object]],
    method: str,
    pred_df: pd.DataFrame,
    budgets: list[int],
    args: argparse.Namespace,
    common: dict[str, object],
    **extra: object,
) -> None:
    for budget in budgets:
        cal_pred = install_mass_ratio_calibrate(pred_df, budget, args.adapt_fraction, args.calibration_min_flow)
        y_true, y_pred = eval_region(cal_pred, args.adapt_fraction)
        append_metric(
            rows,
            f"{method} install-ratio={budget}",
            y_true,
            y_pred,
            label_budget_per_install=max(int(extra.get("label_budget_per_install", 0)), budget),
            finetune_epochs=extra.get("finetune_epochs", 0),
            final_train_loss=extra.get("final_train_loss", ""),
            post_ratio_calibration_samples=budget,
            **{k: v for k, v in extra.items() if k not in {"label_budget_per_install", "finetune_epochs", "final_train_loss"}},
            **common,
        )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)

    checkpoint = load_checkpoint(Path(args.pretrained_checkpoint))
    seq_len = int(checkpoint["model_config"]["seq_len"])
    source_mean = pd.Series(checkpoint["feature_mean"], dtype=float)
    source_std = pd.Series(checkpoint["feature_std"], dtype=float).replace(0, 1)
    source_y_mean = float(checkpoint["residual_mean"])
    source_y_std = float(checkpoint["residual_std"]) or 1.0

    df = prepare_target_frame(Path(args.target_input), Path(args.base_model))
    adapt_df, eval_df = split_adapt_eval(df, args.adapt_fraction)
    pretrained = make_model(checkpoint).to(device)
    direct_pred = predict_rows(pretrained, df, source_mean, source_std, source_y_mean, source_y_std, seq_len, args.batch_size, device)

    rows: list[dict[str, object]] = []
    post_finetune_budgets = parse_int_list(args.post_finetune_calibration_samples)
    y_true, y_pred = eval_region(direct_pred, args.adapt_fraction)
    common = {
        "target_rows": int(len(df)),
        "adapt_rows": int(len(adapt_df)),
        "eval_rows": int(len(eval_df)),
        "seq_len": seq_len,
        "adapt_fraction": args.adapt_fraction,
        "device": str(device),
    }
    append_metric(rows, "Pretrained direct", y_true, y_pred, label_budget_per_install=0, finetune_epochs=0, final_train_loss="", **common)

    for budget in [int(x) for x in args.calibration_samples.split(",") if x.strip()]:
        cal_pred = install_calibrate(direct_pred, budget, args.adapt_fraction, args.calibration_min_flow)
        y_true, y_pred = eval_region(cal_pred, args.adapt_fraction)
        append_metric(rows, f"Pretrained install-calibrate={budget}", y_true, y_pred, label_budget_per_install=budget, finetune_epochs=0, final_train_loss="", **common)
        ratio_pred = install_mass_ratio_calibrate(direct_pred, budget, args.adapt_fraction, args.calibration_min_flow)
        y_true, y_pred = eval_region(ratio_pred, args.adapt_fraction)
        append_metric(rows, f"Pretrained install-ratio={budget}", y_true, y_pred, label_budget_per_install=budget, finetune_epochs=0, final_train_loss="", **common)

    adapt_groups_source = build_groups(adapt_df, source_mean, source_std, source_y_mean, source_y_std)
    adapt_ds_source = SequenceDataset(adapt_groups_source, seq_len, stride=1)

    finetuned = copy.deepcopy(pretrained)
    ft_loss = train_residual_model(
        finetuned,
        adapt_ds_source,
        args.finetune_samples_per_install,
        args.finetune_epochs,
        args.batch_size,
        args.lr,
        device,
        args.num_workers,
        source_y_mean,
        source_y_std,
        0.0,
    )
    ft_pred = predict_rows(finetuned, df, source_mean, source_std, source_y_mean, source_y_std, seq_len, args.batch_size, device)
    y_true, y_pred = eval_region(ft_pred, args.adapt_fraction)
    append_metric(
        rows,
        "Pretrained residual-finetune",
        y_true,
        y_pred,
        label_budget_per_install=args.finetune_samples_per_install,
        finetune_epochs=args.finetune_epochs,
        final_train_loss=ft_loss,
        **common,
    )
    append_post_calibrated_metrics(
        rows,
        "Pretrained residual-finetune",
        ft_pred,
        post_finetune_budgets,
        args,
        common,
        label_budget_per_install=args.finetune_samples_per_install,
        finetune_epochs=args.finetune_epochs,
        final_train_loss=ft_loss,
    )
    append_post_mass_ratio_metrics(
        rows,
        "Pretrained residual-finetune",
        ft_pred,
        post_finetune_budgets,
        args,
        common,
        label_budget_per_install=args.finetune_samples_per_install,
        finetune_epochs=args.finetune_epochs,
        final_train_loss=ft_loss,
    )

    for mass_weight in parse_float_list(args.mass_loss_weights):
        mass_tuned = copy.deepcopy(pretrained)
        mass_loss = train_residual_model(
            mass_tuned,
            adapt_ds_source,
            args.finetune_samples_per_install,
            args.finetune_epochs,
            args.batch_size,
            args.lr,
            device,
            args.num_workers,
            source_y_mean,
            source_y_std,
            mass_weight,
        )
        mass_pred = predict_rows(mass_tuned, df, source_mean, source_std, source_y_mean, source_y_std, seq_len, args.batch_size, device)
        y_true, y_pred = eval_region(mass_pred, args.adapt_fraction)
        append_metric(
            rows,
            f"Pretrained residual-finetune mass={mass_weight:g}",
            y_true,
            y_pred,
            label_budget_per_install=args.finetune_samples_per_install,
            finetune_epochs=args.finetune_epochs,
            final_train_loss=mass_loss,
            mass_loss_weight=mass_weight,
            **common,
        )
        append_post_calibrated_metrics(
            rows,
            f"Pretrained residual-finetune mass={mass_weight:g}",
            mass_pred,
            post_finetune_budgets,
            args,
            common,
            label_budget_per_install=args.finetune_samples_per_install,
            finetune_epochs=args.finetune_epochs,
            final_train_loss=mass_loss,
            mass_loss_weight=mass_weight,
        )
        append_post_mass_ratio_metrics(
            rows,
            f"Pretrained residual-finetune mass={mass_weight:g}",
            mass_pred,
            post_finetune_budgets,
            args,
            common,
            label_budget_per_install=args.finetune_samples_per_install,
            finetune_epochs=args.finetune_epochs,
            final_train_loss=mass_loss,
            mass_loss_weight=mass_weight,
        )

    target_mean = adapt_df[FEATURE_COLS].mean()
    target_std = adapt_df[FEATURE_COLS].std().replace(0, 1)
    target_y_mean = float(adapt_df["residual_flow"].mean())
    target_y_std = float(adapt_df["residual_flow"].std()) or 1.0
    scratch = TCNRegressor(
        n_features=len(FEATURE_COLS),
        channels=int(checkpoint["model_config"]["channels"]),
        levels=int(checkpoint["model_config"]["levels"]),
        dropout=float(checkpoint["model_config"].get("dropout", 0.05)),
    )
    adapt_ds_target = SequenceDataset(build_groups(adapt_df, target_mean, target_std, target_y_mean, target_y_std), seq_len, stride=1)
    scratch_loss = train_residual_model(
        scratch,
        adapt_ds_target,
        args.finetune_samples_per_install,
        args.scratch_epochs,
        args.batch_size,
        args.scratch_lr,
        device,
        args.num_workers,
        target_y_mean,
        target_y_std,
        0.0,
    )
    scratch_pred = predict_rows(scratch, df, target_mean, target_std, target_y_mean, target_y_std, seq_len, args.batch_size, device)
    y_true, y_pred = eval_region(scratch_pred, args.adapt_fraction)
    append_metric(
        rows,
        "Scratch target-only",
        y_true,
        y_pred,
        label_budget_per_install=args.finetune_samples_per_install,
        finetune_epochs=args.scratch_epochs,
        final_train_loss=scratch_loss,
        **common,
    )

    out = Path(args.output)
    finetuned_path = Path(args.finetuned_checkpoint)
    out.parent.mkdir(parents=True, exist_ok=True)
    finetuned_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    torch.save(
        {
            "model_state_dict": finetuned.to("cpu").state_dict(),
            "source_checkpoint": str(args.pretrained_checkpoint),
            "args": vars(args),
            "final_train_loss": ft_loss,
        },
        finetuned_path,
    )
    meta = write_run_metadata(
        out,
        script=__file__,
        args=vars(args),
        inputs=[args.target_input, args.pretrained_checkpoint, args.base_model],
        outputs=[out, finetuned_path],
        notes="Target-domain transfer and few-sample fine-tuning validation for simulation-pretrained PG-TSC.",
    )
    print(f"device={device}")
    print(f"target_rows={len(df)} adapt_rows={len(adapt_df)} eval_rows={len(eval_df)}")
    print(f"wrote={out}")
    print(f"wrote={finetuned_path}")
    print(f"wrote={meta}")
    print(f"elapsed_s={time.perf_counter() - started:.1f}")
    for row in rows:
        print(f"{row['method']}: MAE={row['mae']:.3f}, RMSE={row['rmse']:.3f}, CumErr={row['cum_err_pct']:.3f}%")


if __name__ == "__main__":
    main()
