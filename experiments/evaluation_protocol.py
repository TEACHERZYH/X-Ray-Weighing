from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PROTOCOL_VERSION = "causal-install-holdout-v2"
GROUP_COLS = ["site_id", "device_id", "install_id"]


@dataclass(frozen=True)
class ProtocolEvaluation:
    y_true: np.ndarray
    y_pred: np.ndarray
    adapt_rows: int
    eval_rows: int
    calibration_samples_actual: int
    installations: int
    eval_duration_s: float
    eval_true_mass_kg: float


def _split_position(length: int, adapt_fraction: float) -> int:
    if not 0.0 < adapt_fraction < 1.0:
        raise ValueError("adapt_fraction must be between 0 and 1")
    if length < 2:
        raise ValueError("each installation needs at least two prediction rows")
    return max(1, min(length - 1, int(np.floor(length * adapt_fraction))))


def select_time_uniform_indices(length: int, budget: int) -> np.ndarray:
    if budget <= 0 or length <= 0:
        return np.empty(0, dtype=int)
    count = min(int(budget), int(length))
    return np.unique(np.linspace(0, length - 1, count).round().astype(int))


def fit_affine_calibrator(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, float]:
    if len(y_pred) != len(y_true):
        raise ValueError("calibration predictions and labels must have equal length")
    if len(y_pred) < 2:
        return 1.0, 0.0
    if float(np.ptp(y_pred)) < 1e-8:
        denominator = float(y_pred.sum())
        return (float(y_true.sum()) / denominator, 0.0) if abs(denominator) > 1e-9 else (1.0, 0.0)
    design = np.column_stack([y_pred, np.ones(len(y_pred), dtype=float)])
    a, b = np.linalg.lstsq(design, y_true, rcond=None)[0]
    return float(a), float(b)


def evaluate_installation_holdout(
    prediction_frame: pd.DataFrame,
    *,
    calibration_samples: int,
    adapt_fraction: float = 0.2,
    pred_col: str = "y_pred",
    true_col: str = "y_true",
    min_calibration_flow: float = 5.0,
) -> ProtocolEvaluation:
    required = [*GROUP_COLS, "timestamp_s", pred_col, true_col]
    missing = [col for col in required if col not in prediction_frame]
    if missing:
        raise ValueError(f"prediction frame missing columns: {missing}")

    true_parts: list[np.ndarray] = []
    pred_parts: list[np.ndarray] = []
    adapt_rows = 0
    calibration_samples_actual = 0
    installations = 0
    eval_duration_s = 0.0
    eval_true_mass_kg = 0.0

    for _key, group in prediction_frame.groupby(GROUP_COLS, sort=False):
        ordered = group.sort_values("timestamp_s", kind="mergesort").reset_index(drop=True)
        cut = _split_position(len(ordered), adapt_fraction)
        adapt = ordered.iloc[:cut]
        evaluation = ordered.iloc[cut:]
        adapt_rows += len(adapt)
        installations += 1

        a, b = 1.0, 0.0
        if calibration_samples > 0:
            candidates = adapt[adapt[pred_col].astype(float) >= float(min_calibration_flow)]
            if len(candidates) >= 2:
                local_idx = select_time_uniform_indices(len(candidates), calibration_samples)
                sample = candidates.iloc[local_idx]
                calibration_samples_actual += len(sample)
                a, b = fit_affine_calibrator(
                    sample[pred_col].to_numpy(float),
                    sample[true_col].to_numpy(float),
                )

        true_parts.append(evaluation[true_col].to_numpy(float))
        timestamps = evaluation["timestamp_s"].to_numpy(float)
        positive_steps = np.diff(timestamps)
        positive_steps = positive_steps[positive_steps > 0]
        sample_interval_s = float(np.median(positive_steps)) if len(positive_steps) else 1.0
        eval_duration_s += float(len(evaluation) * sample_interval_s)
        eval_true_mass_kg += float(evaluation[true_col].to_numpy(float).sum() * sample_interval_s)
        pred_parts.append(
            np.maximum(a * evaluation[pred_col].to_numpy(float) + b, 0.0)
        )

    if not true_parts:
        raise ValueError("prediction frame contains no installations")

    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    return ProtocolEvaluation(
        y_true=y_true,
        y_pred=y_pred,
        adapt_rows=adapt_rows,
        eval_rows=len(y_true),
        calibration_samples_actual=calibration_samples_actual,
        installations=installations,
        eval_duration_s=eval_duration_s,
        eval_true_mass_kg=eval_true_mass_kg,
    )


def prediction_frame_from_rows(
    rows: list[dict[str, object]],
    predictions: np.ndarray,
    *,
    true_col: str = "true_flow_kgps",
) -> pd.DataFrame:
    if len(rows) != len(predictions):
        raise ValueError("row and prediction counts must match")
    frame = pd.DataFrame(
        {
            col: [row[col] for row in rows]
            for col in [*GROUP_COLS, "timestamp_s"]
        }
    )
    frame["y_true"] = np.asarray([float(row[true_col]) for row in rows], dtype=float)
    frame["y_pred"] = np.asarray(predictions, dtype=float)
    return frame


def protocol_fields(result: ProtocolEvaluation, adapt_fraction: float) -> dict[str, object]:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "adapt_fraction": float(adapt_fraction),
        "adapt_rows": int(result.adapt_rows),
        "eval_rows": int(result.eval_rows),
        "calibration_samples_actual": int(result.calibration_samples_actual),
        "installations": int(result.installations),
        "eval_duration_s": float(result.eval_duration_s),
        "eval_true_mass_kg": float(result.eval_true_mass_kg),
    }
