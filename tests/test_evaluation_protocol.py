from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
if str(EXPERIMENTS) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS))

from evaluation_protocol import evaluate_installation_holdout  # noqa: E402
from run_sim_baselines import add_online_features  # noqa: E402


class EvaluationProtocolTests(unittest.TestCase):
    def test_calibration_and_evaluation_are_chronologically_disjoint(self) -> None:
        rows = []
        for install in ["I01", "I02"]:
            for timestamp in range(100):
                rows.append(
                    {
                        "site_id": "S04",
                        "device_id": "D01",
                        "install_id": install,
                        "timestamp_s": timestamp,
                        "y_true": float(timestamp + 1),
                        "y_pred": float(timestamp + 1) * 0.9,
                    }
                )
        frame = pd.DataFrame(rows)
        result = evaluate_installation_holdout(
            frame,
            calibration_samples=5,
            adapt_fraction=0.2,
        )
        self.assertEqual(result.adapt_rows, 40)
        self.assertEqual(result.eval_rows, 160)
        self.assertEqual(result.calibration_samples_actual, 10)
        self.assertEqual(result.installations, 2)
        self.assertEqual(result.eval_duration_s, 160.0)
        self.assertEqual(result.eval_true_mass_kg, 9680.0)
        self.assertTrue(np.all(result.y_true >= 21.0))

    def test_online_i0_does_not_depend_on_future_samples(self) -> None:
        rows = self._sensor_rows()
        changed = copy.deepcopy(rows)
        for row in changed[80:]:
            row["rx_value"] = float(row["rx_value"]) * 0.25

        add_online_features(rows, initial_empty_samples=10)
        add_online_features(changed, initial_empty_samples=10)

        for left, right in zip(rows[:80], changed[:80]):
            self.assertAlmostEqual(float(left["i0_online"]), float(right["i0_online"]), places=12)
            self.assertAlmostEqual(float(left["attenuation"]), float(right["attenuation"]), places=12)

    def test_calibration_candidate_count_does_not_depend_on_unknown_labels(self) -> None:
        rows = []
        for timestamp in range(20):
            rows.append(
                {
                    "site_id": "S04",
                    "device_id": "D01",
                    "install_id": "I01",
                    "timestamp_s": timestamp,
                    "y_true": float(timestamp * 12),
                    "y_pred": float(timestamp * 10),
                }
            )
        original = pd.DataFrame(rows)
        changed = original.copy()
        changed.loc[:9, "y_true"] = 0.0

        left = evaluate_installation_holdout(original, calibration_samples=4, adapt_fraction=0.5)
        right = evaluate_installation_holdout(changed, calibration_samples=4, adapt_fraction=0.5)

        self.assertEqual(left.calibration_samples_actual, 4)
        self.assertEqual(right.calibration_samples_actual, 4)

    def test_online_i0_does_not_use_empty_flag(self) -> None:
        rows = self._sensor_rows()
        changed = copy.deepcopy(rows)
        for row in changed:
            row["empty_flag"] = 1 - int(row["empty_flag"])

        add_online_features(rows, initial_empty_samples=10)
        add_online_features(changed, initial_empty_samples=10)

        np.testing.assert_allclose(
            [float(row["i0_online"]) for row in rows],
            [float(row["i0_online"]) for row in changed],
            rtol=0.0,
            atol=0.0,
        )

    @staticmethod
    def _sensor_rows() -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for timestamp in range(120):
            rx = 9000.0 + 0.5 * timestamp if timestamp < 20 else 6500.0 + timestamp
            rows.append(
                {
                    "site_id": "S04",
                    "device_id": "D01",
                    "install_id": "I01",
                    "timestamp_s": timestamp,
                    "rx_value": rx,
                    "empty_flag": int(timestamp < 20),
                }
            )
        return rows


if __name__ == "__main__":
    unittest.main()
