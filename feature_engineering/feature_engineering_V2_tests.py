import unittest
from copy import deepcopy

import numpy as np
import pandas as pd
from feature_engineering_V2 import generate_window_features


class TestGenerateWindowFeatures(unittest.TestCase):
    def setUp(self):
        # Single‑patient, seven time‑steps, increasing vitals
        self.single_df = pd.DataFrame(
            {
                "patient_id": [1] * 7,
                "ICULOS": list(range(7)),
                "HR": [80, 85, 90, 95, 100, 110, 120],
                "O2Sat": [95, 94, 93, 92, 91, 90, 89],
            }
        )
        # Two patients, three time‑steps each
        self.multi_df = pd.DataFrame(
            {
                "patient_id": [1, 1, 1, 2, 2, 2],
                "ICULOS": [0, 1, 2, 0, 1, 2],
                "HR": [80, 85, 90, 60, 65, 70],
                "O2Sat": [95, 94, 93, 98, 97, 96],
            }
        )

    def test_columns_added(self):
        """All expected rolling‑window feature columns are present."""
        cols = ["HR", "O2Sat"]
        out = generate_window_features(self.single_df, cols)
        expected_suffixes = [
            "max_6h",
            "min_6h",
            "mean_6h",
            "median_6h",
            "std_6h",
            "diff_std_6h",
        ]
        for c in cols:
            for suf in expected_suffixes:
                with self.subTest(column=c, suffix=suf):
                    self.assertIn(f"{c}_{suf}", out.columns)

    def test_values_correct_for_single_patient(self):
        """HR_max_6h matches pandas rolling max exactly."""
        out = generate_window_features(self.single_df, ["HR"])
        expected = (
            self.single_df["HR"].rolling(window=6, min_periods=1).max().to_numpy()
        )
        actual = out["HR_max_6h"].to_numpy()
        self.assertTrue(
            np.allclose(actual, expected, equal_nan=True),
            "HR_max_6h does not match pandas rolling max",
        )

    def test_multiple_patients_independent(self):
        """Rolling windows reset per patient."""
        out = generate_window_features(self.multi_df, ["HR"])
        first = out.sort_values(["patient_id", "ICULOS"]).groupby("patient_id").head(1)
        # on each patient's first row, max_6h == the raw HR
        for idx, row in first.iterrows():
            self.assertEqual(
                row["HR_max_6h"],
                row["HR"],
                f"Patient {row['patient_id']} first-window wrong",
            )

    def test_input_dataframe_not_mutated(self):
        """Original DataFrame must remain unchanged."""
        before = deepcopy(self.single_df)
        _ = generate_window_features(self.single_df, ["HR"])
        pd.testing.assert_frame_equal(self.single_df, before)


if __name__ == "__main__":
    unittest.main()
