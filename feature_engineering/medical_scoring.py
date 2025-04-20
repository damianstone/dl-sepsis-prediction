"""
Medical Scoring Features

This file provides functions to calculate medical scoring systems used
in sepsis prediction, including:
- SOFA (Sequential Organ Failure Assessment)
- NEWS (National Early Warning Score)
- qSOFA (quick Sequential Organ Failure Assessment)

Usage:
    from medical_scoring import add_medical_scores
    df = pd.read_parquet(INPUT_DATASET)
    df = add_medical_scores(df)

Required columns in input DataFrame:
    - SOFA: Creatinine, Platelets, Bilirubin_total, SaO2, FiO2
    - NEWS: HR, Resp, Temp, SBP, O2Sat, FiO2
    - qSOFA: Resp, SBP

New columns added to input DataFrame:
    - SOFA_Creatinine, SOFA_Platelets, SOFA_Bilirubin_total, SOFA_SaO2_FiO2, SOFA_score
    - NEWS_HR_score, NEWS_Resp_score, NEWS_Temp_score, NEWS_SBP_score, NEWS_O2Sat_score, NEWS_FiO2_score, NEWS_score
    - qSOFA_Resp_score, qSOFA_SBP_score, qSOFA_score
"""

import unittest
from pathlib import Path

import pandas as pd


# Original sub-score functions
def sofa_renal_score(creatinine: float) -> int:
    """
    Calculate SOFA renal sub-score based on serum creatinine (mg/dL).
    0: < 1.2
    1: 1.2–1.9
    2: 2.0–3.4
    3: 3.5–4.9
    4: ≥ 5.0
    """
    if pd.isna(creatinine):
        # Missing creatinine: assume normal renal function
        return 0
    creatinine = round(creatinine, 1)
    if creatinine < 1.2:
        return 0
    elif creatinine <= 1.9:
        return 1
    elif creatinine <= 3.4:
        return 2
    elif creatinine <= 4.9:
        return 3
    else:
        return 4


def sofa_coagulation_score(platelets: float) -> int:
    """
    Calculate SOFA coagulation sub-score based on platelet count (×10^3/µL).
    0: > 150
    1: ≤ 150
    2: ≤ 100
    3: ≤ 50
    4: ≤ 20
    """
    if pd.isna(platelets):
        # Missing platelet count: assume no coagulopathy
        return 0
    if platelets > 150:
        return 0
    elif platelets > 100:
        return 1
    elif platelets > 50:
        return 2
    elif platelets > 20:
        return 3
    else:
        return 4


def sofa_liver_score(bilirubin: float) -> int:
    """
    Calculate SOFA liver sub-score based on total bilirubin (mg/dL).
    0: < 1.2
    1: 1.2–1.9
    2: 2.0–5.9
    3: 6.0–11.9
    4: ≥ 12.0
    """
    if pd.isna(bilirubin):
        # Missing bilirubin: assume normal liver function
        return 0
    bilirubin = round(bilirubin, 1)
    if bilirubin < 1.2:
        return 0
    elif bilirubin < 2.0:
        return 1
    elif bilirubin < 6.0:
        return 2
    elif bilirubin < 12.0:
        return 3
    else:
        return 4


def sofa_respiratory_score(sao2: float, fio2: float) -> int:
    """
    Calculate SOFA respiratory sub-score based on PaO2/FiO2 ratio.
    PaO2 is estimated as (SaO2 - 30) * 2 for SaO2 > 80%.
    0: > 400
    1: ≤ 400
    2: ≤ 300
    3: ≤ 200
    4: ≤ 100
    """
    if pd.isna(sao2) or pd.isna(fio2) or fio2 == 0:
        # Missing SaO2 or FiO2 or Zero FiO2: assume no respiratory dysfunction
        return 0
    pao2 = (sao2 - 30) * 2
    ratio = pao2 / fio2

    if ratio > 400:
        return 0
    elif ratio > 300:
        return 1
    elif ratio > 200:
        return 2
    elif ratio > 100:
        return 3
    else:
        return 4


def add_sofa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add SOFA sub-score columns and total SOFA score to the DataFrame.
    """
    df["SOFA_Creatinine"] = df["Creatinine"].round(1).apply(sofa_renal_score)
    df["SOFA_Platelets"] = df["Platelets"].apply(sofa_coagulation_score)
    df["SOFA_Bilirubin_total"] = df["Bilirubin_total"].round(1).apply(sofa_liver_score)
    df["SOFA_SaO2_FiO2"] = df.apply(
        lambda row: sofa_respiratory_score(row["SaO2"], row["FiO2"]), axis=1
    )
    df["SOFA_score"] = df[
        ["SOFA_Creatinine", "SOFA_Platelets", "SOFA_Bilirubin_total", "SOFA_SaO2_FiO2"]
    ].sum(axis=1)
    return df


def news_sbp_score(sbp: float) -> int:
    """
    Calculate NEWS sub-score for systolic blood pressure (mmHg).
    3: ≤ 90 or ≥ 220
    2: 91-100
    1: 101-110
    0: 111-219
    """
    if pd.isna(sbp):
        # Missing SBP: assume normal
        return 0
    if sbp <= 90:
        return 3
    elif 91 <= sbp <= 100:
        return 2
    elif 101 <= sbp <= 110:
        return 1
    elif 111 <= sbp <= 219:
        return 0
    else:  # ≥220
        return 3


def news_o2sat_score(o2sat: float) -> int:
    """
    Calculate NEWS sub-score for oxygen saturation (%).
    3: ≤ 91
    2: 92-93
    1: 94-95
    0: ≥ 96
    """
    if pd.isna(o2sat):
        # Missing O2 saturation: assume normal
        return 0
    if o2sat <= 91:
        return 3
    elif 92 <= o2sat <= 93:
        return 2
    elif 94 <= o2sat <= 95:
        return 1
    else:  # ≥96
        return 0


def news_supplemental_o2_score(fio2: float) -> int:
    """
    Calculate NEWS sub-score for supplemental oxygen.
    2: FiO2 > 0.21 (indicates supplemental O₂)
    0: FiO2 ≤ 0.21
    """
    if pd.isna(fio2):
        # Missing FiO2: assume no supplemental oxygen
        return 0
    return 2 if fio2 > 0.21 else 0


def news_hr_score(heart_rate: float) -> int:
    """
    Calculate NEWS sub-score for heart rate (bpm).
    3: ≤ 40 or ≥ 131
    2: 111-130
    1: 41-50 or 91-110
    0: 51-90
    """
    if pd.isna(heart_rate):
        # Missing heart rate: assume normal
        return 0
    if heart_rate <= 40:
        return 3
    elif 41 <= heart_rate <= 50:
        return 1
    elif 51 <= heart_rate <= 90:
        return 0
    elif 91 <= heart_rate <= 110:
        return 1
    elif 111 <= heart_rate <= 130:
        return 2
    else:  # ≥131
        return 3


def news_resp_score(resp_rate: float) -> int:
    """
    Calculate NEWS sub-score for respiratory rate (breaths/min).
    3: ≤ 8 or ≥ 25
    2: 21-24
    1: 9-11
    0: 12-20
    """
    if pd.isna(resp_rate):
        # Missing respiratory rate: assume normal
        return 0
    if resp_rate <= 8:
        return 3
    elif 9 <= resp_rate <= 11:
        return 1
    elif 12 <= resp_rate <= 20:
        return 0
    elif 21 <= resp_rate <= 24:
        return 2
    else:  # ≥25
        return 3


def news_temp_score(temp: float) -> int:
    """
    Calculate NEWS sub-score for temperature (°C).
    3: ≤ 35.0
    2: ≥ 39.1
    1: 35.1-36.0 or 38.1-39.0
    0: 36.1-38.0
    """
    if pd.isna(temp):
        # Missing temperature: assume normal
        return 0
    if temp <= 35.0:
        return 3
    elif 35.1 <= temp <= 36.0:
        return 1
    elif 36.1 <= temp <= 38.0:
        return 0
    elif 38.1 <= temp <= 39.0:
        return 1
    else:  # ≥39.1
        return 2


def add_news_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add NEWS sub-score columns and total NEWS score to the DataFrame.
    """
    df["NEWS_HR_score"] = df["HR"].apply(news_hr_score)
    df["NEWS_Resp_score"] = df["Resp"].apply(news_resp_score)
    df["NEWS_Temp_score"] = df["Temp"].apply(news_temp_score)
    df["NEWS_SBP_score"] = df["SBP"].apply(news_sbp_score)
    df["NEWS_O2Sat_score"] = df["O2Sat"].apply(news_o2sat_score)
    df["NEWS_FiO2_score"] = df["FiO2"].apply(news_supplemental_o2_score)
    df["NEWS_score"] = df[
        [
            "NEWS_HR_score",
            "NEWS_Resp_score",
            "NEWS_Temp_score",
            "NEWS_SBP_score",
            "NEWS_O2Sat_score",
            "NEWS_FiO2_score",
        ]
    ].sum(axis=1)
    return df


def qsofa_resp_score(resp_rate: float) -> int:
    """
    Calculate qSOFA sub-score for respiratory rate.
    1: ≥ 22 breaths/min
    0: < 22 breaths/min
    """
    if pd.isna(resp_rate):
        # Missing respiratory rate: assume normal
        return 0
    return 1 if resp_rate >= 22 else 0


def qsofa_sbp_score(sbp: float) -> int:
    """
    Calculate qSOFA sub-score for systolic blood pressure.
    1: ≤ 100 mmHg
    0: > 100 mmHg
    """
    if pd.isna(sbp):
        # Missing SBP: assume normal
        return 0
    return 1 if sbp <= 100 else 0


def qsofa_gcs_score(gcs: float) -> int:
    """
    Calculate qSOFA sub-score for altered mental status (GCS).
    1: GCS < 15
    0: GCS = 15
    """
    if pd.isna(gcs):
        # Missing GCS: assume normal
        return 0
    return 1 if gcs < 15 else 0


def add_qsofa_score(
    df: pd.DataFrame, resp_col: str = "Resp", sbp_col: str = "SBP"
) -> pd.DataFrame:
    """
    Add qSOFA sub-score columns and total qSOFA score to the DataFrame.
    """
    df["qSOFA_Resp_score"] = df[resp_col].apply(qsofa_resp_score)
    df["qSOFA_SBP_score"] = df[sbp_col].apply(qsofa_sbp_score)
    df["qSOFA_score"] = df[["qSOFA_Resp_score", "qSOFA_SBP_score"]].sum(axis=1)

    return df


def find_project_root(marker=".gitignore"):
    """
    Find project root by walking up from current working directory until
    a directory containing the specified marker is found.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}"
    )


def add_medical_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all medical scores (SOFA, NEWS, qSOFA) to the DataFrame.
    """
    df = df.copy()
    df = add_sofa_scores(df)
    df = add_news_scores(df)
    df = add_qsofa_score(df)
    return df


NaN = float("nan")  # convenience alias


class TestSOFASubScores(unittest.TestCase):
    def test_renal(self):
        cases = [
            (NaN, 0),
            (0.8, 0),
            (1.14, 0),  # just below threshold
            (1.20, 1),
            (1.9, 1),
            (2.0, 2),
            (3.4, 2),
            (3.5, 3),
            (4.96, 4),
            (5.0, 4),
        ]
        for creatinine, expected in cases:
            with self.subTest(creatinine=creatinine):
                self.assertEqual(sofa_renal_score(creatinine), expected)

    def test_coagulation(self):
        cases = [
            (NaN, 0),
            (200, 0),
            (150, 1),
            (101, 1),
            (100, 2),
            (51, 2),
            (50, 3),
            (21, 3),
            (20, 4),
        ]
        for platelets, expected in cases:
            with self.subTest(platelets=platelets):
                self.assertEqual(sofa_coagulation_score(platelets), expected)

    def test_liver(self):
        cases = [
            (NaN, 0),
            (0.5, 0),
            (1.14, 0),
            (1.2, 1),
            (1.9, 1),
            (2.0, 2),
            (5.9, 2),
            (6.0, 3),
            (11.96, 4),
            (12.0, 4),
        ]
        for bilirubin, expected in cases:
            with self.subTest(bilirubin=bilirubin):
                self.assertEqual(sofa_liver_score(bilirubin), expected)

    def test_respiratory(self):
        # Based on implementation: PaO2 = (SaO2 - 30)*2; ratio = PaO2 / FiO2
        # Score: >400→0; >300→1; >200→2; >100→3; else→4
        cases = [
            (NaN, 0.21, 0),  # missing SaO2
            (95, NaN, 0),  # missing FiO2
            (95, 0.0, 0),  # FiO2 zero treated as missing
            (95, 0.21, 0),  # ratio ≈619 → 0
            (92, 0.30, 0),  # ratio ≈413 → 0
            (88, 0.35, 1),  # ratio ≈331 → 1
            (90, 0.90, 3),  # ratio ≈133 → 3
            (85, 1.0, 3),  # ratio = 110 → 3
            (80, 0.25, 1),  # ratio = 400 → 1 (boundary)
        ]
        for sao2, fio2, expected in cases:
            with self.subTest(sao2=sao2, fio2=fio2):
                self.assertEqual(sofa_respiratory_score(sao2, fio2), expected)

    def test_aggregate(self):
        df = pd.DataFrame(
            {
                "Creatinine": [0.8, 5.2],
                "Platelets": [250, 15],
                "Bilirubin_total": [0.9, 15.0],
                "SaO2": [97, 80],
                "FiO2": [0.21, 1.0],
            }
        )
        df = add_sofa_scores(df)
        # row 0 all zeros
        self.assertEqual(df.loc[0, "SOFA_score"], 0)
        # row 1 has 4 in every domain → 16 total
        self.assertEqual(df.loc[1, "SOFA_score"], 16)


class TestNEWSSubScores(unittest.TestCase):
    def test_hr(self):
        cases = [(NaN, 0), (35, 3), (45, 1), (70, 0), (100, 1), (120, 2), (131, 3)]
        for hr, expected in cases:
            with self.subTest(hr=hr):
                self.assertEqual(news_hr_score(hr), expected)

    def test_resp(self):
        cases = [(NaN, 0), (8, 3), (10, 1), (16, 0), (22, 2), (25, 3)]
        for resp, expected in cases:
            with self.subTest(resp=resp):
                self.assertEqual(news_resp_score(resp), expected)

    def test_temp(self):
        cases = [(NaN, 0), (34.9, 3), (35.5, 1), (37.0, 0), (38.5, 1), (40.2, 2)]
        for temp, expected in cases:
            with self.subTest(temp=temp):
                self.assertEqual(news_temp_score(temp), expected)

    def test_sbp(self):
        cases = [(NaN, 0), (85, 3), (95, 2), (105, 1), (150, 0), (225, 3)]
        for sbp, expected in cases:
            with self.subTest(sbp=sbp):
                self.assertEqual(news_sbp_score(sbp), expected)

    def test_o2sat(self):
        cases = [(NaN, 0), (90, 3), (92, 2), (95, 1), (98, 0)]
        for sat, expected in cases:
            with self.subTest(sat=sat):
                self.assertEqual(news_o2sat_score(sat), expected)

    def test_supplemental_o2(self):
        cases = [(NaN, 0), (0.21, 0), (0.30, 2)]
        for fio2, expected in cases:
            with self.subTest(fio2=fio2):
                self.assertEqual(news_supplemental_o2_score(fio2), expected)

    def test_aggregate(self):
        df = pd.DataFrame(
            {
                "HR": [70, 150],
                "Resp": [16, 27],
                "Temp": [37.0, 34.0],
                "SBP": [120, 85],
                "O2Sat": [97, 88],
                "FiO2": [0.21, 0.60],
            }
        )
        df = add_news_scores(df)
        self.assertEqual(df.loc[0, "NEWS_score"], 0)
        self.assertEqual(df.loc[1, "NEWS_score"], 3 + 3 + 3 + 3 + 3 + 2)


class TestQSOFASubScores(unittest.TestCase):
    def test_resp(self):
        cases = [(NaN, 0), (20, 0), (22, 1), (30, 1)]
        for resp, expected in cases:
            with self.subTest(resp=resp):
                self.assertEqual(qsofa_resp_score(resp), expected)

    def test_sbp(self):
        cases = [(NaN, 0), (120, 0), (100, 1), (80, 1)]
        for sbp, expected in cases:
            with self.subTest(sbp=sbp):
                self.assertEqual(qsofa_sbp_score(sbp), expected)

    def test_gcs(self):
        cases = [(NaN, 0), (15, 0), (14, 1), (5, 1)]
        for gcs, expected in cases:
            with self.subTest(gcs=gcs):
                self.assertEqual(qsofa_gcs_score(gcs), expected)

    def test_aggregate(self):
        df = pd.DataFrame({"Resp": [18, 25], "SBP": [120, 90]})
        df = add_qsofa_score(df)
        self.assertEqual(df.loc[0, "qSOFA_score"], 0)
        self.assertEqual(df.loc[1, "qSOFA_score"], 2)


class TestCombinedWrapper(unittest.TestCase):
    def test_add_medical_scores_columns(self):
        """Smoke test to confirm all expected columns are added."""
        df = pd.DataFrame(
            {
                "Creatinine": [0.8],
                "Platelets": [250],
                "Bilirubin_total": [0.9],
                "SaO2": [97],
                "FiO2": [0.21],
                "HR": [70],
                "Resp": [16],
                "Temp": [37.0],
                "SBP": [120],
                "O2Sat": [97],
            }
        )
        df = add_medical_scores(df)
        expected = {
            # SOFA
            "SOFA_Creatinine",
            "SOFA_Platelets",
            "SOFA_Bilirubin_total",
            "SOFA_SaO2_FiO2",
            "SOFA_score",
            # NEWS
            "NEWS_HR_score",
            "NEWS_Resp_score",
            "NEWS_Temp_score",
            "NEWS_SBP_score",
            "NEWS_O2Sat_score",
            "NEWS_FiO2_score",
            "NEWS_score",
            # qSOFA
            "qSOFA_Resp_score",
            "qSOFA_SBP_score",
            "qSOFA_score",
        }
        self.assertEqual(
            set(expected).issubset(df.columns),
            True,
            msg=f"Missing columns: {expected.difference(df.columns)}",
        )


def test_df():
    """
    Test the scoring functions by loading a dataset and applying all scores.
    """
    root = find_project_root()
    INPUT_DATASET = f"{root}/dataset/Fully_imputed_dataset.parquet"
    df = pd.read_parquet(INPUT_DATASET)
    df = add_sofa_scores(df)
    df = add_news_scores(df)
    df = add_qsofa_score(df)
    # test if the columns are added correctly
    assert "SOFA_Creatinine" in df.columns
    assert "SOFA_Platelets" in df.columns
    assert "SOFA_Bilirubin_total" in df.columns
    assert "SOFA_SaO2_FiO2" in df.columns
    assert "SOFA_score" in df.columns
    assert "NEWS_HR_score" in df.columns
    assert "NEWS_Resp_score" in df.columns
    assert "NEWS_Temp_score" in df.columns
    assert "NEWS_SBP_score" in df.columns
    assert "NEWS_O2Sat_score" in df.columns
    assert "NEWS_FiO2_score" in df.columns
    assert "NEWS_score" in df.columns
    assert "qSOFA_Resp_score" in df.columns
    assert "qSOFA_SBP_score" in df.columns
    assert "qSOFA_score" in df.columns
    return df


if __name__ == "__main__":
    df = test_df()
    print(df.columns)
    print("TESTS PASSED")
    unittest.main()
