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
"""

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
    df["renal_score"] = df["Creatinine"].round(1).apply(sofa_renal_score)
    df["coagulation_score"] = df["Platelets"].apply(sofa_coagulation_score)
    df["liver_score"] = df["Bilirubin_total"].round(1).apply(sofa_liver_score)
    df["respiratory_score"] = df.apply(
        lambda row: sofa_respiratory_score(row["SaO2"], row["FiO2"]), axis=1
    )
    df["sofa_score"] = df[
        ["renal_score", "coagulation_score", "liver_score", "respiratory_score"]
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
    df["qsofa_resp_score"] = df[resp_col].apply(qsofa_resp_score)
    df["qsofa_sbp_score"] = df[sbp_col].apply(qsofa_sbp_score)

    df["qsofa_gcs_score"] = 0
    df["qsofa_score"] = df[["qsofa_resp_score", "qsofa_sbp_score"]].sum(axis=1)

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


def test_functions():
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
    assert "renal_score" in df.columns
    assert "coagulation_score" in df.columns
    assert "liver_score" in df.columns
    assert "respiratory_score" in df.columns
    assert "sofa_score" in df.columns
    assert "NEWS_HR_score" in df.columns
    assert "NEWS_Resp_score" in df.columns
    assert "NEWS_Temp_score" in df.columns
    assert "NEWS_SBP_score" in df.columns
    assert "NEWS_O2Sat_score" in df.columns
    assert "NEWS_FiO2_score" in df.columns
    assert "NEWS_score" in df.columns
    assert "qsofa_resp_score" in df.columns
    assert "qsofa_sbp_score" in df.columns
    assert "qsofa_gcs_score" in df.columns
    assert "qsofa_score" in df.columns
    return df


if __name__ == "__main__":
    df = test_functions()
    print(df.columns)
    print("TESTS PASSED")
