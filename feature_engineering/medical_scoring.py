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
    Calculate SOFA respiratory sub-score using SaO2 and FiO2.
    Estimation: PaO2 ≈ (SaO2 - 30) * 2 for SaO2 > 80%
    Ratio = PaO2 / FiO2.
    0: > 400
    1: ≤ 400
    2: ≤ 300
    3: ≤ 200
    4: ≤ 100
    """
    if pd.isna(sao2) or pd.isna(fio2):
        # Missing SaO2 or FiO2: assume no respiratory dysfunction
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
    Add all SOFA sub-score columns to df and return the augmented DataFrame.
    """
    # Compute and attach individual SOFA subscores directly
    df["renal_score"] = df["Creatinine"].round(1).apply(sofa_renal_score)
    df["coagulation_score"] = df["Platelets"].apply(sofa_coagulation_score)
    df["liver_score"] = df["Bilirubin_total"].round(1).apply(sofa_liver_score)
    df["respiratory_score"] = df.apply(
        lambda row: sofa_respiratory_score(row["SaO2"], row["FiO2"]), axis=1
    )
    # Sum into total SOFA score
    df["sofa_score"] = df[
        ["renal_score", "coagulation_score", "liver_score", "respiratory_score"]
    ].sum(axis=1)
    return df


def news_sbp_score(sbp: float) -> int:
    """
    NEWS sub-score for systolic blood pressure (mmHg).
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
    NEWS sub-score for oxygen saturation (%).
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
    NEWS sub-score for supplemental oxygen. FiO2 > 0.21 indicates supplemental O₂.
    """
    if pd.isna(fio2):
        # Missing FiO2: assume no supplemental oxygen
        return 0
    return 2 if fio2 > 0.21 else 0


def news_hr_score(heart_rate: float) -> int:
    """
    NEWS sub-score for heart rate.
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
    NEWS sub-score for respiratory rate.
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
    NEWS sub-score for temperature (°C).
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
    Add all NEWS sub-score columns and 'news_score' total to df.
    """
    # Compute and attach individual NEWS subscores directly
    df["news_hr_score"] = df["HR"].apply(news_hr_score)
    df["news_resp_score"] = df["Resp"].apply(news_resp_score)
    df["news_temp_score"] = df["Temp"].apply(news_temp_score)
    df["news_sbp_score"] = df["SBP"].apply(news_sbp_score)
    df["news_o2sat_score"] = df["O2Sat"].apply(news_o2sat_score)
    df["news_supplemental_o2_score"] = df["FiO2"].apply(news_supplemental_o2_score)
    # Sum into total NEWS score
    df["news_score"] = df[
        [
            "news_hr_score",
            "news_resp_score",
            "news_temp_score",
            "news_sbp_score",
            "news_o2sat_score",
            "news_supplemental_o2_score",
        ]
    ].sum(axis=1)
    return df


def qsofa_resp_score(resp_rate: float) -> int:
    """
    qSOFA sub-score for respiratory rate.
    1 point if ≥ 22 breaths/min.
    """
    if pd.isna(resp_rate):
        # Missing respiratory rate: assume normal
        return 0
    return 1 if resp_rate >= 22 else 0


def qsofa_sbp_score(sbp: float) -> int:
    """
    qSOFA sub-score for systolic blood pressure.
    1 point if ≤ 100 mmHg.
    """
    if pd.isna(sbp):
        # Missing SBP: assume normal
        return 0
    return 1 if sbp <= 100 else 0


def qsofa_gcs_score(gcs: float) -> int:
    """
    qSOFA sub-score for altered mental status (GCS).
    1 point if GCS < 15.
    """
    if pd.isna(gcs):
        # Missing GCS: assume normal
        return 0
    return 1 if gcs < 15 else 0


def add_qsofa_score(
    df: pd.DataFrame, resp_col: str = "Resp", sbp_col: str = "SBP", gcs_col: str = "GCS"
) -> pd.DataFrame:
    """
    Add qSOFA component scores and total qSOFA to DataFrame.
    """
    df["qsofa_resp_score"] = df[resp_col].apply(qsofa_resp_score)
    df["qsofa_sbp_score"] = df[sbp_col].apply(qsofa_sbp_score)
    df["qsofa_gcs_score"] = df[gcs_col].apply(qsofa_gcs_score)
    df["qsofa_score"] = df[
        ["qsofa_resp_score", "qsofa_sbp_score", "qsofa_gcs_score"]
    ].sum(axis=1)
    return df
