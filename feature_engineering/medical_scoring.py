import pandas as pd


# Original sub-score functions
def renal_score(creatinine: float) -> int:
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


def coagulation_score(platelets: float) -> int:
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


def liver_score(bilirubin: float) -> int:
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


def respiratory_score(sao2: float, fio2: float) -> int:
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


# DataFrame helper functions


def add_renal_score(
    df: pd.DataFrame, creatinine_col: str = "Creatinine"
) -> pd.DataFrame:
    """
    Add a 'renal_score' column to df based on the Creatinine column.
    """
    df["renal_score"] = df[creatinine_col].round(1).apply(renal_score)
    return df


def add_coagulation_score(
    df: pd.DataFrame, platelets_col: str = "Platelets"
) -> pd.DataFrame:
    """
    Add a 'coagulation_score' column to df based on the Platelets column.
    """
    df["coagulation_score"] = df[platelets_col].apply(coagulation_score)
    return df


def add_liver_score(
    df: pd.DataFrame, bilirubin_col: str = "Bilirubin_total"
) -> pd.DataFrame:
    """
    Add a 'liver_score' column to df based on the Bilirubin_total column.
    """
    df["liver_score"] = df[bilirubin_col].round(1).apply(liver_score)
    return df


def add_respiratory_score(
    df: pd.DataFrame, sao2_col: str = "SaO2", fio2_col: str = "FiO2"
) -> pd.DataFrame:
    """
    Add a 'respiratory_score' column to df using SaO2 and FiO2 columns.
    """
    df["respiratory_score"] = df.apply(
        lambda row: respiratory_score(row[sao2_col], row[fio2_col]), axis=1
    )
    return df


def add_sofa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all SOFA sub-score columns to df and return the augmented DataFrame.
    """
    df = add_renal_score(df)
    df = add_coagulation_score(df)
    df = add_liver_score(df)
    df = add_respiratory_score(df)
    df["sofa_score"] = df[
        ["renal_score", "coagulation_score", "liver_score", "respiratory_score"]
    ].sum(axis=1)
    return df


# NEWS scoring functions and DataFrame helpers


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


def add_news_hr_score(df: pd.DataFrame, hr_col: str = "HR") -> pd.DataFrame:
    """
    Add 'news_hr_score' column based on heart rate.
    """
    df["news_hr_score"] = df[hr_col].apply(news_hr_score)
    return df


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


def add_news_resp_score(df: pd.DataFrame, resp_col: str = "Resp") -> pd.DataFrame:
    """
    Add 'news_resp_score' column based on respiratory rate.
    """
    df["news_resp_score"] = df[resp_col].apply(news_resp_score)
    return df


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


def add_news_temp_score(df: pd.DataFrame, temp_col: str = "Temp") -> pd.DataFrame:
    """
    Add 'news_temp_score' column based on temperature.
    """
    df["news_temp_score"] = df[temp_col].apply(news_temp_score)
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


def add_news_sbp_score(df: pd.DataFrame, sbp_col: str = "SBP") -> pd.DataFrame:
    """
    Add 'news_sbp_score' column based on systolic blood pressure.
    """
    df["news_sbp_score"] = df[sbp_col].apply(news_sbp_score)
    return df


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


def add_news_o2sat_score(df: pd.DataFrame, o2sat_col: str = "O2Sat") -> pd.DataFrame:
    """
    Add 'news_o2sat_score' column based on oxygen saturation.
    """
    df["news_o2sat_score"] = df[o2sat_col].apply(news_o2sat_score)
    return df


def news_supplemental_o2_score(fio2: float) -> int:
    """
    NEWS sub-score for supplemental oxygen. FiO2 > 0.21 indicates supplemental O₂.
    """
    if pd.isna(fio2):
        # Missing FiO2: assume no supplemental oxygen
        return 0
    return 2 if fio2 > 0.21 else 0


def add_news_supplemental_o2_score(
    df: pd.DataFrame, fio2_col: str = "FiO2"
) -> pd.DataFrame:
    """
    Add 'news_supplemental_o2_score' column based on FiO2.
    """
    df["news_supplemental_o2_score"] = df[fio2_col].apply(news_supplemental_o2_score)
    return df


def add_news_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all NEWS sub-score columns and 'news_score' total to df.
    """
    df = add_news_hr_score(df)
    df = add_news_resp_score(df)
    df = add_news_temp_score(df)
    df = add_news_sbp_score(df)
    df = add_news_o2sat_score(df)
    df = add_news_supplemental_o2_score(df)
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
