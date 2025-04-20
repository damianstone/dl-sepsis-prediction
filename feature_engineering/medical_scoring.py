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
