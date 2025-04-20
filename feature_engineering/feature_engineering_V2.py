from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


def find_project_root(marker=".gitignore"):
    """
    walk up from the current working directory until a directory containing the
    specified marker (e.g., .gitignore) is found.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}")


def score_by_value(value, thresholds):
    for threshold, score in thresholds:
        if value >= threshold:
            return score
    return 0


def calculate_sofa(df, row):
    sofa = 0
    if row.get("MAP", 100) < 70:
        sofa += 1
    sofa += score_by_value(
        row.get("Creatinine", 0), [(5, 4), (3.5, 3), (2, 2), (1.2, 1)]
    )
    sofa += score_by_value(
        row.get("Platelets", float("inf")), [(20, 4), (50, 3), (100, 2), (150, 1)]
    )
    sofa += score_by_value(
        row.get("Bilirubin_total", 0), [(12, 4), (6, 3), (2, 2), (1.2, 1)]
    )
    if row.get("FiO2", 0) > 0 and pd.notna(row.get("SaO2", None)):
        pao2_fio2 = row["SaO2"] / row["FiO2"]
        sofa += score_by_value(pao2_fio2, [(100, 4), (200, 3), (300, 2), (400, 1)])
    return sofa


def calculate_news(row):
    news = 0
    news += score_by_value(
        row.get("HR", 0), [(131, 3), (130, 2), (110, 1), (90, 0), (50, 1), (40, 3)]
    )
    news += score_by_value(
        row.get("Resp", 0), [(24, 3), (21, 2), (11, 0), (9, 1), (8, 3)]
    )
    news += score_by_value(row.get("Temp", 0), [(39.1, 2), (38, 1), (36, 1), (35, 3)])
    sbp = row.get("SBP", row.get("MAP", 100))
    news += score_by_value(sbp, [(90, 3), (100, 2), (110, 1)])
    news += score_by_value(row.get("O2Sat", 0), [(85, 3), (91, 2), (93, 1)])
    if row.get("FiO2", 0) > 0.21:
        news += 2
    return news


def calculate_qsofa(row):
    qsofa = 0
    if row.get("SBP", 120) <= 100:
        qsofa += 1
    if row.get("Resp", 0) >= 22:
        qsofa += 1
    return qsofa


def aggregate_window_features(df, cols, suffix):
    stats = {}
    for col in cols:
        series = df[col].dropna()
        stats[f"{col}_mean_{suffix}"] = series.mean()
        stats[f"{col}_std_{suffix}"] = series.std()
        stats[f"{col}_min_{suffix}"] = series.min()
        stats[f"{col}_max_{suffix}"] = series.max()
        stats[f"{col}_last_{suffix}"] = series.iloc[-1] if not series.empty else np.nan

        is_missing = df[col].isna()
        stats[f"{col}_missing_count_{suffix}"] = is_missing.sum()
        stats[f"{col}_missing_rate_{suffix}"] = is_missing.mean()
    return stats


def aggregate_scores(df, suffix="global"):
    sofa_scores = df.apply(calculate_sofa, axis=1)
    news_scores = df.apply(calculate_news, axis=1)
    qsofa_scores = df.apply(calculate_qsofa, axis=1)

    return {
        f"SOFA_mean_{suffix}": sofa_scores.mean(),
        f"SOFA_max_{suffix}": sofa_scores.max(),
        f"SOFA_last_{suffix}": (
            sofa_scores.iloc[-1] if not sofa_scores.empty else np.nan
        ),
        f"NEWS_mean_{suffix}": news_scores.mean(),
        f"NEWS_max_{suffix}": news_scores.max(),
        f"NEWS_last_{suffix}": (
            news_scores.iloc[-1] if not news_scores.empty else np.nan
        ),
        f"qSOFA_mean_{suffix}": qsofa_scores.mean(),
        f"qSOFA_max_{suffix}": qsofa_scores.max(),
        f"qSOFA_last_{suffix}": (
            qsofa_scores.iloc[-1] if not qsofa_scores.empty else np.nan
        ),
    }


def preprocess_data(input_file, output_file):
    # make functions for the following:
    # sofa
    # qsofa
    # news
    # agregate window features
    # aggregate_scores
    # generate_multiwindow_features (only 6h)
    # drop columns = ["Unit1", "Unit2", "cluster_id", "dataset"]

    # check for nan values (should not be any)
    # print the new columns names added after feature engineering
    # print the total number of features (columns)
    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    root = find_project_root()
    INPUT_DATASET = f"{root}/dataset/Fully_imputed_dataset.parquet"
    OUTPUT_DATASET = f"{root}/dataset/V2_preprocessed.parquet"
    preprocess_data(OUTPUT_DATASET)
