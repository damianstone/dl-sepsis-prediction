from pathlib import Path

import numpy as np
import pandas as pd
from medical_scoring import (
    add_bilirubin_ratio,
    add_news_scores,
    add_qsofa_score,
    add_shock_index,
    add_sofa_scores,
)
from sklearn.preprocessing import LabelEncoder

# continues features
con_col = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]


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
        f"Project root marker '{marker}' not found starting from {current}"
    )


def calculate_scores(df):
    """
    calculate the scores for each measurement value as save it as a new feature
    for example:
      HR_SOFA
      HR_NEWS
      HR_qSOFA
      O2Sat_SOFA
      O2Sat_NEWS
      O2Sat_qSOFA

    it also adds the general scores for each patient:
      SOFA
      NEWS
      qSOFA

    Breaking down scores by individual measurements (HR_SOFA, HR_NEWS) helps the model understand
    how each vital sign contributes to overall risk, enabling better pattern recognition across
    time series data.
    """
    df = df.copy()
    # go to each of these functions to know how they work
    df = add_sofa_scores(df)
    df = add_news_scores(df)
    df = add_qsofa_score(df)
    df = add_shock_index(df)
    df = add_bilirubin_ratio(df)
    return df


def aggregate_global_score_features(df, suffix="global"):
    """
    calculate statistical features (mean, max, last value) from clinical scores in each time step
    This helps capture disease progression patterns and temporal trends,
    which are crucial for predicting sepsis development.

    Example output for one patient:
    'SOFA_mean_global': 3.5,    # Average SOFA during stay
    'SOFA_median_global': 3.0,   # Median SOFA
    'SOFA_max_global': 6.0,      # Highest SOFA
    'SOFA_last_global': 4.0      # Final SOFA
    """
    scores = {}
    for score_name in ["SOFA_score", "NEWS_score", "qSOFA_score"]:
        series = df[score_name].dropna()
        scores[f"{score_name}_mean_{suffix}"] = series.mean()
        scores[f"{score_name}_median_{suffix}"] = series.median()
        scores[f"{score_name}_max_{suffix}"] = series.max()
        scores[f"{score_name}_last_{suffix}"] = (
            series.iloc[-1] if not series.empty else np.nan
        )
    return scores


def aggregate_window_features(df, cols, suffix):
    """
    THIS MOSTLY FOR THE XGBOOST
    This function analyzes vital signs over 6-hour windows, calculating statistics
    (mean, min, max, std) and missing data patterns. It helps detect short-term physiological
    changes and data quality issues that might indicate sepsis onset.

    For transformers, global scores are less crucial since transformers can learn
    temporal patterns directly from sequential data. However, they can still be useful as
    auxiliary features to provide high-level summaries of patient state. Consider them optional.
    """
    stats = {}
    for col in cols:
        series = df[col].dropna()
        stats[f"{col}_mean_{suffix}"] = series.mean()
        stats[f"{col}_median_{suffix}"] = series.median()
        stats[f"{col}_std_{suffix}"] = series.std()
        stats[f"{col}_min_{suffix}"] = series.min()
        stats[f"{col}_max_{suffix}"] = series.max()
        stats[f"{col}_last_{suffix}"] = series.iloc[-1] if not series.empty else np.nan
    return stats


def generate_window_features(df, cols):
    """
    This function generates statistical features (mean, max, last value) from vital signs over 6-hour windows.
    It helps capture short-term physiological changes and data quality issues that might indicate sepsis onset.

    This will be applied for each patient for the following columns: ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp']

    example new columns for each patient:
      HR_mean_6h
      HR_median_6h
      HR_std_6h
      HR_min_6h
      HR_max_6h
      HR_last_6h
      HR_missing_count_6h
      HR_missing_rate_6h

    For each patient:

    1. Sort rows by ICU stay time (`ICULOS`).
    2. For each row, look back at the last 6 rows (6-hour window).
    3. If there are 6 or more rows: compute stats (mean, std, min, max, last, missing count, missing rate).
    4. If fewer than 6 rows: fill all stats with zero.
    5. Save the stats into the current row.
    6. After all patients are processed, return the full dataset with new features.
    """
    df = df.copy()
    all_rows = []

    for pid, group in df.groupby("patient_id"):
        scores = aggregate_global_score_features(group, suffix="global")
        for k, v in scores.items():
            group[k] = v

        group = group.sort_values("ICULOS").copy()
        for i in range(len(group)):
            if i >= 6:
                window = group.iloc[i - 6 : i]
                stats = aggregate_window_features(window, cols, suffix="6h")
            else:
                stats = {
                    f"{col}_{stat}_6h": 0
                    for col in cols
                    for stat in [
                        "mean",
                        "median",
                        "std",
                        "min",
                        "max",
                        "last",
                    ]
                }
            for k, v in stats.items():
                group.loc[group.index[i], k] = v
        all_rows.append(group)

    return pd.concat(all_rows).reset_index(drop=True)


def compute_missingness_summary(df, cols):
    summary = {}
    for col in cols:
        series = df[col]
        is_missing = series.isna()
        summary[f"{col}_missing_count_global"] = is_missing.sum()

        last_seen = None
        intervals = []
        for i, val in enumerate(series):
            if pd.notna(val):
                last_seen = i
            elif last_seen is not None:
                intervals.append(i - last_seen)
            else:
                intervals.append(-1)
        valid_intervals = [v for v in intervals if v >= 0]
        summary[f"{col}_missing_interval_mean_global"] = (
            np.mean(valid_intervals) if valid_intervals else -1
        )
    return summary


def generate_missingness_features(raw_df, imputed_df, df_with_features):
    """
    The frequency of missing values reflects how often a measurement is taken.
    If a patient is generally stable, clinicians may shift their attention to other, more critical issues
    such as rapidly deteriorating patients or specific key indicators. Therefore, a low rate of missingness
    (i.e., frequent measurement) suggests that the patient or that specific variable is receiving more clinical attention.
    """
    raw_df = raw_df.copy()
    imputed_df = imputed_df.copy()
    df_with_features = df_with_features.copy()
    selected_cols = ["HR", "O2Sat", "SBP", "MAP", "Resp"]
    missing_rows = []

    for pid, group in raw_df.groupby("patient_id"):
        summary = compute_missingness_summary(group, selected_cols)
        summary["patient_id"] = pid
        missing_rows.append(summary)

    df_missing = pd.DataFrame(missing_rows)
    df_merged = pd.merge(df_with_features, df_missing, on="patient_id", how="left")
    return df_merged


def print_new_columns(df_features, imputed_df):
    new_columns = set(df_features.columns) - set(imputed_df.columns)
    print(f"New columns added: {new_columns}")


def preprocess_data(raw_file, imputed_file, output_file):
    raw_df = pd.read_parquet(raw_file)
    imputed_df = pd.read_parquet(imputed_file)
    df_features = imputed_df.copy()

    # 1: AIDEN
    # Add medical scoring features including: SOFA, NEWS, qSOFA and component scores
    df_features = calculate_scores(df_features)
    print("CALCULATE SCORES DONE")

    # 3: ZHOU
    # six-hour slide window statistics of selected columns
    columns = ["HR", "O2Sat", "SBP", "MAP", "Resp"]
    df_features = generate_window_features(df_features, columns)
    print("GENERATE WINDOW FEATURES DONE")
    # 4: ZHOU
    # missingness features
    df_features = generate_missingness_features(raw_df, imputed_df, df_features)
    print("GENERATE MISSINGNESS FEATURES DONE")
    # 5: DON'T CARE
    # drop useless columns
    df_features = df_features.drop(columns=["Unit1", "Unit2", "cluster_id", "dataset"])
    print("DROP USELESS COLUMNS DONE")
    # 6: DON'T CARE
    # handle gender as a categorical variable
    df_features["Gender"] = LabelEncoder().fit_transform(
        df_features["Gender"].astype(str)
    )
    print("HANDLE GENDER DONE")

    # 7: DON'T CARE
    # scale the features - min max scaler ? discuss this later

    # NOTE: before save the data, print the following checks:
    # print a check for nan values (should not be any)
    if df_features.isna().sum().sum() > 0:
        # save the dataset with nan values for debugging
        df_features.to_parquet(f"{output_file}_with_nans.parquet")
        raise ValueError("Found NaN values in the dataset")
        # NOTE: handle nan values doing forward fill and then back fill
        # df_features = df_features.fillna(method="ffill")
        # df_features = df_features.fillna(method="bfill")
    else:
        print("No NaN values found in the dataset")
    # print the new columns names added after feature engineering comparing with the imputed_df
    print_new_columns(df_features, imputed_df)
    # print the total number of features (columns)
    print(f"Total number of features: {len(df_features.columns)}")
    # save the data
    df_features.to_parquet(output_file)
    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    root = find_project_root()
    RAW_DATASET = f"{root}/dataset/raw_combined_data.parquet"
    IMPUTED_DATASET = f"{root}/dataset/Fully_imputed_dataset.parquet"
    OUTPUT_DATASET = f"{root}/dataset/V2_preprocessed.parquet"
    preprocess_data(RAW_DATASET, IMPUTED_DATASET, OUTPUT_DATASET)
