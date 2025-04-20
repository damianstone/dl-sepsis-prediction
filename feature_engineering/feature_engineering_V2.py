from pathlib import Path

import numpy as np
import pandas as pd

# continues features
con_col = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2']


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


def score_by_value(value, thresholds):
    for threshold, score in thresholds:
        if value >= threshold:
            return score
    return 0


def calculate_sofa(row):
    sofa = 0
    if row.get("MAP", 100) < 70:
        sofa += 1
    sofa += score_by_value(
        row.get("Creatinine", 0), [(5, 4), (3.5, 3), (2, 2), (1.2, 1)]
    )
    # TODO: this calculation WRONG, when its above 150 should return 0, double check for other calculations
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


def calculate_particupar_scores(df):
    """
    calculate the scores for each measurement value as save it as a new feature
    for example:
      HR_SOFA
      HR_NEWS
      HR_qSOFA
      O2Sat_SOFA
      O2Sat_NEWS
      O2Sat_qSOFA

    Breaking down scores by individual measurements (HR_SOFA, HR_NEWS) helps the model understand 
    how each vital sign contributes to overall risk, enabling better pattern recognition across 
    time series data.
    """
    df = df.copy()
    pass


def calculate_general_scores(df):
    """
    Calculate general scores (SOFA, NEWS, qSOFA) for each patient.
    example new column for each patient:
      SOFA
      NEWS
      qSOFA
    """
    df = df.copy()
    df['SOFA'] = df.apply(calculate_sofa, axis=1)
    df['NEWS'] = df.apply(calculate_news, axis=1)
    df['qSOFA'] = df.apply(calculate_qsofa, axis=1)
    return df


def aggregate_window_features(df, cols, suffix):
    """
    This function analyzes vital signs over 6-hour windows, calculating statistics 
    (mean, min, max, std) and missing data patterns. It helps detect short-term physiological 
    changes and data quality issues that might indicate sepsis onset.
    """
    stats = {}
    for col in cols:
        series = df[col].dropna()
        stats[f"{col}_mean_{suffix}"] = series.mean()
        stats[f"{col}_std_{suffix}"] = series.std()
        stats[f"{col}_min_{suffix}"] = series.min()
        stats[f"{col}_max_{suffix}"] = series.max()
        stats[f"{col}_last_{suffix}"] = series.iloc[-1] if not series.empty else np.nan
        # TODO: include median
        # TODO: include standard deviation

        is_missing = df[col].isna()
        stats[f"{col}_missing_count_{suffix}"] = is_missing.sum()
        stats[f"{col}_missing_rate_{suffix}"] = is_missing.mean()
    return stats


def aggregate_scores(df, suffix="global"):
    """
    calculate statistical features (mean, max, last value) from clinical scores over time windows. 
    This helps capture disease progression patterns and temporal trends, 
    which are crucial for predicting sepsis development.
    """
    # NOTE: sofa, qsofa and news will be already at this point
    return {
        f"SOFA_mean_{suffix}": sofa_scores.mean(),
        f"SOFA_max_{suffix}": sofa_scores.max(),
        f"SOFA_last_{suffix}": (
            sofa_scores.iloc[-1] if not sofa_scores.empty else np.nan
        ),
        # TODO: include median
        f"NEWS_mean_{suffix}": news_scores.mean(),
        f"NEWS_max_{suffix}": news_scores.max(),
        f"NEWS_last_{suffix}": (
            news_scores.iloc[-1] if not news_scores.empty else np.nan
        ),
        # TODO: include median
        f"qSOFA_mean_{suffix}": qsofa_scores.mean(),
        f"qSOFA_max_{suffix}": qsofa_scores.max(),
        f"qSOFA_last_{suffix}": (
            qsofa_scores.iloc[-1] if not qsofa_scores.empty else np.nan
        ),
        # TODO: include median
    }


def generate_window_features(df, cols):
    """
    This function generates statistical features (mean, max, last value) from vital signs over 6-hour windows. 
    It helps capture short-term physiological changes and data quality issues that might indicate sepsis onset.

    This will be applied for each patient for the following columns: ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp']

    example new columns for each patient:
      HR_mean_6h
      HR_std_6h
      HR_min_6h
      HR_max_6h
      HR_last_6h
      HR_missing_count_6h
      HR_missing_rate_6h
    """
    df = df.copy()
    window_features = {}
    pass


def generate_missingness_features(raw_df, imputed_df, df_with_features):
    """
    This function generates missingness features for each patient.
    """
    df = df.copy()
    pass


def print_new_columns(df_features, imputed_df):
    new_columns = set(df_features.columns) - set(imputed_df.columns)
    print(f"New columns added: {new_columns}")


def preprocess_data(raw_file, imputed_file, output_file):
    raw_df = pd.read_parquet(raw_file)
    imputed_df = pd.read_parquet(imputed_file)
    df_features = imputed_df.copy()

    # 1: AIDEN
    # particular scores columns
    df_features = calculate_particupar_scores(df_features)

    # 2: DAMIAN
    # get general sofa, qsofa and news scrores
    df_features = calculate_general_scores(df_features)

    # 3: ZHOU
    # six-hour slide window statistics of selected columns
    columns = ['HR', 'O2Sat', 'SBP', 'MAP', 'Resp']
    df_features = generate_window_features(df_features, columns)

    # 4: ZHOU
    # missingness features
    df_features = generate_missingness_features(raw_df, imputed_df, df_features)

    # 5: DON'T CARE
    # drop useless columns
    df_features = df_features.drop(columns=["Unit1", "Unit2", "cluster_id", "dataset"])

    # 6: DON'T CARE
    # handle gender as a categorical variable
    df_features['Gender'] = LabelEncoder().fit_transform(
        df_features['Gender'].astype(str))

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
    INPUT_DATASET = f"{root}/dataset/Fully_imputed_dataset.parquet"
    OUTPUT_DATASET = f"{root}/dataset/V2_preprocessed.parquet"
    preprocess_data(OUTPUT_DATASET)
