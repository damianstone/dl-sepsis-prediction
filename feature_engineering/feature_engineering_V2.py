from pathlib import Path

import numpy as np
import pandas as pd
from medical_scoring import add_medical_scores
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
    df_out = []
    for pid, group in df.groupby("patient_id"):
        group = group.sort_values("ICULOS").copy()
        group_features = group.copy()
        for i in range(len(group)):
            if i >= 6:
                window = group.iloc[i - 6:i]
                stats = aggregate_window_features(window, cols, suffix="6h")
            else:
                stats = {
                    f"{col}_{stat}_6h": 0
                    for col in cols
                    for stat in ["mean", "std", "min", "max", "last"]
                }
            for k, v in stats.items():
                group_features.at[group.index[i], k] = v
        df_out.append(group_features)
    return pd.concat(df_out)



def generate_missingness_features(raw_df, imputed_df, df_features):
    
    #missing count、missing rate
    
    selected_cols = ["HR", "O2Sat", "SBP", "MAP", "Resp"]
    missingness_rows = []

    for pid, group in raw_df.groupby("patient_id"):
        row = {"patient_id": pid}
        for col in selected_cols:
            is_missing = group[col].isna()
            row[f"{col}_missing_count"] = is_missing.sum()
            row[f"{col}_missing_rate"] = is_missing.mean()
            
            missing_indices = group[is_missing].index.to_list()
            if len(missing_indices) >= 2:
                gaps = np.diff(missing_indices)
                row[f"{col}_missing_interval_avg"] = np.mean(gaps)
            else:
                row[f"{col}_missing_interval_avg"] = 0
        missingness_rows.append(row)

    df_missing = pd.DataFrame(missingness_rows)
    df_features = pd.merge(df_features, df_missing, on="patient_id", how="left")
    return df_features


def print_new_columns(df_features, imputed_df):
    new_columns = set(df_features.columns) - set(imputed_df.columns)
    print(f"New columns added: {new_columns}")



def preprocess_data(raw_file, imputed_file, output_file):
    raw_df = pd.read_parquet(raw_file)
    imputed_df = pd.read_parquet(imputed_file)
    df_features = imputed_df.copy()

    # 1: AIDEN
    # Add medical scoring features including: SOFA, NEWS, qSOFA and component scores
    df_features = add_medical_scores(df_features)

    # 3: ZHOU
    # six-hour slide window statistics of selected columns
    columns = ["HR", "O2Sat", "SBP", "MAP", "Resp"]
    df_features = generate_window_features(df_features, columns)

    # 4: ZHOU
    # missingness features
    df_features = generate_missingness_features(raw_df, imputed_df, df_features)

    # 5: DON'T CARE
    # drop useless columns
    df_features = df_features.drop(columns=["Unit1", "Unit2", "cluster_id", "dataset"])

    # 6: DON'T CARE
    # handle gender as a categorical variable
    df_features["Gender"] = LabelEncoder().fit_transform(
        df_features["Gender"].astype(str)
    )

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
