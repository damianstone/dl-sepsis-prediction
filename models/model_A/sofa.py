import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# Input/output
DATA_DIR_A = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\dataset\training_setA"
DATA_DIR_B = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\dataset\training_setB"
OUTPUT_PATH = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\dataset\raw_combined_sofa24.parquet"

# Core physiological features
PHYSIO_FEATURES = ['HR', 'O2Sat', 'MAP', 'Creatinine', 'Platelets']

# Load all .psv files
def load_all_psv(folders):
    all_data = []
    pid = 0
    for folder in folders:
        files = [f for f in os.listdir(folder) if f.endswith(".psv")]
        for f in tqdm(files, desc=f"Loading {os.path.basename(folder)}"):
            df = pd.read_csv(os.path.join(folder, f), sep='|')
            df["patient_id"] = pid
            pid += 1
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Drop unused columns
def remove_columns(df):
    return df.drop(columns=[c for c in ["Unit1", "Unit2"] if c in df.columns])

# Keep patients with at least 24 rows
def filter_24hr_patients(df):
    counts = df.groupby("patient_id").size()
    valid_ids = counts[counts >= 24].index
    return df[df["patient_id"].isin(valid_ids)].copy()

# Compute SOFA score and components
def calculate_sofa(df):
    df = df.copy()
    df["SOFA_resp"] = pd.cut(df["FiO2"], [-np.inf, 0.21, 0.3, 0.4, 0.5, np.inf], labels=[0, 1, 2, 3, 4]).astype(float)
    df["SOFA_coag"] = pd.cut(df["Platelets"], [-np.inf, 20, 50, 100, 150, np.inf], labels=[4, 3, 2, 1, 0]).astype(float)
    df["SOFA_liver"] = pd.cut(df["Bilirubin_total"], [-np.inf, 1.2, 2, 6, 12, np.inf], labels=[0, 1, 2, 3, 4]).astype(float)
    df["SOFA_renal"] = pd.cut(df["Creatinine"], [-np.inf, 1.2, 2, 3.5, 5, np.inf], labels=[0, 1, 2, 3, 4]).astype(float)
    df["SOFA_cardio"] = (df["MAP"] < 70).astype(float)
    df["SOFA"] = df[["SOFA_resp", "SOFA_coag", "SOFA_liver", "SOFA_renal", "SOFA_cardio"]].sum(axis=1, skipna=True)
    return df

# Add diff for features and SOFA
def add_diff_features(df, features, group_col="patient_id", deltas=[1,3,5]):
    for feat in features:
        for d in deltas:
            col = f"{feat}_diff_{d}"
            df[col] = df.groupby(group_col)[feat].diff(periods=d)
    return df

# Add informative missingness mask
def add_missingness_mask(df):
    for col in df.columns:
        if col in ['SepsisLabel', 'patient_id']:
            continue
        df[f"{col}_is_missing"] = df[col].isna().astype(int)
    return df


def preprocess_all():
    df = load_all_psv([DATA_DIR_A, DATA_DIR_B])
    df = remove_columns(df)
    df = filter_24hr_patients(df)
    df = calculate_sofa(df)
    df = add_diff_features(df, PHYSIO_FEATURES + ['SOFA'])
    df = add_missingness_mask(df)
    df.to_parquet(OUTPUT_PATH, index=False)

    n_patients = df['patient_id'].nunique()
    print(f"\nSaved to {OUTPUT_PATH}")
    print(f"Remaining patients with ≥24 time points: {n_patients}")

if __name__ == "__main__":
    preprocess_all()
