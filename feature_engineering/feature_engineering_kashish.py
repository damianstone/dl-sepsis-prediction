
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


root = find_project_root()
INPUT_DATASET = f"{root}/dataset/Fully_imputed_dataset.parquet"
OUTPUT_DATASET = f"{root}/dataset/preprocessed_data.parquet"

df = pd.read_parquet(INPUT_DATASET)
max_length = df.groupby("patient_id").size().max()


def calculate_sofa(row):
    sofa = 0

    def assign_score(value, thresholds):
        for threshold, score in thresholds:
            if value >= threshold:
                return score
        return 0

    # Respiration
    if row.get('FiO2', 0) > 0:
        pao2_fio2 = row.get('SaO2', 0) / row['FiO2']
        sofa += assign_score(pao2_fio2, [(100, 4), (200, 3), (300, 2), (400, 1)])

    # Coagulation
    sofa += assign_score(row.get('Platelets', float('inf')),
                         [(20, 4), (50, 3), (100, 2), (150, 1)])

    # Liver Function
    sofa += assign_score(row.get('Bilirubin_total', 0),
                         [(12, 4), (6, 3), (2, 2), (1.2, 1)])

    # Cardiovascular
    if row.get('MAP', 100) < 70:
        sofa += 1

    # Renal Function
    sofa += assign_score(row.get('Creatinine', 0), [(5, 4), (3.5, 3), (2, 2), (1.2, 1)])

    return sofa


def calculate_news(row):
    news = 0

    def assign_news_score(value, thresholds):
        for threshold, score in thresholds:
            if value >= threshold:
                return score
        return 0

    # HR (Heart Rate)
    news += assign_news_score(row.get('HR', 0),
                              [(40, 3), (50, 1), (90, 0), (110, 1), (130, 2), (131, 3)])

    # Respiration Rate
    news += assign_news_score(row.get('Resp', 0),
                              [(8, 3), (9, 1), (11, 0), (21, 2), (24, 3)])

    # Temperature
    news += assign_news_score(row.get('Temp', 0),
                              [(35, 3), (36, 1), (38, 1), (39.1, 2)])

    # SBP (Systolic BP) or MAP (Mean Arterial Pressure)
    sbp = row.get('SBP', row.get('MAP', 100))
    news += assign_news_score(sbp, [(90, 3), (100, 2), (110, 1)])

    # O2 Saturation
    news += assign_news_score(row.get('O2Sat', 0), [(85, 3), (91, 2), (93, 1)])

    # Supplemental Oxygen (if available)
    if row.get('FiO2', 0) > 0.21:
        news += 2

    return news


def calculate_qsofa(row):
    qsofa = 0

    # SBP ≤ 100 mmHg
    if row.get('SBP', 120) <= 100:
        qsofa += 1

    # Respiration Rate ≥ 22
    if row.get('Resp', 0) >= 22:
        qsofa += 1

    return qsofa


def num_recorded_values(row):
    recorded_measurements = df.notnull().sum()

    return recorded_measurements


def missingness_feature(row):

    if 'ICULOS' and 'ICULOS' in df.columns:
        df = df.sort_values(by='ICULOS')
        time_intervals = df['ICULOS'].diff()

    return time_intervals


def add_temporal_features(df):
    # Adds rolling statistics (moving averages, standard deviation, rate of change) for some features (may or may not be useful).
    time_window_sizes = [3, 6, 12]  # Rolling window sizes (in time steps)
    feature_cols = ['HeartRate', 'RespiratoryRate',
                    'MAP', 'SpO2', 'Creatinine', 'Platelets']

    df.sort_values(['patient_id', 'ICULOS'], inplace=True)

    for col in feature_cols:
        if col in df.columns:
            for window in time_window_sizes:
                df[f'{col}_MA_{window}h'] = df.groupby('patient_id')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean())
                df[f'{col}_SD_{window}h'] = df.groupby('patient_id')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std())
                df[f'{col}_Delta'] = df.groupby('patient_id')[col].diff()
    return df


def preprocess_data(output_file):
    global df

    df['SOFA'] = df.apply(calculate_sofa, axis=1)
    df['NEWS'] = df.apply(calculate_news, axis=1)
    df['qSOFA'] = df.apply(calculate_qsofa, axis=1)
    # df['num_recorded_values'] = df.apply(num_recorded_values, axis=1)
    # df['missingness_feature'] = df.apply(missingness_feature, axis=1)
    df = add_temporal_features(df)

    if 'Gender' in df.columns:
        df['Gender'] = LabelEncoder().fit_transform(df['Gender'].astype(str))

    feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # drop columns that are not needed
    columns_to_drop = ["Unit1", "Unit2", "cluster_id", "dataset"]
    df = df.drop(columns=columns_to_drop)

    # NOTE: handle nan values doing forward fill and then back fill
    df = df.fillna(method="ffill")
    df = df.fillna(method="bfill")

    df.to_parquet(output_file, index=False)

    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    preprocess_data(OUTPUT_DATASET)
