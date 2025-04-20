from pathlib import Path

import numpy as np
import pandas as pd


def remove_highly_correlated_features_global(df, threshold=0.85):
    corr_matrix = df.corr().abs()
    upper = np.triu(corr_matrix, k=1)
    upper_df = pd.DataFrame(
        upper, columns=corr_matrix.columns, index=corr_matrix.columns
    )

    to_drop = set()

    for col in upper_df.columns:
        high_corr = upper_df[col][upper_df[col] > threshold]
        to_drop.update(high_corr.index.tolist())

    report = []
    for col in df.columns:
        max_corr = corr_matrix[col].drop(col).max()
        report.append({"feature": col, "max_corr": max_corr, "dropped": col in to_drop})

    df_filtered = df.drop(columns=to_drop)
    return df_filtered, list(to_drop), report, corr_matrix


def filter_balanced_dataset_main():
    input_path = Path("dataset/feature_engineering/balanced_dataset.parquet")
    output_path = Path(
        "dataset/feature_engineering/balanced_dataset_filtered.parquet")
    report_path = Path("dataset/feature_engineering/correlation_report.csv")

    df = pd.read_parquet(input_path)
    if "patient_id" not in df.columns and "patient_id" in df.index.names:
        df = df.reset_index()
    drop_cols = ["Unit1", "Unit2", "HospAdmTime", "cluster_id", "dataset"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    non_feature_cols = ["patient_id", "ICULOS", "SepsisLabel"]
    all_cols = df.columns.tolist()
    physio_cols = [col for col in all_cols if col not in non_feature_cols]

    df_physio = df[physio_cols]
    df_static = df[non_feature_cols]
    df_physio_filtered, dropped, corr_report, corr_matrix = (
        remove_highly_correlated_features_global(df_physio, threshold=0.85)
    )

    df_filtered = pd.concat([df_static, df_physio_filtered], axis=1)
    df_filtered.to_parquet(output_path, index=False)

    df_report = pd.DataFrame(corr_report)
    df_report.to_csv(report_path, index=False)

    print(f"Filtered dataset saved to: {output_path}")
    print(f"Correlation report saved to: {report_path}")
    print(f"Dropped features: {dropped}")


if __name__ == "__main__":
    filter_balanced_dataset_main()
