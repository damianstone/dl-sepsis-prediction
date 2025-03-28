from pathlib import Path
import pandas as pd
import numpy as np

def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError("Project root not found")

def create_balanced_dataset(input_file="features_with_missing_interval.parquet", output_file="balanced_dataset.parquet"):
    root = find_project_root()
    data_path = root / "dataset" / "XGBoost" / "feature_engineering" / input_file
    df = pd.read_parquet(data_path)

    if not isinstance(df.index, pd.MultiIndex):
        df.set_index(["patient_id", "ICULOS"], inplace=True)

    grouped = df.reset_index().groupby("patient_id")["SepsisLabel"]
    patient_is_positive = grouped.max()

    positive_ids = patient_is_positive[patient_is_positive == 1].index
    negative_ids = patient_is_positive[patient_is_positive == 0].index

    print(f"Number of positive patients: {len(positive_ids)}")
    print(f"Number of negative patients: {len(negative_ids)}")

    np.random.seed(1)
    sampled_negative_ids = np.random.choice(negative_ids, size=len(positive_ids)*4, replace=False)
    keep_ids = set(positive_ids).union(set(sampled_negative_ids))
    df_balanced = df.loc[df.index.get_level_values("patient_id").isin(keep_ids)]

    out_path = root / "dataset" / "XGBoost" / "feature_engineering" / output_file
    df_balanced.to_parquet(out_path)
    print(f"save to: {out_path}")
    

    final_patient_ids = df_balanced.reset_index()["patient_id"].unique()
    final_label_map = df_balanced.reset_index().groupby("patient_id")["SepsisLabel"].max()

    final_positive = (final_label_map == 1).sum()
    final_negative = (final_label_map == 0).sum()

    print(f"retained positive patients: {final_positive}")
    print(f"retained negative patients: {final_negative}")
    print(f"Positive ratio: {final_positive / (final_positive + final_negative):.2%}")



if __name__ == "__main__":
    create_balanced_dataset()
