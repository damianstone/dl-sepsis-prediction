from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError("Project root not found")


def create_balanced_dataset(
    input_file="raw_combined_data.parquet", output_file="balanced_dataset.parquet"
):
    root = find_project_root()
    data_path = root / "dataset" / input_file
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
    sampled_negative_ids = np.random.choice(
        negative_ids, size=int(len(positive_ids) * (70 / 30)), replace=False
    )
    keep_ids = set(positive_ids).union(set(sampled_negative_ids))
    df_balanced = df.loc[df.index.get_level_values("patient_id").isin(keep_ids)]

    out_path = root / "dataset" / "feature_engineering" / output_file
    df_balanced.to_parquet(out_path)
    print(f"save to: {out_path}")

    final_label_map = (
        df_balanced.reset_index().groupby("patient_id")["SepsisLabel"].max())

    final_positive = (final_label_map == 1).sum()
    final_negative = (final_label_map == 0).sum()

    print(f"retained positive patients: {final_positive}")
    print(f"retained negative patients: {final_negative}")
    print(f"Positive ratio: {final_positive / (final_positive + final_negative)}")


def split_balanced_dataset_maintain_ratio(
    input_file="after_feature_engineering.parquet"):
    root = find_project_root()
    data_path = root / "dataset" / input_file
    df_balanced = pd.read_parquet(data_path)

    df_reset = df_balanced.reset_index()
    label_col = (
        "SepsisLabel_patient"
        if "SepsisLabel_patient" in df_reset.columns
        else "SepsisLabel"
    )
    patient_labels = df_reset.groupby("patient_id")[label_col].max()

    pos_ids = patient_labels[patient_labels == 1].index.to_numpy()
    neg_ids = patient_labels[patient_labels == 0].index.to_numpy()

    np.random.seed(0)
    np.random.shuffle(pos_ids)
    np.random.shuffle(neg_ids)

    total_train = int(0.8 * (len(pos_ids) + len(neg_ids)))
    total_test = (len(pos_ids) + len(neg_ids)) - total_train
    train_pos = int(total_train * 0.3)
    test_pos = int(total_test * 0.3)

    train_neg = total_train - train_pos
    test_neg = total_test - test_pos

    selected_train_pos = pos_ids[:train_pos]
    selected_test_pos = pos_ids[train_pos : train_pos + test_pos]
    selected_train_neg = neg_ids[:train_neg]
    selected_test_neg = neg_ids[train_neg : train_neg + test_neg]
    train_ids = set(selected_train_pos).union(set(selected_train_neg))
    test_ids = set(selected_test_pos).union(set(selected_test_neg))

    df_train = df_balanced[df_balanced["patient_id"].isin(train_ids)]
    df_test = df_balanced[df_balanced["patient_id"].isin(test_ids)]

    out_dir = root / "dataset" / "XGBoost"
    df_train.to_parquet(out_dir / "train_balanced.parquet")
    df_test.to_parquet(out_dir / "test_balanced.parquet")

    print(
        f"Train: {len(selected_train_pos)} positive, {len(selected_train_neg)} negative patients"
    )
    print(
        f"Test: {len(selected_test_pos)} positive, {len(selected_test_neg)} negative patients"
    )
    print(f"Train saved to: {out_dir / 'train_balanced.parquet'}")
    print(f"Test saved to: {out_dir / 'test_balanced.parquet'}")


if __name__ == "__main__":
    create_balanced_dataset()
    split_balanced_dataset_maintain_ratio()
