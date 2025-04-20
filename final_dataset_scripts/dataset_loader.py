from pathlib import Path

import pandas as pd

types = ["no_sampling", "oversampled", "undersampled"]


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


def load_train_data(type, fraction: float = 1.0) -> dict:
    """
    Loads the final dataset for a given type.

    Args:
        type: Type of dataset to load (no_sampling, oversampled, undersampled)
        fraction: Fraction of patients to sample (default: 1.0 = all patients)
    """
    root = find_project_root()
    if type not in types:
        raise ValueError(f"Invalid dataset type: {type}. Must be one of {types}")
    try:
        train_df = pd.read_parquet(
            f"{root}/dataset/final_datasets/{type}_train.parquet"
        )
    except Exception:
        raise FileNotFoundError(f"Dataset for type {type} not found.")

    if fraction < 1.0:
        # Group by patient_id and get the max SepsisLabel for each patient
        patient_labels = (
            train_df.groupby("patient_id")["SepsisLabel"].max().reset_index()
        )

        # Stratified sampling of patient IDs based on their SepsisLabel
        sampled_patients = patient_labels.groupby(
            "SepsisLabel", group_keys=False
        ).apply(lambda x: x.sample(frac=fraction, random_state=42))

        # Filter the original dataframe to include only the sampled patients
        train_df = train_df[train_df["patient_id"].isin(sampled_patients["patient_id"])]

    data_splits = {
        "X": train_df.drop(columns=["SepsisLabel", "patient_id"]),
        "y": train_df["SepsisLabel"],
        "patient_ids": train_df["patient_id"],
    }
    return data_splits


def load_val_data() -> dict:
    """
    Loads the validation dataset.
    """
    root = find_project_root()
    val_df = pd.read_parquet(f"{root}/dataset/final_datasets/val.parquet")
    data_splits = {
        "X": val_df.drop(columns=["SepsisLabel", "patient_id"]),
        "y": val_df["SepsisLabel"],
        "patient_ids": val_df["patient_id"],
    }
    return data_splits


def load_test_data() -> dict:
    """
    Loads the test dataset.
    """
    root = find_project_root()
    test_df = pd.read_parquet(f"{root}/dataset/final_datasets/test.parquet")
    data_splits = {
        "X": test_df.drop(columns=["SepsisLabel", "patient_id"]),
        "y": test_df["SepsisLabel"],
        "patient_ids": test_df["patient_id"],
    }
    return data_splits
