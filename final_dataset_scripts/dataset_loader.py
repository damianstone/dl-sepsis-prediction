from pathlib import Path

import pandas as pd

types = ["no_sampling", "oversampling", "undersampling"]


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


def load_train_data(type) -> dict:
    """
    Loads the final dataset for a given type.
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
