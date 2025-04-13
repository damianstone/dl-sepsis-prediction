import os
import sys
from pathlib import Path

import pandas as pd
from model_utils.helper_functions import display_balance_statistics
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# NOTE: purpose is just to split the data


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


def save_processed_data(root, train_df, val_df, test_df, file_name):
    df_train = train_df.copy()
    df_val = val_df.copy()
    df_test = test_df.copy()

    save_path = f"{root}/dataset/{file_name}.parquet"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df_train.to_parquet(f"{root}/dataset/{file_name}_train.parquet")
    df_val.to_parquet(f"{root}/dataset/{file_name}_val.parquet")
    df_test.to_parquet(f"{root}/dataset/{file_name}_test.parquet")


def load_processed_data(root, file_name):
    train_df = pd.read_parquet(f"{root}/dataset/{file_name}_train.parquet")
    val_df = pd.read_parquet(f"{root}/dataset/{file_name}_val.parquet")
    test_df = pd.read_parquet(f"{root}/dataset/{file_name}_test.parquet")

    data_splits = {
        "X_train": train_df.drop(columns=["SepsisLabel", "patient_id"]),
        "y_train": train_df["SepsisLabel"],
        "patient_ids_train": train_df["patient_id"],
        "X_val": val_df.drop(columns=["SepsisLabel", "patient_id"]),
        "y_val": val_df["SepsisLabel"],
        "patient_ids_val": val_df["patient_id"],
        "X_test": test_df.drop(columns=["SepsisLabel", "patient_id"]),
        "y_test": test_df["SepsisLabel"],
        "patient_ids_test": test_df["patient_id"],
    }

    print("Loaded from last processed data")
    return data_splits


def over_under_sample(df, method="oversample", minority_ratio=0.3):
    """
    Balances the dataset at the patient level.

    Each patient's overall sepsis label is taken as the maximum value
    (if any record shows sepsis, the patient is marked as septic).

    We then either oversample the septic (minority) patients or undersample
    the non-septic (majority) patients to change the ratio.

    In the final dataset, each copy of a patient gets a unique ID so that
    oversampled patients appear as separate instances.
    """
    # Create a patient-level summary with one record per patient.
    patient_df = df.groupby("patient_id")["SepsisLabel"].max().reset_index()

    # Count patients in each group.
    counts = patient_df["SepsisLabel"].value_counts()
    majority_class = counts.idxmax()
    minority_class = counts.idxmin()

    # Split the patients into majority and minority groups.
    majority_patients = patient_df[patient_df["SepsisLabel"] == majority_class]
    minority_patients = patient_df[patient_df["SepsisLabel"] == minority_class]

    # Resample based on the chosen method.
    if method == "oversample":
        # Duplicate minority patients to reach desired ratio.
        n_desired_minority = int(
            (minority_ratio * len(majority_patients)) / (1 - minority_ratio)
        )
        minority_upsampled = resample(
            minority_patients,
            replace=True,
            n_samples=n_desired_minority,
            random_state=42,
        )
        balanced_patient_df = pd.concat([majority_patients, minority_upsampled])
    elif method == "undersample":
        # Remove some majority patients to reach desired ratio.
        n_desired_majority = int(
            ((1 - minority_ratio) / minority_ratio) * len(minority_patients)
        )
        majority_downsampled = resample(
            majority_patients,
            replace=False,
            n_samples=n_desired_majority,
            random_state=42,
        )
        balanced_patient_df = pd.concat([majority_downsampled, minority_patients])
    else:
        raise ValueError("Method must be 'oversample' or 'undersample'")

    # Rebuild the full dataset with patient records.
    # If a patient appears more than once due to resampling,
    # assign a new unique patient ID to each duplicate.
    final_dfs = []
    patient_occurrences = {}

    for pid in balanced_patient_df["patient_id"]:
        # Get all records for this patient.
        patient_records = df[df["patient_id"] == pid].copy()
        # Count how many times this patient has been added.
        if pid in patient_occurrences:
            patient_occurrences[pid] += 1
            # Create a new unique ID by appending a suffix.
            new_pid = pid * 1000 + patient_occurrences[pid]
            patient_records["patient_id"] = new_pid
        else:
            # first occurrence, keep original ID
            patient_occurrences[pid] = 0
        final_dfs.append(patient_records)

    balanced_df = pd.concat(final_dfs, ignore_index=True)

    display_balance_statistics(balanced_df)
    return balanced_df


def reduce_dataset(df, train_sample_fraction=0.05):
    patient_df = df.groupby("patient_id")["SepsisLabel"].max().reset_index()

    # stratified sampling to get a subset of patient IDs
    sample_ids, _ = train_test_split(
        patient_df["patient_id"],
        train_size=train_sample_fraction,
        stratify=patient_df["SepsisLabel"],
        random_state=42,
    )
    quick_train_df = df[df["patient_id"].isin(sample_ids)].copy()
    return quick_train_df


def preprocess_data(
    data_file_name,
    sampling=True,
    use_last_processed_data=False,
    sampling_method="oversample",
    sampling_minority_ratio=0.3,
    train_sample_fraction=0.05,
    test_size=0.2,
    random_state=42,
):

    project_root = find_project_root()
    print("Project root:", project_root)
    if project_root not in sys.path:
        sys.path.append(project_root)

    if use_last_processed_data:
        return load_processed_data(
            root=project_root,
            file_name="small_imputed_sofa",
        )

    df_path = f"{project_root}/dataset/{data_file_name}.parquet"
    try:
        df = pd.read_parquet(df_path)
    except Exception as e:
        sys.exit(f"Error loading dataset from {df_path}: {e}")

    # First get patient-level labels (1 if any record has sepsis)
    patient_labels = df.groupby("patient_id")["SepsisLabel"].max()

    train_val_patients, test_patients = train_test_split(
        patient_labels.index,
        test_size=test_size,
        random_state=random_state,
        stratify=patient_labels,  # Stratify by patient-level labels
    )

    train_val_df = df[df["patient_id"].isin(train_val_patients)]
    test_df = df[df["patient_id"].isin(test_patients)]

    if train_sample_fraction < 1.0:
        train_val_df = reduce_dataset(
            df=train_val_df, train_sample_fraction=train_sample_fraction
        )

    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=0.125,
        random_state=42,
        stratify=patient_labels[train_val_patients],
    )

    train_df = train_val_df[train_val_df["patient_id"].isin(train_patients)]
    val_df = train_val_df[train_val_df["patient_id"].isin(val_patients)]

    # Then balance both sets separately to ensure exact same balance
    if sampling:
        train_df = over_under_sample(
            df=train_df, method=sampling_method, minority_ratio=sampling_minority_ratio
        )
        val_df = over_under_sample(
            df=val_df, method=sampling_method, minority_ratio=sampling_minority_ratio
        )

    save_processed_data(
        project_root, train_df, val_df, test_df, file_name="small_imputed_sofa"
    )

    print("Processed data saved")

    # as pandas dataframe
    data_splits = {
        "X_train": train_df.drop(columns=["SepsisLabel", "patient_id"]),
        "y_train": train_df["SepsisLabel"],
        "patient_ids_train": train_df["patient_id"],
        "X_val": val_df.drop(columns=["SepsisLabel", "patient_id"]),
        "y_val": val_df["SepsisLabel"],
        "patient_ids_val": val_df["patient_id"],
        "X_test": test_df.drop(columns=["SepsisLabel", "patient_id"]),
        "y_test": test_df["SepsisLabel"],
        "patient_ids_test": test_df["patient_id"],
    }

    return data_splits


if __name__ == "__main__":
    data_splits = preprocess_data(
        use_last_processed_data=False, data_file_name="big_imputed_sofa"
    )
    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]
    patient_ids_train = data_splits["patient_ids_train"]

    X_val = data_splits["X_val"]
    y_val = data_splits["y_val"]
    patient_ids_val = data_splits["patient_ids_val"]

    X_test = data_splits["X_test"]
    y_test = data_splits["y_test"]
    patient_ids_test = data_splits["patient_ids_test"]
