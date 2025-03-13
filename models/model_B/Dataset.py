import os
import torch
import pandas as pd
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from t_utils import display_balance_statistics


class Dataset:
    def __init__(self,
                 data_path="imputed_sofa.parquet",
                 save_path="dataset_tensors.pth",
                 method="oversample",
                 balance=False,
                 minority_ratio=0.2,
                 target_column="SepsisLabel"):
        self.data_path = data_path
        self.save_path = save_path
        self.method = method
        self.balance = balance
        self.minority_ratio = minority_ratio
        self.target_column = target_column
        self.post_weight_ratio = None
        self.train_df = None
        self.test_df = None

        current_dir = os.getcwd()
        self.project_root = os.path.abspath(os.path.join(current_dir, "../.."))
        self.tensor_ds_path = os.path.join(self.project_root, "dataset", self.save_path)

    # TODO: over and under sampling
    def _balance_over_under_sample(self, df):
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
        patient_df = df.groupby("patient_id")[
            "SepsisLabel"].max().reset_index()

        # Count patients in each group.
        counts = patient_df["SepsisLabel"].value_counts()
        majority_class = counts.idxmax()  # e.g., non-septic (0)
        minority_class = counts.idxmin()  # e.g., septic (1)

        # Split the patients into majority and minority groups.
        majority_patients = patient_df[patient_df["SepsisLabel"]
                                       == majority_class]
        minority_patients = patient_df[patient_df["SepsisLabel"]
                                       == minority_class]

        # Resample based on the chosen method.
        if self.method == "oversample":
            # Duplicate minority patients to reach desired ratio.
            # n_samples = int(len(majority_patients) * minority_ratio)
            n_desired_minority = int(
                (self.minority_ratio * len(majority_patients)) / (1 - self.minority_ratio))
            minority_upsampled = resample(minority_patients, replace=True,
                                          n_samples=n_desired_minority, random_state=42)
            balanced_patient_df = pd.concat(
                [majority_patients, minority_upsampled])
        elif self.method == "undersample":
            # Remove some majority patients to reach desired ratio.
            # n_samples = int(len(minority_patients) / minority_ratio)
            n_desired_majority = int(
                ((1 - self.minority_ratio) / self.minority_ratio) * len(minority_patients))
            majority_downsampled = resample(majority_patients, replace=False,
                                            n_samples=n_desired_majority, random_state=42)
            balanced_patient_df = pd.concat(
                [majority_downsampled, minority_patients])
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
                new_pid = f"{pid}_dup{patient_occurrences[pid]}"
                patient_records["patient_id"] = new_pid
            else:
                # first occurrence, keep original ID
                patient_occurrences[pid] = 0
            final_dfs.append(patient_records)

        balanced_df = pd.concat(final_dfs, ignore_index=True)

        display_balance_statistics(balanced_df)
        return balanced_df

    # TODO: method to get a smaller dataset preserving the balance proportions
    def _reduce_dataset(self, df, sample_fraction=0.1):
        patient_df = df.groupby("patient_id")[
            "SepsisLabel"].max().reset_index()

        # stratified sampling to get a subset of patient IDs
        sample_ids, _ = train_test_split(
            patient_df["patient_id"],
            train_size=sample_fraction,
            stratify=patient_df["SepsisLabel"],
            random_state=42
        )
        quick_train_df = df[df["patient_id"].isin(sample_ids)].copy()
        return quick_train_df
    
    def _get_post_weight_ratio(self, train_df):
        """
        useful for post_weight loss function
        """
        patient_summary = train_df.groupby("patient_id")["SepsisLabel"].max().reset_index()
        negative_count = (patient_summary["SepsisLabel"] == 0).sum()
        positive_count = (patient_summary["SepsisLabel"] == 1).sum()
        if positive_count == 0:
            raise ValueError("No positive samples found in training set.")
        print("Negative-to-positive ratio (per patient):", negative_count / positive_count)
        return round(negative_count / positive_count)

    def _pre_process_train_df(self, sample_fraction):
        """
        1. split into train and test from the original imbalance dataset
        2. balance the train set using over or under sampling
        3. calculate the post_weight to the use it in the loss function
        4. use the method "reduce_dataset" to make it smaller preserving the balance proportions 
        5. drop useless columns
        """
        # Step 1: Load dataset and split into train and test sets (test remains unbalanced).
        df = pd.read_parquet(os.path.join(
            self.project_root, "dataset", self.data_path))

        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df[self.target_column]
        )
        
        self.test_df = test_df.drop(columns=["Unit1", "Unit2", "cluster_id","dataset", "patient_id"], errors='ignore')

        # Step 2: Balance the training set using the specified sampling method.
        if self.balance:
            train_df = self._balance_over_under_sample(train_df)

        # Step 3
        self.post_weight_ratio = self._get_post_weight_ratio(train_df)
        
        # Step 5: Reduce the balanced training set to speed up experiments.
        if sample_fraction:
            train_df = self._reduce_dataset(train_df, sample_fraction)
            print(f"Total records in reduced training set: {train_df.shape[0]}")
            
        # drop useless columns
        train_df = train_df.drop(columns=["Unit1", "Unit2", "cluster_id","dataset", "patient_id"], errors='ignore')
            
        return train_df
        
    # TODO: method to get the training and test data as tensor
    def get_train_test_tensors(self, size: Literal['small', 'full'], sample_fraction):
        """
        5. get X_train and y_test from the new reduced balanced dataset
        6. get X_test and y_test from the original imbalance dataset
        7. save X_train, X_test, y_train and y_test as dataset_tensors.pth
        """
        # check if saved tensors exist.
        tensor_ds_path = f"{size}_{self.tensor_ds_path}"
        if os.path.exists(tensor_ds_path):
            data = torch.load(tensor_ds_path)
            print("Loaded saved dataset tensors")
            return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

        if self.train_df is None:
            self.train_df = self._pre_process_train_df(sample_fraction)

        # Step 5 & 6: Convert DataFrames to PyTorch tensors
        X_train = torch.tensor(self.train_df.drop(columns=[self.target_column]).values, dtype=torch.float32)
        y_train = torch.tensor(self.train_df[self.target_column].values, dtype=torch.long)
        X_test = torch.tensor(self.test_df.drop(columns=[self.target_column]).values, dtype=torch.float32)
        y_test = torch.tensor(self.test_df[self.target_column].values, dtype=torch.long)

        # Step 7: Save the tensor dataset for future use
        dataset_size = size
        save_full_path = os.path.join(self.project_root, "dataset", f"{dataset_size}_{self.save_path}")
        torch.save({"X_train": X_train, "X_test": X_test,
                   "y_train": y_train, "y_test": y_test}, save_full_path)

        return X_train, X_test, y_train, y_test
