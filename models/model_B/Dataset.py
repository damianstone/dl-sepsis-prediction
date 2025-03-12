import os
import torch
import pandas as pd
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

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
        self.pos_weight = None

        current_dir = os.getcwd()
        self.project_root = os.path.abspath(os.path.join(current_dir, "../.."))
        self.tensor_ds_path = os.path.join(self.project_root, "dataset", self.save_path)

    # TODO: balance formula
    def balance_dataset(self, df):
        """
        sampling technique: oversampling increases the number of minority class samples, 
        while undersampling reduces the number of majority class samples

        what it does: adjusts class distribution by either duplicating 
        minority samples (oversampling) or removing majority samples (undersampling) 
        to improve model learning balance
        """
        counts = df[self.target_column].value_counts()
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()
        
        df_majority = df[df[self.target_column] == majority_class]
        df_minority = df[df[self.target_column] == minority_class]

        if self.method == "oversample":
            n_samples = int(len(df_majority) * self.minority_ratio)
            df_minority_upsampled = resample(df_minority, replace=True, n_samples=n_samples, random_state=42)
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
        elif self.method == "undersample":
            n_samples = int(len(df_majority) * self.minority_ratio)
            df_majority_downsampled = resample(df_majority, replace=False, n_samples=n_samples, random_state=42)
            df_balanced = pd.concat([df_majority_downsampled, df_minority])
        else:
            raise ValueError("Method must be 'oversample' or 'undersample'")
        
        return df_balanced.sample(frac=1, random_state=42).reset_index(drop=True) 

    # TODO: method to get a smaller dataset preserving the balance proportions
    def reduce_dataset(self, df, train_size=0.3):
        reduced_df, _ = train_test_split(
            df, train_size=train_size, random_state=42, stratify=df[self.target_column]
        )
        return reduced_df
    
    # TODO: method to get the training and test data as tensor
    def get_train_test_tensors(self, size: Literal['small', 'full'], train_size):
        """
        0. if dataset_tensors.pth exist then return the train test data and skip the step below
        1. split into train and test from the original imbalance dataset
        2. calculate the post_weight to the use it in the loss function
        3. balance the train set 80/20 using sampling or the method "balance_dataset"
        4. use the method "reduce_dataset" to make it smaller preserving the balance proportions 
        5. get X_train and y_test from the new reduced balanced dataset
        6. get X_test and y_test from the original imbalance dataset
        7. save X_train, X_test, y_train and y_test as dataset_tensors.pth
        """
        # Step 0: Check if saved tensors exist.
        tensor_ds_path = f"{size}_{self.tensor_ds_path}"
        if os.path.exists(tensor_ds_path):
            data = torch.load(tensor_ds_path)
            print("Loaded saved dataset tensors")
            return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

        # Step 1: Load dataset and split into train and test sets (test remains unbalanced).
        df = pd.read_parquet(os.path.join(self.project_root, "dataset", self.data_path))
        
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df[self.target_column]
        )

        # Step 3: Balance the training set using the specified sampling method.
        if self.balance:
            train_df = self.balance_dataset(train_df)
            print("Balanced training set balance:")
            print(train_df[self.target_column].value_counts())

        # Step 4: Reduce the balanced training set to speed up experiments.
        if train_size:
            reduced_train_df = self.reduce_dataset(train_df, train_size=train_size)
            print("Reduced balanced training set balance:")
            print(reduced_train_df[self.target_column].value_counts())
            print(f"Total records in reduced training set: {reduced_train_df.shape[0]}")
        else:
            reduced_train_df = train_df

        # Step 5 & 6: Convert DataFrames to PyTorch tensors.
        X_train = torch.tensor(reduced_train_df.drop(columns=[self.target_column]).values, dtype=torch.float32)
        y_train = torch.tensor(reduced_train_df[self.target_column].values, dtype=torch.long)
        X_test = torch.tensor(test_df.drop(columns=[self.target_column]).values, dtype=torch.float32)
        y_test = torch.tensor(test_df[self.target_column].values, dtype=torch.long)

        # Step 7: Save the tensor dataset for future use.
        dataset_size=size
        save_full_path = os.path.join(self.project_root, "dataset", f"{dataset_size}_{self.save_path}")
        torch.save({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}, save_full_path)
        
        return X_train, X_test, y_train, y_test
    
