import torch
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn


def get_dataset_tensors(data_path="imputed_sofa_ds.csv", save_path="dataset_tensors.pth"):
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    tensor_ds_path = f"{project_root}/dataset/{save_path}"
    if os.path.exists(tensor_ds_path):
        data = torch.load(tensor_ds_path)
        print("from saved dataset")
        return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    imputed_df = pd.read_csv(f"{project_root}/dataset/{data_path}")
    
    X = imputed_df.drop(columns=['SepsisLabel']).values
    y = imputed_df['SepsisLabel'].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

    path = f"{project_root}/dataset/{save_path}"
    torch.save({"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}, path)

    return X_train, X_test, y_train, y_test
