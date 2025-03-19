import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

def find_project_root(marker=".gitignore"):
    """
    walk up from the current working directory until a directory containing the
    specified marker (e.g., .gitignore) is found.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(f"Project root marker '{marker}' not found starting from {current}")

def calculate_sofa(row):
    sofa = 0
    
    if row['FiO2'] > 0:
        pao2_fio2 = row['SaO2'] / row['FiO2']
        if pao2_fio2 < 100: sofa += 4
        elif pao2_fio2 < 200: sofa += 3
        elif pao2_fio2 < 300: sofa += 2
        elif pao2_fio2 < 400: sofa += 1

    if row['Platelets'] < 20: sofa += 4
    elif row['Platelets'] < 50: sofa += 3
    elif row['Platelets'] < 100: sofa += 2
    elif row['Platelets'] < 150: sofa += 1

    if row['Bilirubin_total'] >= 12: sofa += 4
    elif row['Bilirubin_total'] >= 6: sofa += 3
    elif row['Bilirubin_total'] >= 2: sofa += 2
    elif row['Bilirubin_total'] >= 1.2: sofa += 1

    if row['MAP'] < 70: sofa += 1

    if row['Creatinine'] >= 5: sofa += 4
    elif row['Creatinine'] >= 3.5: sofa += 3
    elif row['Creatinine'] >= 2: sofa += 2
    elif row['Creatinine'] >= 1.2: sofa += 1

    return sofa

def pre_process_sofa(input_file,output_file):
    project_root = find_project_root()
    print("Project root:", project_root)
    if project_root not in sys.path:
        sys.path.append(project_root)

    # Import imputed dataset
    from utils import get_data
    DATA_PATH = get_data.get_dataset_abspath()
    load_path = os.path.join(DATA_PATH, input_file)

    # Load imputed datset
    try:
        imputed_df = pd.read_parquet(load_path)
    except Exception as e:
        sys.exit(f"Error loading dataset from {load_path}: {e}")

    # Remove excluded columns if they exist
    exclude_columns = ["Unit1", "Unit2", "cluster_id", "dataset"]
    imputed_df.drop(columns=exclude_columns, errors='ignore', inplace=True)
    # NOTE: for now manually drop this nan values
    imputed_df = imputed_df.dropna(subset=["HospAdmTime"])

    # Check for any remaining NaN values in the dataset; if found, return an error.
    if imputed_df.isna().sum().sum() > 0:
        print(imputed_df.isna().sum())
        sys.exit("Error: The dataset contains missing values.")

    # If 'SOFA' column does not exist, calculate and add it.
    if "SOFA" not in imputed_df.columns:
        imputed_df["SOFA"] = imputed_df.apply(calculate_sofa, axis=1)

    # Print all feature names and the count of missing values per column.
    print("Features in the dataset:")
    print(imputed_df.columns.tolist())
    print("\nMissing values per column:")
    print(imputed_df.isna().sum())

    # Save the processed dataset to the provided output file
    try:
        to_save = f"{project_root}/dataset/{output_file}"
        imputed_df.to_parquet(to_save, index=False)
        print(f"\nDataset successfully saved to {to_save}")
    except Exception as e:
        sys.exit(f"Error saving dataset to {to_save}: {e}")

if __name__ == '__main__':
    pre_process_sofa("Fully_imputed_dataset.parquet","big_imputed_sofa.parquet")