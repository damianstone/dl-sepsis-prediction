from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


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
        f"Project root marker '{marker}' not found starting from {current}")


def preprocess_data(input_file, output_file):
    # make functions for the following:
    # sofa
    # qsofa
    # news
    # agregate window features
    # aggregate_scores
    # generate_multiwindow_features (only 6h)
    # drop columns = ["Unit1", "Unit2", "cluster_id", "dataset"]

    # check for nan values (should not be any)
    # print the new columns names added after feature engineering
    # print the total number of features (columns)
    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    root = find_project_root()
    INPUT_DATASET = f"{root}/dataset/Fully_imputed_dataset.parquet"
    OUTPUT_DATASET = f"{root}/dataset/V2_preprocessed.parquet"
    preprocess_data(OUTPUT_DATASET)
