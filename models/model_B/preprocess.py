import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# NOTE: purpose is just to split the data 

def preprocess_data(data_file_name, test_size=0.2, random_state=42):
    notebook_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(notebook_dir, "../.."))
    print("Project root:", project_root)
    if project_root not in sys.path:
        sys.path.append(project_root)
        
    # TODO: 1: get the imputed dataset
    from utils import get_data
    DATA_PATH = get_data.get_dataset_abspath()
    input_file = f"{data_file_name}.parquet"
    load_path = os.path.join(DATA_PATH, input_file)
    
    try:
        df = pd.read_parquet(load_path)
    except Exception as e:
        sys.exit(f"Error loading dataset from {load_path}: {e}")
    
    # TODO: 2: split
    X = df.drop(columns=["SepsisLabel"])
    y = df["SepsisLabel"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test 
    

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = preprocess_data("fulle_imputed_sofa")
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")