import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

notebook_dir = os.getcwd()

project_root = os.path.abspath(os.path.join(notebook_dir, "../.."))
print(project_root)
if project_root not in sys.path:
    sys.path.append(project_root)

from utils import get_data

DATA_PATH = get_data.get_dataset_abspath()
load_path = os.path.join(DATA_PATH, "imputed_combined_data.parquet")
imputed_df = pd.read_parquet(load_path)

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

imputed_df = imputed_df.drop(columns=["Unit1", "Unit2", "cluster_id","dataset", "patient_id"], errors='ignore')
imputed_df = imputed_df.dropna(subset=["HospAdmTime"])
imputed_df["SOFA"] = imputed_df.apply(calculate_sofa, axis=1)

print(imputed_df.isna().sum())

imputed_df.to_parquet(f"{project_root}/dataset/imputed_sofa.parquet", index=False)