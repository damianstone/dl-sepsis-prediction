import matplotlib.pyplot as plt
import numpy as np


def summariseSeperateDatasets(df, hospital_name):
    num_patients = df['patient_id'].nunique()
    septic_patients = df.groupby('patient_id')['SepsisLabel'].max().sum()
    sepsis_prevalence = septic_patients / num_patients * 100
    num_rows = len(df)
    num_entries = df.count().sum()
    total_cells = df.shape[0] * df.shape[1]
    density = num_entries / total_cells * 100
    print(f"Hospital system: {hospital_name}")
    print(f"  Number of patients: {num_patients}")
    print(f"  Number of septic patients: {int(septic_patients)}")
    print(f"  Sepsis prevalence: {sepsis_prevalence:.1f}%")
    print(f"  Number of rows: {num_rows}")
    print(f"  Number of entries: {num_entries}")
    print(f"  Density of entries: {density:.1f}%")
    print("\n")

def summaryStatistics(df):
    print("Overall descriptive statistics:")
    print(df.describe(include=[float, int]))
    print("\n")

    missing_counts = df.isnull().sum()
    missing_percent = (missing_counts / len(df)) * 100
    print("Missing data per column:")
    for col in df.columns:
        print(f"{col}: {missing_counts[col]} missing ({missing_percent[col]:.1f}%)")
    print("\n")

    measurements_per_patient = df.groupby('patient_id').size()
    print("Average number of measurements per patient:", measurements_per_patient.mean())
    print("Median number of measurements per patient:", measurements_per_patient.median())
    print("\n")

    df_grouped = df.groupby('SepsisLabel')[['HR', 'Temp', 'SBP']].agg(['mean', 'std', 'median', 'min', 'max'])
    print("Summary of key vitals by sepsis status:")
    print(df_grouped)
    print("\n")

def covariance(df):
    key_vars = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2',
       'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN',
       'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct',
       'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium',
       'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC',
       'Fibrinogen', 'Platelets', 'Age', 'Gender', 'Unit1', 'Unit2',
       'HospAdmTime', 'ICULOS', 'SepsisLabel']
    corr_matrix = df[key_vars].corr()
    print("Correlation matrix among key vital signs:")
    print(corr_matrix)
    
    plt.figure(figsize=(12, 12))
    im = plt.imshow(corr_matrix, cmap='viridis', interpolation='none')
    plt.title("Correlation Matrix of Key Variables")
    plt.colorbar(im)
    plt.xticks(np.arange(len(key_vars)), key_vars, rotation=45)
    plt.yticks(np.arange(len(key_vars)), key_vars)
    plt.tight_layout()
    plt.show()

def nullCols(df):
    cols_to_analyse = [col for col in df.columns if col not in ['patient_id', 'dataset']]
    missing_counts = df[cols_to_analyse].isnull().sum()
    missing_percent = missing_counts / len(df) * 100
    missing_percent_sorted = missing_percent.sort_values()
    plt.figure(figsize=(12, 6))
    plt.bar(missing_percent_sorted.index, missing_percent_sorted.values, color='skyblue')
    plt.xlabel("Columns")
    plt.ylabel("Missing Percentage (%)")
    plt.title("Missing Data Percentage per Column")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()