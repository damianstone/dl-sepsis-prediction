import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
import math
from scipy.stats import ks_2samp, wasserstein_distance, entropy, chi2_contingency
from scipy.spatial.distance import jensenshannon

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

def distributions(df, columns=None, ncols=3):
    if columns is None:
        exclude = ['patient_id', 'dataset', 'SepsisLabel', 'Unit1', 'Unit2']
        numeric_cols = df.select_dtypes(include=['number']).columns
        columns = [col for col in numeric_cols if col not in exclude]
    n = len(columns)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 5, nrows * 4))
    if nrows * ncols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    for i, col in enumerate(columns):
        sepsis_data = df[df['SepsisLabel'] == 1][col].dropna()
        non_sepsis_data = df[df['SepsisLabel'] == 0][col].dropna()
        ks_result = ks_2samp(sepsis_data, non_sepsis_data)
        ax = axes[i]
        sns.kdeplot(sepsis_data, ax=ax, label='Sepsis', shade=True, color='red', alpha=0.5)
        sns.kdeplot(non_sepsis_data, ax=ax, label='Non-Sepsis', shade=True, color='blue', alpha=0.5)
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()




def compute_histogram(arr, bins=50):
    hist, bin_edges = np.histogram(arr, bins=bins, density=True)
    pmf = hist / np.sum(hist)
    return pmf, bin_edges

def compute_jsd(arr1, arr2, bins=50):
    pmf1, _ = compute_histogram(arr1, bins)
    pmf2, _ = compute_histogram(arr2, bins)
    jsd = jensenshannon(pmf1, pmf2, base=2.0)
    return jsd**2

def compute_kl(arr1, arr2, bins=50):
    pmf1, _ = compute_histogram(arr1, bins)
    pmf2, _ = compute_histogram(arr2, bins)
    epsilon = 1e-10
    pmf1 += epsilon
    pmf2 += epsilon
    return entropy(pmf1, pmf2)



def compare_two_datasets(df_A, df_B, bins=50, gamma=1.0,
                         ks_p_threshold=0.05, jsd_threshold=0.2, chi2_p_threshold=0.05):
    results = {}
    common_cols = set(df_A.columns).intersection(set(df_B.columns))
    
    for col in common_cols:
        res = {}
        series_A = df_A[col].dropna()
        series_B = df_B[col].dropna()
        
        if pd.api.types.is_numeric_dtype(df_A[col]):
            if len(series_A) > 10 and len(series_B) > 10:
                ks_stat, ks_p = ks_2samp(series_A, series_B)
                res['KS Statistic'] = ks_stat
                res['KS p-value'] = ks_p
                res['Wasserstein Distance'] = wasserstein_distance(series_A, series_B)
                res['JSD'] = compute_jsd(series_A, series_B, bins=bins)
                res['KL Divergence'] = compute_kl(series_A, series_B, bins=bins)
                
                # if KS p-value is high and JSD is low, merge is valid.
                valid = (ks_p >= ks_p_threshold) and (res['JSD'] <= jsd_threshold)
                res['Valid Merge'] = valid
            else:
                res['Note'] = "Not enough numeric data for reliable metrics."
                res['Valid Merge'] = False
        else:
            counts_A = series_A.value_counts().sort_index()
            counts_B = series_B.value_counts().sort_index()
            all_categories = sorted(set(counts_A.index).union(set(counts_B.index)))
            vec_A = np.array([counts_A.get(cat, 0) for cat in all_categories])
            vec_B = np.array([counts_B.get(cat, 0) for cat in all_categories])
            contingency = np.vstack([vec_A, vec_B])
            try:
                chi2, p_val, dof, ex = chi2_contingency(contingency)
                res['Chi2 Statistic'] = chi2
                res['Chi2 p-value'] = p_val
                res['Valid Merge'] = (p_val >= chi2_p_threshold)
            except Exception as e:
                res['Chi2 Error'] = str(e)
                res['Valid Merge'] = False
        
        results[col] = res
    return results



