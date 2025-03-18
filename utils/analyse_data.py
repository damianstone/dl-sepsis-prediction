import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp
import math
from scipy.stats import ks_2samp, wasserstein_distance, entropy, chi2_contingency
from scipy.spatial.distance import jensenshannon
from scipy.stats import linregress
import matplotlib.colors as mcolors

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
    missing_percent_sorted = missing_percent.sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    plt.bar(missing_percent_sorted.index, missing_percent_sorted.values, color='#7393B3')
    plt.ylabel("Missing Percentage")
    plt.xticks(rotation=90, ha='right')
    plt.tight_layout()
    plt.show()

def plot_density_of_actual_values(df, value_col='ICULOS'):
    patient_iculos_density = df.groupby(['patient_id', value_col]).apply(lambda x: x.notna().mean().mean())

    avg_density = patient_iculos_density.groupby(value_col).mean()

    plt.figure(figsize=(12, 6))
    plt.plot(avg_density.index, avg_density.values, color='navy')
    plt.fill_between(avg_density.index, avg_density.values, alpha=0.3, color='navy')
    plt.xlabel('ICU Length of Stay (hours)')
    plt.ylabel('Average Density of Actual Values')
    plt.grid(alpha=0.3)
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
        ax.axvline(sepsis_data.mean(), color='red', linestyle='--', linewidth=1.5, label='Sepsis Mean')
        ax.axvline(non_sepsis_data.mean(), color='blue', linestyle='--', linewidth=1.5, label='Non-Sepsis Mean')
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.legend()
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

def plot_percentage_sepsis_grid(df):
    exclude = ['patient_id', 'dataset', 'SepsisLabel', 'Unit1', 'Unit2', 'Gender', 'FiO2']
    features = [col for col in df.columns if col not in exclude and df[col].dtype in [np.float64, np.int64]]

    n_cols = 3
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        df_non_null = df.dropna(subset=[col])
        bins = np.linspace(df_non_null[col].min(), df_non_null[col].max(), 200)
        df_non_null['MeasurementBin'] = pd.cut(df_non_null[col], bins=bins)

        bin_counts = df_non_null.groupby('MeasurementBin').size()
        valid_bins = bin_counts[bin_counts >= 1].index

        sepsis_percentage = df_non_null[df_non_null['MeasurementBin'].isin(valid_bins)].groupby('MeasurementBin')['SepsisLabel'].mean() * 100
        sepsis_percentage = sepsis_percentage[sepsis_percentage > 0]
        bin_centers = [interval.mid for interval in sepsis_percentage.index]

        axes[i].plot(bin_centers, sepsis_percentage, ',-', label='Sepsis %', color='black', linewidth=1)
        
        slope, intercept, r_value, p_value, std_err = linregress(bin_centers, sepsis_percentage)
        print(f"{col}:slope:{slope}, c:{intercept}")
        best_fit = slope * np.array(bin_centers) + intercept
        axes[i].plot(bin_centers, best_fit, 'r--', label=f'Best Fit (slope={slope:.2f})')

        axes[i].set_title(f'Sepsis % by {col} Gradient: {slope:.4f}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Sepsis Percentage (%)')
        axes[i].legend()
        axes[i].grid(alpha=0.3)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def plot_avg_missing_data_sepsis(df):
    missing_data = df.isnull().mean(axis=1)
    avg_missingness = missing_data = missing_data = missing_data = df.assign(missing=missing_data).groupby('SepsisLabel')['missing'].mean() * 100
    plt.figure(figsize=(8,6))
    sns.barplot(x=avg_missingness.index, y=avg_missingness.values, palette=['skyblue', 'salmon'])
    plt.xticks([0, 1], ['Non-Sepsis', 'Sepsis'])
    plt.ylabel('Average Missing Data (%)')
    plt.xlabel('Sepsis Label')
    plt.title('Average Percentage of Missing Data by Sepsis Status')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_average_lab_trajectories(df):
    lab_values = ['Lactate', 'WBC', 'Creatinine', 'Platelets']
    df_grouped = df.groupby(['ICULOS', 'SepsisLabel'])[lab_values].mean().reset_index()

    for lab in lab_values:
        plt.figure(figsize=(12, 6))
        sns.lineplot(x='ICULOS', y=lab, hue='SepsisLabel', data=df_grouped)
        plt.title(f'Average {lab} Trajectory: Sepsis vs Non-Sepsis')
        plt.xlabel('ICU Length of Stay (hours)')
        plt.ylabel(f'{lab} Levels')
        plt.grid(alpha=0.3)
        plt.show()

def plot_sepsis_by_age_bucket(df):
    bins = [0, 20, 40, 60, 80, 100]
    labels = ['0-20', '21-40', '41-60', '61-80', '81-100']
    df['AgeBucket'] = pd.cut(df['Age'], bins=bins, labels=labels)

    sepsis_by_age = df.groupby('AgeBucket')['SepsisLabel'].mean() * 100

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sepsis_by_age.index, y=sepsis_by_age.values, palette='Blues')
    plt.title('Percentage of Sepsis Occurrence by Age Group')
    plt.xlabel('Age Bucket')
    plt.ylabel('Percentage with Sepsis (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
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




def plot_combined_metrics(results):
    df = pd.DataFrame(results).T
    
    metrics_to_plot = ['KL Divergence', 'JSD', 'Wasserstein Distance', 'KS Statistic']
    
    df_plot = df[metrics_to_plot].apply(pd.to_numeric, errors='coerce').dropna()
    
    df_norm = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())
    
    n = len(df_norm)
    indices = np.arange(n)  
    width = 0.2             
    
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.bar(indices + 1.5*width, df_norm['KS Statistic'], width, label='KS Statistic')
    ax.bar(indices + 0.5*width, df_norm['Wasserstein Distance'], width, label='Wasserstein Distance')
    ax.bar(indices - 1.5*width, df_norm['KL Divergence'], width, label='KL Divergence')
    ax.bar(indices - 0.5*width, df_norm['JSD'], width, label='JSD')
    
    ax.set_ylabel('Normalised Value')
    ax.set_xticks(indices)
    ax.set_xticklabels(df_norm.index, rotation=90, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.show()


def plot_inbalance_pie_chart():
    values = [8.8, 91.2]
    colors = ['#7393B3', '#E0E0E0']
    
    plt.figure(figsize=(5, 5))
    wedges, texts, autotexts = plt.pie(values, autopct='%1.1f%%', colors=colors, startangle=90,
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}, textprops={'fontsize': 20})
    plt.title('Hospital A (8.8%)')
    plt.axis('equal')  
    plt.show()

    values = [5.7, 94.3]
    colors = ['#7393B3', '#E0E0E0']
    
    plt.figure(figsize=(5, 5))
    wedges, texts, autotexts = plt.pie(values, autopct='%1.1f%%', colors=colors, startangle=90,
                                       wedgeprops={'edgecolor': 'black', 'linewidth': 1.5}, textprops={'fontsize': 20})
    plt.title('Hospital B (5.7%)')
    plt.axis('equal')  
    plt.show()

def plot_combined_heatmap(results):
    df = pd.DataFrame(results).T
    
    metrics_to_plot = ['KS Statistic', 'Wasserstein Distance', 'KL Divergence', 'JSD' ]
    
    df_plot = df[metrics_to_plot].apply(pd.to_numeric, errors='coerce').dropna()
    
    df_norm = (df_plot - df_plot.min()) / (df_plot.max() - df_plot.min())
    
    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(df_norm, aspect='auto', cmap='viridis')
    
    ax.set_xticks(np.arange(len(metrics_to_plot)))
    ax.set_xticklabels(metrics_to_plot, rotation=10, ha='right')
    
    ax.set_yticks(np.arange(len(df_norm.index)))
    ax.set_yticklabels(df_norm.index)
    
    # for i in range(len(df_norm.index)):
    #     for j in range(len(metrics_to_plot)):
    #         value = df_norm.iloc[i, j]
    #         ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="w")
    # heatmap
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()





def plot_metrics(results):
    # Create a DataFrame from the results and transpose it so that rows become the different items
    df_metrics = pd.DataFrame(results).T

    
    # Convert all columns to numeric (coercing errors to NaN)
    for col in df_metrics.columns:
        df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce')
    
    # Identify numeric columns that are not completely NaN
    numeric_cols = df_metrics.columns[~df_metrics.isna().all()]
    
    # Filter for the specific metric (here only 'JSD' is plotted; modify as needed)
    numeric_cols = [col for col in numeric_cols if col == 'JSD']
    
    if len(numeric_cols) == 0:
        print("No numeric metrics available to plot.")
        return

    num_metrics = len(numeric_cols)
    
    # Create subplots, one for each metric
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 5 * num_metrics))
    if num_metrics == 1:
        axes = [axes]
    
    # Plot each metric after sorting its values in descending order
    for ax, metric in zip(axes, numeric_cols):
        sorted_data = df_metrics[metric].sort_values(ascending=False)
        sorted_data = sorted_data[:-2]
        print(sorted_data)
        print(len(sorted_data))
        ax.bar(sorted_data.index, sorted_data.values, color='#7393B3')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=90)
    
    plt.tight_layout()
    plt.show()






def compare_two_datasets(df_A, df_B, bins=50, gamma=1.0,
                         ks_p_threshold=0.05, jsd_threshold=0.5, chi2_p_threshold=0.05):
    results = {}
    common_cols = set(df_A.columns).intersection(set(df_B.columns))
    common_cols.discard("patient_id")
    for col in common_cols:
        res = {}
        series_A = df_A[col].dropna()
        series_B = df_B[col].dropna()
        
        if pd.api.types.is_numeric_dtype(df_A[col]):
            if len(series_A) > 10 and len(series_B) > 10:
                ks_stat, ks_p = ks_2samp(series_A, series_B)
                res['KS Statistic'] = ks_stat
                res['Wasserstein Distance'] = wasserstein_distance(series_A, series_B)
                res['JSD'] = compute_jsd(series_A, series_B, bins=bins)
                res['KL Divergence'] = compute_kl(series_A, series_B, bins=bins)
                
                # if KS p-value is high and JSD is low, merge is valid.
                valid = (ks_p >= ks_p_threshold) and (res['JSD'] <= jsd_threshold)
                res['Valid Merge'] = True
            else:
                res['Note'] = "Not enough numeric data for reliable metrics."
                res['Valid Merge'] = True
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



def plot_feature_presence_by_prevalence(df: pd.DataFrame) -> None:
    """
    Shows a matrix of per-patient feature presence (0 = all NaN, 1 = some data),
    with rows in the original patient order, and columns sorted by overall
    feature prevalence.
    """
    # Exclude columns you do not want in the presence map
    exclude_cols = [
        "patient_id", "Age", "Gender", "HospAdmTime", 
        "dataset", "SepsisLabel", "ICULOS", "Unit1", "Unit2"
    ]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Capture the original order of patients as they appear in the DataFrame
    # (groupby can change ordering, so we'll do this explicitly)
    unique_patients_in_order = df["patient_id"].unique()
    
    # Build a list of DataFrames, one per patient, in the original order
    patient_dfs = []
    for pid in unique_patients_in_order:
        patient_dfs.append(df[df["patient_id"] == pid])

    # For each patient, determine if each feature is all NaN (0) or has data (1)
    presence_rows = []
    for pid_df in patient_dfs:
        pid = pid_df["patient_id"].iloc[0]
        row_data = {}
        for col in feature_cols:
            # 0 if completely empty (all NaN), 1 if there's at least one non-NaN
            row_data[col] = 0 if pid_df[col].isna().all() else 1
        presence_rows.append((pid, row_data))

    # Convert to a DataFrame where rows = patients, columns = features
    patient_presence = pd.DataFrame(
        [row_data for (_, row_data) in presence_rows],
        index=[pid for (pid, _) in presence_rows]
    )
    patient_presence.index.name = "patient_id"

    # Compute overall prevalence of each feature = fraction of patients that have at least one value
    feature_prevalence = patient_presence.mean(axis=0)  # average of 0/1 => fraction of 1's
    
    # Sort features by ascending prevalence
    feature_prevalence_sorted = feature_prevalence.sort_values(ascending=False)
    sorted_features = feature_prevalence_sorted.index.tolist()
    
    # Reorder columns in the presence matrix
    presence_matrix = patient_presence[sorted_features].values

    # Create a ListedColormap: 0 -> lightblue, 1 -> lightred
    cmap = mcolors.ListedColormap(['#E0E0E0', '#7393B3'])

    plt.figure(figsize=(12, 8))
    plt.imshow(presence_matrix, aspect='auto', cmap=cmap, interpolation='none')
    
    # Colorbar with custom ticks
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Empty', 'Non-empty'])

    # X-axis labels = sorted features
    plt.xticks(ticks=np.arange(len(sorted_features)), labels=sorted_features, rotation=90)

    # Hide per-row y tick labels but keep the axis label
    plt.yticks([])
    plt.ylabel("Patient ID")

    #plt.title("Per-Patient Feature Presence Map (Features Sorted by Prevalence)")
    plt.tight_layout()
    plt.show()

    # Print the feature prevalence (in ascending order)
    print("Feature Prevalence (percentage of patients with at least one data point):")
    for feature in feature_prevalence_sorted.index:
        print(f"{feature}: {feature_prevalence[feature]*100:.2f}%")