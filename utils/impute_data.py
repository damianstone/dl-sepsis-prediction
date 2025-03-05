from sklearn.impute import KNNImputer
import pandas as pd
import numpy as np

def impute_missing_values_knn(data, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed_data = imputer.fit_transform(data)
    return pd.DataFrame(imputed_data, columns=data.columns)

def impute_linear_interpolation(data, column_name):
    imputed_data = data
    imputed_data[column_name] = data[column_name].interpolate(method='linear')
    return imputed_data

def impute_forward_fill_last_recorded(data, column_name, global_mean=None):
    imputed_data = data
    if pd.isnull(data[column_name].iloc[0]):
        imputed_data[column_name].iloc[0] = data[column_name].mean()
    imputed_data[column_name] = data[column_name].ffill()
    if global_mean != None:
        if pd.isnull(data[column_name].iloc[0]):
            imputed_data[column_name] = global_mean
    return imputed_data

    
def impute_patient_group(group, global_means, linear_cols, exclude_cols, missing_threshold):
    # Sort by ICULOS and make a copy to avoid SettingWithCopy warnings.
    group = group.sort_values('ICULOS').copy()
    
    for col in group.columns:
        if col in linear_cols:
            group[col] = group[col].interpolate(method='linear')
        elif col in exclude_cols:
            continue
        else:
            missing_pct = group[col].isnull().mean()
            if missing_pct < missing_threshold:
                # Forward fill then fill remaining NaNs with the global mean.
                group[col] = group[col].ffill().fillna(global_means[col])
            else:
                # Fill missing values with the patient’s mean if available, otherwise use global mean.
                patient_mean = group[col].mean()
                if pd.isnull(patient_mean):
                    patient_mean = global_means[col]
                group[col] = group[col].fillna(patient_mean)
    return group

def impute_df(df, missing_threshold=0.3):
    df_imputed = df.copy()
    # Select numeric columns for calculating global means.
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    global_means = df_imputed[numeric_cols].mean(skipna=True)
    
    # Define which columns should use linear interpolation and which to exclude.
    linear_cols = ['HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp']
    exclude_cols = ['patient_id', 'dataset', 'SepsisLabel', 'ICULOS']
    
    # Apply the imputation function to each patient group.
    df_imputed = df_imputed.groupby('patient_id').apply(
        lambda group: impute_patient_group(group, global_means, linear_cols, exclude_cols, missing_threshold)
    )
    
    # When using groupby.apply, the patient_id may become part of the index. Reset it if needed.
    if 'patient_id' not in df_imputed.columns:
        df_imputed = df_imputed.reset_index(level=0, drop=True)
        
    return df_imputed


