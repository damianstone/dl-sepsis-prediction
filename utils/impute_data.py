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

    
def impute_df(df, missing_threshold=0.3):
    df_imputed = df.copy()
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    global_means = df_imputed[numeric_cols].mean(skipna=True)
    linear_cols = ['HR', 'O2Sat', 'SBP', 'MAP', 'DBP', 'Resp']
    exclude_cols = ['patient_id', 'dataset', 'SepsisLabel', 'ICULOS']
    for pid, group in df_imputed.groupby('patient_id'):
        group = group.sort_values('ICULOS')
        for col in group.columns:
            if col in linear_cols:
                group = impute_linear_interpolation(group, col)
            elif col in exclude_cols:
                continue
            else:
                missing_pct = group[col].isnull().mean()
                if missing_pct < missing_threshold:
                    group = impute_forward_fill_last_recorded(group, col, global_mean=global_means[col])
                else:
                    patient_mean = group[col].mean()
                    if pd.isnull(patient_mean):
                        patient_mean = global_means[col]
                    group[col] = group[col].fillna(patient_mean)
        df_imputed.loc[group.index, :] = group
    
    return df_imputed


