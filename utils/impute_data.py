import numpy as np
import pandas as pd

def impute_linear_interpolation(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Interpolates missing values in 'col' using a linear method,
    sorting the DataFrame by 'ICULOS' for time consistency.
    """
    df = df.sort_values("ICULOS").copy()
    df[col] = df[col].interpolate(method='linear')
    return df

def compute_bin_edges(series: pd.Series, n_bins: int) -> np.ndarray:
    """
    Computes unique percentile-based bin edges for a given series.
    """
    quantiles = [i / n_bins for i in range(n_bins + 1)]
    edges = series.quantile(quantiles).unique()
    return np.sort(edges)

def get_bin(value: float, edges: np.ndarray) -> int:
    """
    Returns the bin index for 'value' given bin edges, or -1 if 'value' is NaN.
    """
    if pd.isna(value):
        return -1
    idx = np.searchsorted(edges, value, side='right') - 1
    return max(0, min(idx, len(edges) - 2))

def cluster_mean_imputation(df: pd.DataFrame, col: str) -> tuple:
    """
    Imputes NaN values in 'col' using:
      1. The mean of the current cluster (if available),
      2. Otherwise, the mean of the nearest cluster (using Hamming distance).
    Returns the updated DataFrame along with counts for each fill type.
    """
    cluster_means = df.groupby("cluster_id")[col].mean()
    valid_cluster_means = {cid: mean for cid, mean in cluster_means.items() if not pd.isna(mean)}
    cluster_count = 0
    nearest_cluster_count = 0

    def parse_cluster_id(cid: str) -> list:
        return [int(x) if x != "X" else -1 for x in cid.split("_")]

    def hamming_distance(cid1: list, cid2: list) -> int:
        return sum(a != b for a, b in zip(cid1, cid2))

    def fill_func(row: pd.Series):
        nonlocal cluster_count, nearest_cluster_count
        if pd.isna(row[col]):
            cid = row["cluster_id"]
            own_mean = cluster_means.get(cid, np.nan)
            if not pd.isna(own_mean):
                cluster_count += 1
                return own_mean
            # Find the nearest cluster mean by Hamming distance
            cid_parsed = parse_cluster_id(cid)
            best_match, best_distance = None, float("inf")
            for other_cid, other_mean in valid_cluster_means.items():
                other_distance = hamming_distance(cid_parsed, parse_cluster_id(other_cid))
                if other_distance < best_distance:
                    best_distance = other_distance
                    best_match = other_mean
            if best_match is not None:
                nearest_cluster_count += 1
                return best_match
            return np.nan
        return row[col]

    df[col] = df.apply(fill_func, axis=1)
    return df, cluster_count, nearest_cluster_count

def impute_df_no_nans(
    df: pd.DataFrame,
    nan_density: float = 0.3,
    gender_bins: int = 2,
    age_bins: int = 10,
    hr_bins: int = 5,
    map_bins: int = 5,

) -> tuple:
    """
    Imputes missing values in the DataFrame using two approaches:
      1. Linear interpolation for columns with missing fraction below nan_density.
      2. Cluster-based imputation for columns with higher missing fraction.
      
    A cluster_id is assigned for each patient based on binned features:
      [Gender, Age, HR, MAP, O2Sat, SBP, Resp]
      
    If both cluster and nearest cluster values are missing, the value remains NaN.
    """
    df_imputed = df.copy()

    # Identify numeric columns to impute (exclude static columns)
    exclude_cols = ["patient_id", "dataset", "SepsisLabel", "ICULOS", "Age", "Gender", "HospAdmTime", "Unit1", "Unit2"]
    candidate_cols = [
        c for c in df_imputed.select_dtypes(include=[np.number]).columns
        if c not in exclude_cols
    ]

    replacement_stats = {}
    lin_cols, cluster_cols = [], []

    # Decide which columns get linear vs cluster fill
    for col in candidate_cols:

        if df_imputed[col].isna().mean() < nan_density:
            lin_cols.append(col)
        else:
            cluster_cols.append(col)

    # Linear Interpolation for columns in lin_cols
    for col in lin_cols:
        missing_before = df_imputed[col].isna().sum()
        df_imputed = df_imputed.groupby("patient_id").apply(
            lambda g: impute_linear_interpolation(g, col)
        ).reset_index(drop=True)
        # Fill leftover NaNs with mean per patient, then global mean if needed
        df_imputed[col] = df_imputed.groupby("patient_id")[col].transform(
            lambda x: x.fillna(x.mean())
        )
        df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mean(skipna=True))
        replacement_stats[col] = {
            "Method": "linear",
            "Initial missing": missing_before,
            "Linear fill": 100.00
        }

    # Binning parameters
    clustering_params = {
        "Gender": gender_bins,
        "Age": age_bins,
        "HR": hr_bins,
        "MAP": map_bins,

    }
    clustering_features = {feat: bins for feat, bins in clustering_params.items() if bins > 0}

    # Build bin edges from patient means
    if clustering_features:
        patient_means = df_imputed.groupby("patient_id")[list(clustering_features.keys())].mean().reset_index()
    else:
        patient_means = pd.DataFrame({"patient_id": df_imputed["patient_id"].unique()})
    bin_edges_dict = {}
    for feat, bins in clustering_features.items():
        feat_series = patient_means[feat].dropna()
        if not feat_series.empty:
            bin_edges_dict[feat] = compute_bin_edges(feat_series, n_bins=bins)
        else:
            bin_edges_dict[feat] = np.arange(0, bins + 1)

    # Assign cluster_id per patient
    def assign_cluster_id_for_patient(df_patient: pd.DataFrame) -> pd.DataFrame:
        df_patient = df_patient.copy()
        features_order = ["Gender", "Age", "HR", "MAP", "O2Sat", "SBP", "Resp"]
        parts = []
        for feat in features_order:
            if feat in clustering_features:
                if feat == "Gender":
                    val = (
                        int(df_patient[feat].iloc[0])
                        if not df_patient[feat].isna().all()
                        else -1
                    )
                    parts.append(str(val))
                else:
                    mean_val = df_patient[feat].mean(skipna=True)
                    parts.append(str(get_bin(mean_val, bin_edges_dict[feat])))
            else:
                parts.append("X")
        df_patient["cluster_id"] = "_".join(parts)
        return df_patient

    df_imputed = df_imputed.groupby("patient_id").apply(assign_cluster_id_for_patient).reset_index(drop=True)

    # Cluster-based imputation
    for col in cluster_cols:
        missing_before = df_imputed[col].isna().sum()
        df_imputed, clust_count, nearest_count = cluster_mean_imputation(df_imputed, col)
        replacement_stats[col] = {
            "Method": "cluster",
            "Initial missing": missing_before,
            "Cluster fill": (clust_count / missing_before * 100) if missing_before > 0 else 0,
            "Nearest cluster fill": (nearest_count / missing_before * 100) if missing_before > 0 else 0
        }

    return df_imputed, replacement_stats
