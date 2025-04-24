import pandas as pd
from pathlib import Path
from feature_engineering_V2 import find_project_root


def vectorization(df, time_col="ICULOS", id_col="patient_id"):
    df_original = df.copy()
    df = df.copy()
    raw_cols = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets",
        "HospAdmTime"]
    drop_cols = [col for col in raw_cols if col in df.columns]
    df = df.drop(columns=drop_cols) 

    df["relative_step"] = df.groupby(id_col).cumcount() + 1
    df = df.set_index([id_col, "relative_step"])
    df_vector = df.drop(columns=[time_col], errors="ignore").unstack("relative_step")
    df_vector.columns = [f"{col}_{step}" for col, step in df_vector.columns]
    df_vector = df_vector.reset_index()
    iculos_last = df_original.groupby(id_col)["ICULOS"].max().rename("ICULOS_last")
    df_vector = df_vector.merge(iculos_last, on=id_col, how="left")
    sepsis_labels = df_original.groupby(id_col)["SepsisLabel"].max().rename("SepsisLabel")
    df_vector = df_vector.merge(sepsis_labels, on=id_col, how="left")
    df_vector = df_vector[[col for col in df_vector.columns if not (col.startswith("SepsisLabel_") and col != "SepsisLabel")]]
    return df_vector


def clean_vectorized_output(df_vector):
    keep_static = ["patient_id", "SepsisLabel", "Age", "Gender", "ICULOS_last"]
    stat_keywords = ["_mean_", "_std_", "_min_", "_max_", "_median_", "_last_", "_first_"]
    keep_dynamic = [col for col in df_vector.columns if any(kw in col for kw in stat_keywords)]
    keep_cols = keep_static + keep_dynamic
    return df_vector[[col for col in keep_cols if col in df_vector.columns]]


if __name__ == "__main__":
    project_root = find_project_root()
    input_path = f"{project_root}/dataset/V2_preprocessed.parquet"
    output_path = f"{project_root}/dataset/after-vectorization.parquet"

    df = pd.read_parquet(input_path)
    print(f"Loaded dataset: {input_path} (shape: {df.shape})")

    df_vectorized = vectorization(df)
    df_cleaned = clean_vectorized_output(df_vectorized)

    df_cleaned.to_parquet(output_path)
    print(f"Vectorized dataset saved to: {output_path} (shape: {df_cleaned.shape})")
