from pathlib import Path
import pandas as pd
import numpy as np

PHYSIO_FEATURES = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
    "Fibrinogen", "Platelets"
]

def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(f"Project root marker '{marker}' not found starting from {current}")

def compute_missing_features(df):
    result = df.copy()
    for col in PHYSIO_FEATURES:
        if col not in df.columns:
            continue
        series = df[col]
        is_missing = series.isna().astype(int)
        interval = []
        last_seen = None
        for i, val in enumerate(series):
            if pd.notna(val):
                last_seen = i
                interval.append(0)
            else:
                interval.append(i - last_seen if last_seen is not None else -1)
        result[f"{col}_ismissing"] = is_missing
        result[f"{col}_interval"] = interval
    return result

def run_missing_feature_pipeline(input_file="raw_combined_data.parquet",output_file="features_with_missing_interval.parquet"):
    root = find_project_root()
    input_path = root / "dataset" / input_file
    df = pd.read_parquet(input_path)

    print("First 5 rows of original data:")
    print(df.head())


    if not isinstance(df.index, pd.MultiIndex):
        df.set_index(["patient_id", "ICULOS"], inplace=True)

    df = df.drop(columns=["dataset", "Unit1", "Unit2"], errors="ignore")

    # missing + interval
    features = df.groupby(level=0).apply(lambda p: compute_missing_features(p.droplevel(0)))

    # save
    output_path = root / "dataset" / "XGBoost"/"feature_engineering"
    output_path.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path / output_file, index=True)

    print("\nFirst 5 rows of processed features:")
    print(features.head())
    print(f"\nFeatures saved to {output_path / output_file}")

if __name__ == "__main__":
    run_missing_feature_pipeline()
