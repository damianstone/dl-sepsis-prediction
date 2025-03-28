from pathlib import Path
import pandas as pd
import numpy as np

def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError("Project root not found")

def compute_sofa_score(row):
    score = 0
    if pd.notna(row.get("MAP")) and row["MAP"] < 70:
        score += 1
    if pd.notna(row.get("Creatinine")):
        if row["Creatinine"] >= 5.0:
            score += 4
        elif row["Creatinine"] >= 3.5:
            score += 3
        elif row["Creatinine"] >= 2.0:
            score += 2
        elif row["Creatinine"] >= 1.2:
            score += 1
    if pd.notna(row.get("Platelets")):
        if row["Platelets"] < 20:
            score += 4
        elif row["Platelets"] < 50:
            score += 3
        elif row["Platelets"] < 100:
            score += 2
        elif row["Platelets"] < 150:
            score += 1
    if pd.notna(row.get("Bilirubin_total")):
        if row["Bilirubin_total"] >= 12.0:
            score += 4
        elif row["Bilirubin_total"] >= 6.0:
            score += 3
        elif row["Bilirubin_total"] >= 2.0:
            score += 2
        elif row["Bilirubin_total"] >= 1.2:
            score += 1
    if pd.notna(row.get("FiO2")) and pd.notna(row.get("PaCO2")) and row["FiO2"] > 0:
        ratio = row["PaCO2"] / row["FiO2"]
        if ratio < 100:
            score += 4
        elif ratio < 200:
            score += 3
        elif ratio < 300:
            score += 2
        elif ratio < 400:
            score += 1
    return score

def add_features(group, feature_cols):
    group = group.sort_values("ICULOS")

    sofa_series = group.apply(compute_sofa_score, axis=1)
    sofa_df = pd.DataFrame({"SOFA": sofa_series})
    for d in [1, 3, 5]:
        sofa_df[f"SOFA_diff_{d}"] = sofa_series.diff(d)

    new_feature_dfs = [sofa_df]

    for col in feature_cols:
        data = group[col]
        feats = {}

        for d in [1, 3, 5]:
            feats[f"{col}_diff_{d}"] = data.diff(d)

        for lag in range(1, 6):
            feats[f"{col}_lag_{lag}"] = data.shift(lag)

        feats[f"{col}_rollmean_5"] = data.rolling(window=5, min_periods=1).mean()
        feats[f"{col}_rollstd_5"] = data.rolling(window=5, min_periods=1).std()

        new_feature_dfs.append(pd.DataFrame(feats, index=group.index))

    new_features = pd.concat(new_feature_dfs, axis=1)

    group = pd.concat([group, new_features], axis=1)

    return group


def run_feature_engineering(input_file="balanced_dataset.parquet", output_file="balanced_dataset_with_features.parquet"):
    root = find_project_root()
    data_path = root / "dataset" / "XGBoost" / "feature_engineering" / input_file
    out_path = root / "dataset" / "XGBoost" / "feature_engineering" / output_file

    df = pd.read_parquet(data_path).reset_index()

    feature_cols = [
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
        "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total",
        "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen", "Platelets"
    ]

    df_feat = df.groupby("patient_id").apply(lambda g: add_features(g, feature_cols))
    df_feat = df_feat.reset_index(drop=True)
    df_feat.set_index(["patient_id", "ICULOS"], inplace=True)
    df_feat.to_parquet(out_path)

    print(f"feature engineering done, save to: {out_path}")

if __name__ == "__main__":
    run_feature_engineering()
