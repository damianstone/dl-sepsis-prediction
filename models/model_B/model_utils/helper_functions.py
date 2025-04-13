import os
from pathlib import Path

import yaml

FEATURE_NAMES = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "BaseExcess",
    "HCO3",
    "FiO2",
    "pH",
    "PaCO2",
    "SaO2",
    "AST",
    "BUN",
    "Alkalinephos",
    "Calcium",
    "Chloride",
    "Creatinine",
    "Bilirubin_direct",
    "Glucose",
    "Lactate",
    "Magnesium",
    "Phosphate",
    "Potassium",
    "Bilirubin_total",
    "TroponinI",
    "Hct",
    "Hgb",
    "PTT",
    "WBC",
    "Fibrinogen",
    "Platelets",
    "Age",
    "Gender",
    "HospAdmTime",
    "ICULOS",
    "SOFA",
]


def save_xperiment_csv(root, xperiment_name, df):
    save_path = f"{root}/models/model_B/results/{xperiment_name}"
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(
        f"{root}/models/model_B/results/{xperiment_name}/{xperiment_name}.csv",
        index=False,
    )


def save_xperiment_yaml(root, config):
    xperiment_name = config["xperiment"]["name"]
    save_path = f"{root}/models/model_B/results/{xperiment_name}/xperiment.yml"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as file:
        yaml.dump(
            config,
            file,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=70,
        )


def display_balance_statistics(df):
    patient_df = df.groupby("patient_id")["SepsisLabel"].max().reset_index()
    counts = patient_df["SepsisLabel"].value_counts()
    total_patients = counts.sum()
    print("Patient-level balance statistics:")
    print("Total patients:", total_patients)
    for label, count in counts.items():
        perc = (count / total_patients) * 100
        print(f"Label {label}: {count} patients ({perc:.2f}%)")
    if len(counts) >= 2:
        imbalance_ratio = counts.max() / counts.min()
        print(f"Imbalance ratio (majority/minority): {imbalance_ratio:.2f}")


def find_project_root(marker=".gitignore"):
    """
    walk up from the current working directory until a directory containing the
    specified marker (e.g., .gitignore) is found.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}"
    )
