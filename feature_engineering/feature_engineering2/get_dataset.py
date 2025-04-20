from pathlib import Path
from balanced_data import create_balanced_dataset, split_balanced_dataset_maintain_ratio
from filter_correlated_features import filter_balanced_dataset_main
from feature_engineering import run_multiwindow_feature_engineering
from SHAP import run_shap_on_data


def run_full_pipeline():
    # print("Step 1: Balancing data with 30% positives")
    # create_balanced_dataset(input_file="Fully_imputed_dataset.parquet", output_file="balanced_dataset.parquet")
    # print("Balanced dataset created\n")

    # print("Step 2: Filtering highly correlated physiological features")
    # filter_balanced_dataset_main()
    # print("Correlated features removed\n")    
    
    # print("Step 3: Feature engineering for patient-level dataset")
    # run_multiwindow_feature_engineering()
    # print("Feature engineering complete\n")

    print("Step 4: Splitting into train/test with balanced label ratio")
    split_balanced_dataset_maintain_ratio(input_file="XGBoost/after_feature_engineering.parquet")
    print("Train/test split done\n")

    print("Step 5: SHAP analysis on engineered training set")
    run_shap_on_data(parquet_file="XGBoost/after_feature_engineering.parquet", output_tag="engineered")
    print("Final SHAP completed on patient-level features\n")


if __name__ == "__main__":
    run_full_pipeline()
