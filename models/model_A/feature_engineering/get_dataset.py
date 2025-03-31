
from pathlib import Path
from balanced_data import create_balanced_dataset, split_balanced_dataset_maintain_ratio
from missing_features import run_missing_feature_pipeline
from feature_engineering import run_feature_engineering
from SHAP import run_shap_on_data, export_single_shap_to_csv


def run_full_pipeline():
    print("Step 1: Balancing data with 20% positives")
    create_balanced_dataset(input_file="raw_combined_data.parquet", output_file="balanced_dataset.parquet")
    print("\ndone")

    print("\nStep 2: Generating missing feature indicators")
    run_missing_feature_pipeline(input_file="balanced_dataset.parquet", output_file="features_with_missing_interval.parquet")
    print("\ndone")

    print("\nStep 3: Applying feature engineering")
    run_feature_engineering(input_file="features_with_missing_interval.parquet", output_file="after_feature_engineering.parquet")
    print("\ndone")

    print("\nStep 4: Splitting into train/test with balanced label ratio")
    split_balanced_dataset_maintain_ratio(input_file="after_feature_engineering.parquet")
    print("\ndone")

    print("\nStep 5: Running SHAP analysis on training set")
    shap_values, features, sample_index, output_path = run_shap_on_data(parquet_file="train_balanced.parquet")
    export_single_shap_to_csv(shap_values, features, sample_index, output_path)
    print("\ndone")

if __name__ == "__main__":
    run_full_pipeline()
