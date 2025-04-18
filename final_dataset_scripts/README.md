# Final Dataset Scripts

## Instructions
1. Pre-req: Make sure the feature engineering dataset `preprocessed.parquet` file exists by running the `feature_engineering/feature_engineering_data.ipynb` notebook if you haven't already
2. Run the `final_dataset_processing.ipynb` notebook to generate the final datasets
3. Use the `dataset_loader.py` to load the datasets within your code

# Explanation

The`final_dataset_processing.ipynb` notebook splits the `preprocessed.parquet` data into train, val and test sets and applies various sampling methods for the train set: no_sampling, upsampling and downsampling

The `dataset_loader.py` file returns the given dataset in a dictionary format with the following keys : X, y, patient_ids
