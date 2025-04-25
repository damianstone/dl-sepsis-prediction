"""
SHAP Value Computation for Sepsis Prediction Model

This script computes SHAP (SHapley Additive exPlanations) values for model interpretability. It:
  - Prepares background data from training set with balanced patient samples
  - Creates evaluation dataset from test data
  - Loads a trained model
  - Computes and saves SHAP values for model interpretation

SHAP values help explain model predictions by showing the contribution of each feature
to the prediction for individual patients.
"""

# ============================================================================
# Configuration and Constants
# ============================================================================
SEED = 42
QUICK_DEBUG = False
MEDIUM_DEBUG = False
RANDOM_STATE = SEED
OUTPUT_FOLDER_NAME = "final_datasets/shap"
BACKGROUND_SIZE = 256

# ============================================================================
# Imports
# ============================================================================
import copy
import json
import os

from explain_model_helpers import ShapModel, get_config
from final_pipeline import ModelWrapper, get_data, setup_device
from full_pipeline import find_project_root
from preprocess import over_under_sample

# ============================================================================
# Setup and Initialization
# ============================================================================
project_root = find_project_root()
data_output_folder = f"{project_root}/dataset/{OUTPUT_FOLDER_NAME}"
if not os.path.exists(data_output_folder):
    os.makedirs(data_output_folder)

project_root = find_project_root()
results_name = "medium_model_no_sampling"
config = get_config(project_root, results_name)

device = setup_device()
train_data = get_data(config, "train")
val_data = get_data(config, "val")
test_data = get_data(config, "test")
in_dim = train_data.X.shape[1]

# ============================================================================
# Background Data Preparation
# ============================================================================
background_df = copy.deepcopy(train_data.df)

background_patient_df = (
    background_df.groupby("patient_id")["SepsisLabel"].max().reset_index()
)

# Count patients in each group.
counts = background_patient_df["SepsisLabel"].value_counts()

neg_patients = background_patient_df[background_patient_df["SepsisLabel"] == 0]
pos_patients = background_patient_df[background_patient_df["SepsisLabel"] == 1]
sampled_neg_patients = neg_patients.sample(
    n=BACKGROUND_SIZE // 2, replace=False, random_state=SEED
)
sampled_pos_patients = pos_patients.sample(
    n=BACKGROUND_SIZE // 2, replace=False, random_state=SEED
)

background_df = background_df[
    background_df["patient_id"].isin(sampled_neg_patients["patient_id"])
    | background_df["patient_id"].isin(sampled_pos_patients["patient_id"])
]

print(background_df.shape)
print(background_df.groupby("patient_id")["SepsisLabel"].max().value_counts())

background_df.to_parquet(f"{data_output_folder}/background.parquet")

# save the patient ids as json
patient_ids = background_df["patient_id"].unique()
with open(f"{data_output_folder}/background_patient_ids.json", "w") as f:
    json.dump(patient_ids.tolist(), f)

# ============================================================================
# Evaluation Data Preparation
# ============================================================================
eval_df = copy.deepcopy(test_data.df)
eval_df = over_under_sample(
    eval_df, method="undersample", minority_ratio=0.5, random_state=SEED
)
eval_df.to_parquet(f"{data_output_folder}/eval.parquet")

patient_ids = eval_df["patient_id"].unique()
with open(f"{data_output_folder}/eval_patient_ids.json", "w") as f:
    json.dump(patient_ids.tolist(), f)

# ============================================================================
# SHAP Model Computation
# ============================================================================
background_data = get_data(config, "shap_background")
eval_data = get_data(config, "shap_eval")

shap_output_folder = (
    f"{project_root}/models/model_B/results/{config['xperiment']['name']}/shap"
)

model = ModelWrapper(config, device, in_dim)
model.load_saved_weights()

shap_model = ShapModel(model.model, background_data, device, pad_value=0.0)
shap_vals, masks = shap_model.get_shap_values(eval_data)  # (256, 400, 107)
shap_model.save(shap_output_folder)
