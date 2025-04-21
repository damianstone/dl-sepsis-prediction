"""
Grid Search Script for Sepsis Prediction

This script performs a hyperparameter grid search for a time series transformer model to predict sepsis onset. It includes:
  - Configuration setup
  - Device setup
  - Loss and metric functions
  - GridSearchModel and DataWrapper classes
  - Grid search execution and pipeline orchestration

Usage:
    python grid_search.py
"""

import copy
import os
import random
import sys

import numpy as np
import torch
from custom_dataset import SepsisPatientDataset, collate_fn
from full_pipeline import data_plots_and_metrics, get_model, get_pos_weight
from testing import testing_loop
from torch import nn
from torch.utils.data import DataLoader
from training import (
    delete_model,
    get_f1_score,
    get_f2_score,
    save_model,
    training_loop,
    validation_loop,
)

# Set up project path
file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(file_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from final_dataset_scripts.dataset_loader import (
    load_test_data,
    load_train_data,
    load_val_data,
)

# Fixed random seed for reproducibility
RANDOM_STATE = 42

# Import local modules directly

# Import from final_dataset_scripts

# ============================================================================
# Configuration and Utility Functions
# ============================================================================


def set_seeds(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (if available)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except AttributeError:
        pass


def setup_base_config():
    """Set up the base configuration for experiments including model name, training, and testing parameters."""
    return {
        "xperiment": {
            "name": "time_series_transformer_grid_search",
            "model": "time_series",
        },
        "training": {
            "batch_size": 256,
            "use_post_weight": True,
            "max_post_weight": 5,
            "lr": 0.0001,
            "epochs": 1000,
            "weight_decay": 0.02,
        },
        "testing": {
            "batch_size": 256,
            "threshold": 0.5,
        },
    }


def setup_device():
    """Determine and return the available device (MPS, CUDA, or CPU) for PyTorch computations."""
    device_type = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using {device_type.upper()} device")
    return torch.device(device_type)


def get_loss_fn(config, train_data, device):
    """Get the binary cross entropy loss function, optionally applying a positive class weight.

    Args:
        config (dict): Configuration dictionary with training settings.
        train_data (DataWrapper): Training data wrapper.
        device (torch.device): Device for computing pos_weight.

    Returns:
        torch.nn.Module: Loss function with optional pos_weight.
    """
    if config["training"]["use_post_weight"]:
        _, pos_weight = get_pos_weight(
            train_data.patient_ids,
            train_data.y,
            config["training"]["max_post_weight"] + 1,
            device,
        )
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


# ============================================================================
# GridSearchModel Class
# ============================================================================


class GridSearchModel:
    def __init__(self, config, device, train_data, val_data, in_dim):
        """Initialize the grid search model with configuration, device, and data."""
        self.config = config
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.val_loader = val_data.loader
        self.model = get_model(
            model_to_use=config["xperiment"]["model"],
            config=config,
            in_dim=in_dim,
            device=device,
        )
        self.f2_score = 0
        self.f1_score = 0
        self.model_name = config["xperiment"]["name"]
        self.loss_fn = get_loss_fn(config, train_data, device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )

    def train_and_evaluate(self):
        """Train the model and evaluate on validation data, storing performance metrics."""
        res = training_loop(
            self.model_name,
            self.model,
            self.train_data.loader,
            self.val_data.loader,
            self.optimizer,
            self.loss_fn,
            self.config["training"]["epochs"],
            self.device,
        )
        self.epoch_counter = res["epoch_counter"]
        self.loss_counter = res["loss_counter"]
        self.acc_counter = res["acc_counter"]
        self.model = res["model"]

        _, _, _, _, _, y_pred, y_true = validation_loop(
            self.model,
            self.val_loader,
            self.loss_fn,
            self.device,
            self.config["testing"]["threshold"],
        )
        self.f2_score = get_f2_score(y_pred, y_true)
        # Also compute F1-score
        self.f1_score = get_f1_score(y_pred, y_true)

    def test_model(self, test_loader):
        """Test the trained model on test data and generate plots and metrics."""
        all_y_logits, all_y_probs, all_y_pred, all_y_test = testing_loop(
            model=self.model,
            test_loader=test_loader,
            loss_fn=self.loss_fn,
            device=self.device,
            threshold=0.5,
        )

        data_plots_and_metrics(
            project_root,
            self.config,
            all_y_logits,
            all_y_probs,
            all_y_pred,
            all_y_test,
            self.epoch_counter,
            self.loss_counter,
            self.acc_counter,
            self.model,
            feature_names=self.train_data.X.columns.tolist(),
        )

    def delete(self):
        """Delete saved model files associated with this experiment."""
        delete_model(self.model_name)

    def save(self):
        """Save the current model state with the experiment name."""
        save_model(self.model_name, self.model)


# ============================================================================
# DataWrapper Class
# ============================================================================


class DataWrapper:
    """Wrapper for dataset components including features, labels, patient IDs, and DataLoader."""

    def __init__(self, X, y, patient_ids, dataset, loader):
        self.X = X
        self.y = y
        self.patient_ids = patient_ids
        self.dataset = dataset
        self.loader = loader

    @staticmethod
    def from_map(map):
        """Create a DataWrapper instance from a mapping containing X, y, patient_ids, dataset, and loader."""
        return DataWrapper(
            map["X"], map["y"], map["patient_ids"], map["dataset"], map["loader"]
        )


# ============================================================================
# Data Loading Function
# ============================================================================


def get_data(config, type):
    """Load and wrap train, val, or test data based on the config and dataset type.

    Args:
        config (dict): Configuration dict including batch sizes and dataset type.
        type (str): One of 'train', 'val', or 'test'.

    Returns:
        DataWrapper: Wrapped data with DataLoader.
    """
    if type == "train":
        data = load_train_data(config["dataset_type"])
    elif type == "val":
        data = load_val_data()
    elif type == "test":
        data = load_test_data()
    else:
        raise ValueError(f"Unknown data type: {type}")

    dataset = SepsisPatientDataset(
        data["X"].values,
        data["y"].values,
        data["patient_ids"].values,
        time_index=data["X"].columns.get_loc("ICULOS"),
    )

    loader = DataLoader(
        dataset,
        batch_size=config["testing" if type == "test" else "training"]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=12,
    )

    return DataWrapper.from_map(
        {
            "X": data["X"],
            "y": data["y"],
            "patient_ids": data["patient_ids"],
            "dataset": dataset,
            "loader": loader,
        }
    )


# ============================================================================
# Grid Search Execution Functions
# ============================================================================


def save_best_models(best_models):
    """Save all models in the best_models dictionary."""
    for best_model in best_models.values():
        best_model.save()


def run_grid_search(config, device, train_data, val_data, in_dim) -> GridSearchModel:
    """Run hyperparameter grid search over specified dimensions and return the best model."""
    best_model = None
    iterations = 0
    total_iterations = 4

    num_heads = 4
    drop_out = 0.4
    for d_model in [128]:
        for num_layers in [4]:
            iterations += 1
            print(
                f"Running grid search: {iterations}/{total_iterations} " f"iterations"
            )
            experiment_name = f"{config['dataset_type']}_{iterations}"
            config_new = copy.deepcopy(config)
            config_new["xperiment"]["name"] = experiment_name
            config_new["model"] = {
                "d_model": d_model,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "drop_out": drop_out,
                "input_dimension": in_dim,
            }

            model = GridSearchModel(config_new, device, train_data, val_data, in_dim)
            model.train_and_evaluate()

            if best_model is None or model.f2_score > best_model.f2_score:
                print(f"New best model found: {model.f2_score}")
                if best_model:
                    best_model.delete()

                best_model = model
                best_model.save()

    if best_model is None:
        raise ValueError("No best model found")
    return best_model


# ============================================================================
# Main Pipeline
# ============================================================================


def pipeline():
    """Orchestrate the entire workflow: data loading, grid search per dataset type, testing, and returning best models."""
    config = setup_base_config()
    device = setup_device()
    val_data = get_data(config, "val")
    test_data = get_data(config, "test")
    best_models = {}
    for dataset_type in ["oversampled"]:
        print(f"Running grid search for {dataset_type}")
        config_new = copy.deepcopy(config)
        config_new["dataset_type"] = dataset_type

        train_data = get_data(config_new, "train")
        best_model = run_grid_search(
            config_new, device, train_data, val_data, train_data.X.shape[1]
        )
        best_model.test_model(test_data.loader)
        best_models[dataset_type] = best_model

    return best_models


if __name__ == "__main__":
    set_seeds()
    pipeline()
