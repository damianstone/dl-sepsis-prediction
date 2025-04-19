import os
import sys

import torch
from custom_dataset import SepsisPatientDataset, collate_fn
from full_pipeline import get_model, get_pos_weight, training_loop
from torch import nn
from torch.utils.data import DataLoader
from training import training_loop

file_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(file_dir, "../.."))
print
if project_root not in sys.path:
    sys.path.append(project_root)

from final_dataset_scripts.dataset_loader import (
    load_test_data,
    load_train_data,
    load_val_data,
)

"""
The goal here is to save the best parameters for each dataset type.
1. create the train and evaluate function
2. run the shit in the supercomputer
3. save the best parameters in a json file
"""


def setup_base_config():
    config = {
        "xperiment": {
            "name": "time_series_transformer_grid_search",
            "model": "time_series",
        },
        "training": {
            "batch_size": 32,
            "use_post_weight": True,
            "max_post_weight": 5,
            "lr": 0.001,
            "epochs": 1000,
        },
        "testing": {
            "batch_size": 32,
            "threshold": 0.3,
            "device": "mps",
        },
    }
    return config


def setup_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def get_loss_fn(config, train_data, device):
    if config["training"]["use_post_weight"]:
        max_p = config["training"]["max_post_weight"]
        weight, pos_weight = get_pos_weight(
            train_data["patient_ids"], train_data["y"], max_p, device
        )
        config["training"]["weight"] = float(weight)
        config["training"]["post_weight"] = float(pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    return loss_fn


class GridSearchModel:
    def __init__(self, config, device, train_data, val_data, in_dim, model_name):
        self.config = config
        self.device = device
        self.train_data = train_data
        self.val_data = val_data
        self.in_dim = in_dim
        self.model = get_model(
            model_to_use=config["xperiment"]["model"],
            config=config,
            in_dim=in_dim,
            device=device,
        )
        self.f2_score = 0
        self.model_name = model_name
        self.loss_fn = get_loss_fn(config, train_data, device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config["training"]["lr"]
        )

    def train_and_evaluate(self):
        res = training_loop(
            self.model_name,
            self.model,
            self.train_loader,
            self.val_loader,
            self.optimizer,
            self.loss_fn,
            self.config["training"]["epochs"],
            self.device,
        )
        self.epoch_counter = res["epoch_counter"]
        self.loss_counter = res["loss_counter"]
        self.acc_counter = res["acc_counter"]
        self.best_loss = res["best_loss"]
        self.model = res["model"]


class DataWrapper:
    def __init__(self, X, y, patient_ids, dataset, loader):
        self.X = X
        self.y = y
        self.patient_ids = patient_ids
        self.dataset = dataset
        self.loader = loader

    def from_map(map):
        return DataWrapper(
            map["X"], map["y"], map["patient_ids"], map["dataset"], map["loader"]
        )


def get_data(config, type):
    """
    type: "train", "val", "test"
    """
    data = {}
    if type == "train":
        data = load_train_data(config["dataset_type"])
    elif type == "val":
        data = load_val_data()
    elif type == "test":
        data = load_test_data()
    X = data["X"]
    y = data["y"]
    patient_ids = data["patient_ids"]
    dataset = SepsisPatientDataset(
        X.values, y.values, patient_ids.values, time_index=X.columns.get_loc("ICULOS")
    )
    training_or_testing = "testing" if type == "test" else "training"
    loader = DataLoader(
        dataset,
        batch_size=config[training_or_testing]["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )
    data["dataset"] = dataset
    data["loader"] = loader
    return DataWrapper.from_map(data)


def save_best_models(best_models):
    for dataset_type, best_model in best_models.items():
        best_model.save()


def run_grid_search(config, device, train_data, val_data, in_dim) -> GridSearchModel:
    best_model = None
    iterations = 0
    total_iterations = 4 * 3 * 3 * 3
    for d_model in [64, 128, 256]:
        for num_heads in [2, 4, 8]:
            if d_model % num_heads != 0:
                continue
            for num_layers in [1, 2, 3]:
                for drop_out in [0.1, 0.2, 0.3]:
                    iterations += 1
                    print(
                        f"Running grid search: {iterations} / {total_iterations} iterations"
                    )
                    model_name = f"{config['dataset_type']}_{iterations}"
                    config_new = config.copy()
                    config_new["model"] = {}
                    config_new["model"]["d_model"] = d_model
                    config_new["model"]["num_heads"] = num_heads
                    config_new["model"]["num_layers"] = num_layers
                    config_new["model"]["drop_out"] = drop_out
                    config_new["model"]["input_dimention"] = in_dim

                    model = GridSearchModel(
                        config_new, device, train_data, val_data, in_dim, model_name
                    )
                    model.train_and_evaluate()

                    if best_model is None or model.f2_score < best_model.f2_score:
                        best_model = model
    return best_model


def pipeline():
    config = setup_base_config()
    device = setup_device()
    val_data = get_data(config, "val")
    get_data(config, "test")

    datasets = ["imbalanced", "oversampled", "downsampled"]
    best_models = {}
    for dataset_type in datasets:
        print(f"Running grid search for {dataset_type}")
        config_new = config.copy()
        config_new["dataset_type"] = dataset_type

        train_data = get_data(config_new, "train")

        best_model = run_grid_search(config_new, device, train_data, val_data)
        best_models[dataset_type] = best_model

    save_best_models(best_models)
