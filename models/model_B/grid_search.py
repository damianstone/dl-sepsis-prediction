import os
import sys

import torch
from custom_dataset import SepsisPatientDataset, collate_fn
from full_pipeline import get_model, training_loop
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


def train_and_evaluate(config, model, train_loader, val_loader, device):
    training_loop(config, model, train_loader, val_loader, device)


def pipeline():
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

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")

    # -------------------------------- DATA SPLIT --------------------------------
    val_data, test_data = load_val_data(), load_test_data()

    X_val = val_data["X"]
    y_val = val_data["y"]
    patient_ids_val = val_data["patient_ids"]

    test_data["X"]
    test_data["y"]
    test_data["patient_ids"]

    batch_size = config["training"]["batch_size"]
    val_dataset = SepsisPatientDataset(
        X_val.values,
        y_val.values,
        patient_ids_val.values,
        time_index=X_val.columns.get_loc("ICULOS"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    datasets = ["imbalanced", "oversampled", "downsampled"]
    for dataset_type in datasets:
        train_data = load_train_data(dataset_type)
        X_train = train_data["X"]
        y_train = train_data["y"]
        patient_ids_train = train_data["patient_ids"]

        train_dataset = SepsisPatientDataset(
            X_train.values,
            y_train.values,
            patient_ids_train.values,
            time_index=X_train.columns.get_loc("ICULOS"),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        best_score = -float("inf")
        best_params = {}
        best_model = None
        results = {}
        for d_model in [64, 128, 256]:
            for num_heads in [2, 4, 8]:
                if d_model % num_heads != 0:
                    continue
                for num_layers in [1, 2, 3]:
                    for drop_out in [0.1, 0.2, 0.3]:
                        config["model"] = {}
                        config["model"]["d_model"] = d_model
                        config["model"]["num_heads"] = num_heads
                        config["model"]["num_layers"] = num_layers
                        config["model"]["drop_out"] = drop_out

                        in_dim = X_train.shape[1]
                        config["model"]["input_dimention"] = in_dim

                        model = get_model(
                            model_to_use=config["xperiment"]["model"],
                            config=config,
                            in_dim=in_dim,
                            device=device,
                        )

                        score = train_and_evaluate(
                            config, d_model, num_heads, num_layers, drop_out
                        )
                        print(
                            f"[{dataset_type}] d_model={d_model}, heads={num_heads},",
                            f"layers={num_layers}, drop={drop_out} ->",
                            f"val_score={score:.4f}",
                        )
                        if score > best_score:
                            best_score = score
                            best_params = {
                                "d_model": d_model,
                                "num_heads": num_heads,
                                "num_layers": num_layers,
                                "drop_out": drop_out,
                            }
                            best_model = model

        results[dataset_type] = {
            "score": best_score,
            "params": best_params,
            "model": best_model,
        }
