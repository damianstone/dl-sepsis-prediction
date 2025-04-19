import copy
import os
import sys

# Set up project path
file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(file_dir, "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import torch

# Import local modules directly
from custom_dataset import SepsisPatientDataset, collate_fn
from full_pipeline import data_plots_and_metrics, get_model, get_pos_weight
from sklearn.metrics import fbeta_score
from testing import testing_loop
from torch import nn
from torch.utils.data import DataLoader
from training import delete_model, save_model, training_loop, validation_loop

# Import from final_dataset_scripts
from final_dataset_scripts.dataset_loader import (
    load_test_data,
    load_train_data,
    load_val_data,
)


def setup_base_config():
    return {
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
            "threshold": 0.5,
            "device": "mps",
        },
    }


def setup_device():
    device_type = (
        "mps"
        if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using {device_type.upper()} device")
    return torch.device(device_type)


def get_loss_fn(config, train_data, device):
    if config["training"]["use_post_weight"]:
        _, pos_weight = get_pos_weight(
            train_data.patient_ids,
            train_data.y,
            config["training"]["max_post_weight"],
            device,
        )
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return nn.BCEWithLogitsLoss()


def get_f2_score(y_pred, y_true):
    return fbeta_score(y_true, y_pred, beta=2)


class GridSearchModel:
    def __init__(self, config, device, train_data, val_data, in_dim, model_name):
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
        self.model_name = model_name
        self.loss_fn = get_loss_fn(config, train_data, device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config["training"]["lr"]
        )

    def train_and_evaluate(self):
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
        self.best_loss = res["best_loss"]
        self.model = res["model"]

        _, _, _, _, y_pred, y_true = validation_loop(
            self.model,
            self.val_loader,
            self.loss_fn,
            self.device,
            self.config["testing"]["threshold"],
        )
        self.f2_score = get_f2_score(y_pred, y_true)

    def test_model(self, test_loader):
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
        delete_model(self.model_name)

    def save(self):
        save_model(self.model_name, self.model)


class DataWrapper:
    def __init__(self, X, y, patient_ids, dataset, loader):
        self.X = X
        self.y = y
        self.patient_ids = patient_ids
        self.dataset = dataset
        self.loader = loader

    @staticmethod
    def from_map(map):
        return DataWrapper(
            map["X"], map["y"], map["patient_ids"], map["dataset"], map["loader"]
        )


def get_data(config, type):
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
        drop_last=True,
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


def save_best_models(best_models):
    for best_model in best_models.values():
        best_model.save()


def run_grid_search(config, device, train_data, val_data, in_dim) -> GridSearchModel:
    best_model = None
    iterations = 0
    total_iterations = 4 * 3 * 3 * 3

    for d_model in [64, 128, 256]:
        for num_heads in [2, 4, 8]:
            if d_model % num_heads != 0:
                iterations += 3 * 3
                continue
            for num_layers in [1, 2, 3]:
                for drop_out in [0.1, 0.2, 0.3]:
                    iterations += 1
                    print(
                        f"Running grid search: {iterations}/{total_iterations} "
                        f"iterations"
                    )
                    model_name = f"{config['dataset_type']}_{iterations}"
                    config_new = copy.deepcopy(config)
                    config_new["model"] = {
                        "d_model": d_model,
                        "num_heads": num_heads,
                        "num_layers": num_layers,
                        "drop_out": drop_out,
                        "input_dimension": in_dim,
                    }

                    model = GridSearchModel(
                        config_new, device, train_data, val_data, in_dim, model_name
                    )
                    model.train_and_evaluate()

                    if best_model is None or model.f2_score > best_model.f2_score:
                        if best_model:
                            best_model.delete()
                        best_model = model
                        best_model.save()

    if best_model is None:
        raise ValueError("No best model found")
    return best_model


def pipeline():
    config = setup_base_config()
    device = setup_device()
    val_data = get_data(config, "val")
    test_data = get_data(config, "test")
    datasets = ["no_sampling", "oversampling", "undersampling"]
    best_models = {}
    for dataset_type in datasets:
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
    pipeline()
