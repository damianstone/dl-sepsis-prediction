import os
import sys
from pathlib import Path

import pandas as pd
import torch
import yaml
from architectures import TransformerClassifier, TransformerTimeSeries
from custom_dataset import SepsisPatientDataset, collate_fn
from model_utils.helper_functions import save_xperiment_csv, save_xperiment_yaml
from model_utils.metrics import save_metrics
from model_utils.plots import save_plots
from preprocess import preprocess_data
from testing import testing_loop
from torch import nn
from torch.utils.data import DataLoader
from training import training_loop


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


def get_config(root, name_file):
    config_path = f"{root}/models/model_B/config/{name_file}"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def save_model(xperiment_name, model):
    model_path = Path("./saved")
    model_path.mkdir(exist_ok=True)
    model_file = model_path / f"{xperiment_name}.pth"
    torch.save(model.state_dict(), model_file)


def print_imbalance_ratio(y, patient_ids, dataset_name):
    df = pd.DataFrame({"patient_id": patient_ids, "SepsisLabel": y})

    patient_summary = df.groupby("patient_id")["SepsisLabel"].max()
    counts = patient_summary.value_counts()
    total_patients = counts.sum()

    print(f"Dataset: {dataset_name}")
    print("Patient-level balance statistics:")
    print("Total patients:", total_patients)
    for label, count in counts.items():
        perc = (count / total_patients) * 100
        print(f"Label {label}: {count} patients ({perc:.2f}%)")
    if len(counts) >= 2:
        imbalance_ratio = counts.max() / counts.min()
        print(f"Imbalance ratio (majority/minority): {imbalance_ratio:.2f}")
    print("-" * 50)


def get_pos_weight(ids, y, max_weight=5, device=torch.device("cpu")):
    train_df = pd.DataFrame({"patient_id": ids, "SepsisLabel": y})
    patient_summary = train_df.groupby("patient_id")["SepsisLabel"].max().reset_index()
    negative_count = (patient_summary["SepsisLabel"] == 0).sum()
    positive_count = (patient_summary["SepsisLabel"] == 1).sum()

    if positive_count == 0:
        raise ValueError("No positive samples found in training set.")

    weight = round(negative_count / positive_count, 2)
    p_w = min(weight, max_weight)
    return weight, torch.tensor([p_w], dtype=torch.float32, device=device)


def data_plots_and_metrics(
    root,
    config,
    all_y_logits,
    all_y_probs,
    all_y_pred,
    all_y_test,
    epoch_counter,
    loss_counter,
    acc_counter,
    model,
    feature_names,
):
    xperiment_name = config["xperiment"]["name"]
    all_y_logits = torch.cat(all_y_logits).numpy().flatten()
    all_y_probs = torch.cat(all_y_probs).numpy().flatten()
    all_y_pred = torch.cat(all_y_pred).numpy().flatten()
    all_y_test = torch.cat(all_y_test).numpy().astype(int)
    df = pd.DataFrame(
        {
            "y_logits": all_y_logits,
            "y_probs": all_y_probs,
            "y_pred": all_y_pred,
            "y_test": all_y_test,
        }
    )
    y_test = df["y_test"].values
    y_probs = df["y_probs"].values
    y_pred = df["y_pred"].values
    try:
        save_xperiment_csv(root, xperiment_name, df)
    except Exception as e:
        raise RuntimeError("Failed to save experiment CSV") from e

    try:
        save_xperiment_yaml(root, config)
    except Exception as e:
        raise RuntimeError("Failed to save experiment config") from e

    try:
        save_metrics(root, xperiment_name, y_test, y_probs, y_pred)
    except Exception as e:
        raise RuntimeError("Failed to save metrics") from e

    try:
        save_plots(
            root,
            xperiment_name,
            loss_counter,
            y_test,
            y_probs,
            y_pred,
            model,
            feature_names,
        )
    except Exception as e:
        raise RuntimeError("Failed to save plots") from e


def get_model(model_to_use, config, in_dim, device):
    if model_to_use == "time_series":
        model = TransformerTimeSeries(
            input_dim=in_dim,
            n_heads=config["model"]["num_heads"],
            d_model=config["model"]["d_model"],
            n_layers=config["model"]["num_layers"],
            dropout=config["model"]["drop_out"],
        ).to(device)
    else:
        model = TransformerClassifier(
            input_dim=in_dim,
            n_heads=config["model"]["num_heads"],
            drop_out=config["model"]["drop_out"],
            num_layers=config["model"]["num_layers"],
        ).to(device)
    return model


def full_pipeline():
    project_root = find_project_root()
    if project_root not in sys.path:
        sys.path.append(project_root)
    if len(sys.argv) < 2:
        print("Usage: python full_pipeline.py <config_name_file>")
        sys.exit(1)
    config_name_file = sys.argv[1]
    config = get_config(project_root, config_name_file)

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
    data_config = config["data"]
    data_splits = preprocess_data(**data_config)

    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]
    patient_ids_train = data_splits["patient_ids_train"]
    print_imbalance_ratio(y_train, patient_ids_train, "Training set")

    X_val = data_splits["X_val"]
    y_val = data_splits["y_val"]
    patient_ids_val = data_splits["patient_ids_val"]
    print_imbalance_ratio(y_val, patient_ids_val, "Validation set")

    X_test = data_splits["X_test"]
    y_test = data_splits["y_test"]
    patient_ids_test = data_splits["patient_ids_test"]
    print_imbalance_ratio(y_test, patient_ids_test, "Test set")

    batch_size = config["training"]["batch_size"]
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

    # -------------------------------- MODEL --------------------------------
    in_dim = X_train.shape[1]
    config["model"]["input_dimention"] = in_dim

    model = get_model(
        model_to_use=config["xperiment"]["model"],
        config=config,
        in_dim=in_dim,
        device=device,
    )

    # NOTE: get pos_weight to balance the loss for imbalanced classes
    if config["training"]["use_post_weight"]:
        max_p = config["training"]["max_post_weight"]
        weight, pos_weight = get_pos_weight(
            patient_ids_train, y_train.values, max_p, device
        )
        config["training"]["weight"] = float(weight)
        config["training"]["post_weight"] = float(pos_weight)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"])

    # -------------------------------- TRAINING LOOP --------------------------------
    epoch_counter, loss_counter, acc_counter = training_loop(
        experiment_name=config["xperiment"]["name"],
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=config["training"]["epochs"],
        device=device,
    )

    # -------------------------------- TESTING LOOP --------------------------------
    batch_size = config["testing"]["batch_size"]
    dataset = SepsisPatientDataset(
        X_test.values,
        y_test.values,
        patient_ids_test.values,
        time_index=X_test.columns.get_loc("ICULOS"),
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
    )

    # -------------------------------- METRICS AND PLOTS --------------------------------

    model_path = (
        f"{project_root}/models/model_B/saved/{config['xperiment']['name']}.pth"
    )
    model.load_state_dict(torch.load(model_path))
    all_y_logits, all_y_probs, all_y_pred, all_y_test = testing_loop(
        model=model,
        test_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
        threshold=0.5,
    )

    data_plots_and_metrics(
        project_root,
        config,
        all_y_logits,
        all_y_probs,
        all_y_pred,
        all_y_test,
        epoch_counter,
        loss_counter,
        acc_counter,
        model,
        feature_names=X_train.columns.tolist(),
    )


if __name__ == "__main__":
    full_pipeline()
