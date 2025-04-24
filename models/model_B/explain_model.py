import os
import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
import yaml
from custom_dataset import collate_fn
from final_pipeline import ModelWrapper, get_data, get_loss_fn, set_seeds, setup_device
from full_pipeline import find_project_root
from torch import nn
from torch.utils.data import DataLoader
from training import validation_loop

SEED = 42


def to_batch_major_order(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert collate_fn output into (T, B, F) and mask (T, B).
    """
    xs, ys, mask = batch
    xs_t = xs.permute(1, 0, 2)  # (T, B, F)
    mask_t = mask.permute(1, 0)  # (T, B)
    return xs_t, mask_t


class ShapModel(nn.Module):
    def __init__(
        self,
        base_model,
        background_data,
        device,
        pad_value: float = 0.0,
    ):
        """
        Wrap model for SHAP compatibility.

        Args:
            base_model: Transformer model that expects (B, T, F) + mask
            pad_value: Value used to pad sequences (often 0)
        """
        super().__init__()
        self.device = device
        self.base = base_model
        self.pad_value = pad_value
        bg_loader = DataLoader(
            background_data.dataset,
            batch_size=min(100, len(background_data.dataset)),
            shuffle=False,
            generator=torch.Generator().manual_seed(SEED),
            collate_fn=collate_fn_wrapper,
            drop_last=False,
        )
        self.bg_batch = next(iter(bg_loader))
        self.bg_xs, self.bg_mask = to_batch_major_order(self.bg_batch)
        self.explainer = shap.GradientExplainer(self, self.bg_xs)

    def forward(self, xs):
        # xs comes in as (B, T, F)
        xs = xs.to(self.device)  # move to appropriate device

        # Reconstruct the mask: True for non-padding timesteps
        mask = xs.abs().sum(dim=-1) != (self.pad_value * xs.size(-1))
        mask = mask.transpose(0, 1)  # (B, T) -> (T, B)

        # Transpose into (T, B, F) for transformer
        xs_t = xs.transpose(0, 1)

        logits = self.base(xs_t, mask=mask)  # returns (B,) or (B,1)

        # Ensure SHAP always sees a 2D output
        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)  # (B,1)

        return logits

    def get_shap_values(self, data):
        loader = DataLoader(
            data.dataset,
            batch_size=64,
            shuffle=False,
            collate_fn=collate_fn_wrapper,
            drop_last=False,
        )
        shap_vals = []

        batch = next(iter(loader))
        # change this to loop through batches with tqdm
        xs, mask = to_batch_major_order(batch)
        shap_vals.append(self.explainer.shap_values(xs))

        shap_vals = np.concatenate(shap_vals, axis=0)
        shap_vals = np.squeeze(shap_vals, -1)
        # Broadcast mask to SHAP values without transposing to ensure matching dimensions
        # shap_vals shape: (B, T, F), mask shape: (B, T)
        shap_vals = shap_vals * mask.cpu().numpy()[:, :, np.newaxis]

        return shap_vals


def get_config(root, folder_name):
    config_path = f"{root}/models/model_B/results/{folder_name}/xperiment.yml"
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


# --------------------------------------------------------------------------
# Helper to compute predictions, ground-truth labels, and patient IDs
# --------------------------------------------------------------------------


def get_patient_predictions(model_wrapper, data, device, pred_threshold):
    """Return lists of patient IDs, y_true, and predicted probabilities.

    The order matches the dataset order (shuffle=False).
    """

    from torch.utils.data import DataLoader  # local import to avoid polluting namespace

    loader = DataLoader(
        data.dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        drop_last=False,
    )

    patient_ids = data.dataset.unique_patient_ids
    y_trues = []
    y_preds = []

    model = model_wrapper.model  # underlying TransformerTimeSeries
    model.eval()

    with torch.no_grad():
        for xs, ys, mask in loader:
            xs = xs.to(device)
            ys = ys.to(device)
            mask = mask.to(device)

            logits = model(xs, mask=mask)  # (B, 1) or (B,)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs >= pred_threshold).type(torch.int32).cpu().numpy()
            y_trues.extend(ys.cpu().numpy())
            y_preds.extend(preds)

    return patient_ids, np.array(y_trues), np.array(y_preds)


def collate_fn_wrapper(batch):
    """Wrapper for the collate function with fixed max_len."""
    return collate_fn(batch, max_len=400)


def prepare_shap_values(explainer, test_xs, test_mask) -> np.ndarray:
    """Prepare and process SHAP values with proper masking."""
    # Calculate SHAP values
    sh_vals = explainer.shap_values(test_xs)

    # Convert (N, T, F, 1) → (N, T, F)
    shap_vals = np.squeeze(sh_vals, -1)

    # Convert test_mask to (B, T, 1) for broadcasting
    test_mask_np = test_mask.cpu().numpy()[:, :, np.newaxis]  # (B, T, 1)

    # Apply mask - multiply SHAP values by mask (1 for real data, 0 for padding)
    return shap_vals * test_mask_np


# --- Visualization functions ---
def patient_heatmap(
    shap_vals,
    feature_names,
    time_index,
    idx: int,
    patient_id: str,
    y_true: float,
    y_pred: float,
) -> None:
    """
    Heat-map of SHAP values across time (x) and features (y) for one patient.

    Args:
        shap_vals: SHAP values array
        feature_names: List of feature names
        time_index: Array of time indices
        idx: Patient index to visualize
        top_10_features: List of top 10 features to visualize
    """

    # truncate the time dimension if there is no data for the last hours
    # check which hours have no data
    # Identify hours where SHAP values are zero for every patient **and** every feature
    # shap_vals has shape (N, T, F) -> we want a 1-D boolean mask over T
    no_data_hours = (shap_vals == 0).all(axis=(0, 2))  # shape (T,)
    if no_data_hours.any():
        time_index = time_index[~no_data_hours]
        shap_vals = shap_vals[:, ~no_data_hours]

    # Further crop timesteps that are blank for the selected patient only
    patient_no_data = (shap_vals[idx] == 0).all(axis=1)  # shape (T,)
    if patient_no_data.any():
        time_index = time_index[~patient_no_data]
        shap_vals = shap_vals[:, ~patient_no_data]

    # Dynamically scale the figure width so the heat-map fills the canvas
    n_timesteps = shap_vals.shape[1]
    width = max(6, n_timesteps * 0.2)  # 0.2 inch per timestep, min width 6
    plt.figure(figsize=(width, 6))
    sns.heatmap(
        shap_vals[idx].T,  # (F, T)
        cmap="vlag",
        center=0,
        yticklabels=feature_names,
        xticklabels=time_index,  # show every 10-hour tick
    )
    plt.title(
        f"Patient {patient_id}  |  y_true={int(y_true)}  |  y_pred={y_pred:.2f}",
        fontsize=14,
    )
    plt.xlabel("ICU LOS (hours)")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()


def plot_global_feature_importance(abs_vals, feature_names) -> None:
    """
    Bar plot of mean |SHAP| for each feature aggregated over patients & time.

    Args:
        abs_vals: Absolute SHAP values
        feature_names: List of feature names
    """
    global_imp = abs_vals.mean(axis=(0, 1))  # (F,)
    order = np.argsort(-global_imp)  # descending

    plt.figure(figsize=(8, 10))
    plt.barh(np.array(feature_names)[order], global_imp[order])
    plt.gca().invert_yaxis()
    plt.title("Mean |SHAP| per feature (all patients, all timesteps)")
    plt.xlabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.show()


def plot_temporal_importance(abs_vals, time_index) -> None:
    """
    Line plot of mean |SHAP| at each time-step (aggregated over patients & features).

    Args:
        abs_vals: Absolute SHAP values
        time_index: Array of time indices
    """
    time_imp = abs_vals.mean(axis=(0, 2))  # (T,)
    plt.figure(figsize=(12, 4))
    plt.plot(time_index, time_imp)
    plt.title("Mean |SHAP| across time (all patients, all features)")
    plt.xlabel("ICU LOS (hours)")
    plt.ylabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.show()


def beeswarm_collapsed_over_time(shap_vals, feature_names) -> None:
    """
    Classic SHAP 'beeswarm' after collapsing each patient-feature pair
    to the timestep with the largest absolute SHAP value, while keeping
    the original sign.

    Args:
        shap_vals: SHAP values array
        feature_names: List of feature names
    """
    # Index of the timestep where |SHAP| is largest for every patient-feature
    idx_max = np.abs(shap_vals).argmax(axis=1)  # shape (N, F)

    # Gather the signed SHAP value at that timestep
    collapse = shap_vals[
        np.arange(shap_vals.shape[0])[:, None],  # patient axis
        idx_max,
        np.arange(shap_vals.shape[2]),  # feature axis
    ]  # resulting shape (N, F)

    shap.summary_plot(
        collapse,
        features=None,
        feature_names=feature_names,
        show=True,
    )


def get_pred_threshold(model, val_data, device, config):
    """Get the prediction threshold for the given model."""
    val_loader = DataLoader(
        val_data.dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        drop_last=False,
    )
    loss_fn = get_loss_fn(config, val_data, device)
    (
        val_loss,
        val_acc,
        val_prec,
        val_rec,
        f2_score,
        full_y_pred,
        full_y_true,
        best_thr,
    ) = validation_loop(model.model, val_loader, loss_fn, device)
    return best_thr


def shap_pipeline():
    """Main function to run the SHAP analysis pipeline."""

    project_root = find_project_root()
    if len(sys.argv) > 1:
        results_name = sys.argv[1]
    else:
        raise ValueError("No results name provided")

    config = get_config(project_root, results_name)

    device = setup_device()

    train_data = get_data(config, "train")
    val_data = get_data(config, "val")

    test_data = get_data(config, "test")
    in_dim = train_data.X.shape[1]

    model = ModelWrapper(config, device, in_dim)
    model.load_saved_weights()

    shap_model = ShapModel(model.model, train_data, device, pad_value=0.0)
    shap_vals = shap_model.get_shap_values(test_data)

    pred_threshold = get_pred_threshold(model, val_data, device, config)
    print(f"Prediction threshold: {pred_threshold}")

    feature_names = train_data.X.columns.tolist()  # list of feature names
    time_index = np.arange(shap_vals.shape[1])

    abs_vals = np.abs(shap_vals)
    top_10_features = np.argsort(abs_vals.mean(axis=(0, 1)))[-10:]
    top_10_shap_vals = shap_vals[:, :, top_10_features]
    top_10_feature_names = np.array(feature_names)[top_10_features]
    top_10_abs_vals = np.abs(top_10_shap_vals)

    patient_ids, y_trues, y_preds = get_patient_predictions(
        model, test_data, device, pred_threshold
    )
    for idx, (patient_id, y_true, y_pred) in enumerate(
        zip(patient_ids, y_trues, y_preds)
    ):
        patient_heatmap(
            top_10_shap_vals,
            top_10_feature_names,
            time_index,
            idx,
            patient_id,
            y_true,
            y_pred,
        )
    plot_global_feature_importance(top_10_abs_vals, top_10_feature_names)
    plot_temporal_importance(abs_vals, time_index)
    beeswarm_collapsed_over_time(shap_vals, feature_names)


# --- Run the code ---
if __name__ == "__main__":
    set_seeds()
    shap_pipeline()
