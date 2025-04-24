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
from final_pipeline import ModelWrapper, get_data, set_seeds, setup_device
from full_pipeline import find_project_root
from torch import nn
from torch.utils.data import DataLoader

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
def patient_heatmap(shap_vals, feature_names, time_index, idx: int) -> None:
    """
    Heat-map of SHAP values across time (x) and features (y) for one patient.

    Args:
        shap_vals: SHAP values array
        feature_names: List of feature names
        time_index: Array of time indices
        idx: Patient index to visualize
    """
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        shap_vals[idx].T,  # (F, T)
        cmap="vlag",
        center=0,
        yticklabels=feature_names,
        xticklabels=time_index[::10],  # show every 10-hour tick
    )
    plt.title(f"SHAP values for patient {idx}")
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


def shap_pipeline():
    """Main function to run the SHAP analysis pipeline."""

    project_root = find_project_root()
    if len(sys.argv) > 1:
        results_name = sys.argv[1]
    else:
        raise ValueError("No results name provided")

    config = get_config(project_root, results_name)
    print(config)

    device = setup_device()

    train_data = get_data(config, "train")
    test_data = get_data(config, "test")
    in_dim = train_data.X.shape[1]
    # max_len = 0
    # i = 0
    # import tqdm

    # # loop through batches
    # for xs, ys, mask in tqdm.tqdm(train_data.loader):
    #     max_len = max(max_len, xs.shape[0])
    #     i += 1
    #     print(f"Batch {i} of {len(train_data.loader)}")
    # print(f"Max length: {max_len}")
    # return

    model = ModelWrapper(config, device, in_dim)
    model.load_saved_weights()

    shap_model = ShapModel(model.model, train_data, device, pad_value=0.0)
    shap_vals = shap_model.get_shap_values(test_data)

    abs_vals = np.abs(shap_vals)

    feature_names = train_data.X.columns.tolist()  # list of feature names
    time_index = np.arange(shap_vals.shape[1])  # 0 … 246 hours

    patient_heatmap(shap_vals, feature_names, time_index, 0)
    plot_global_feature_importance(abs_vals, feature_names)
    plot_temporal_importance(abs_vals, time_index)
    beeswarm_collapsed_over_time(shap_vals, feature_names)


# --- Run the code ---
if __name__ == "__main__":
    set_seeds()
    shap_pipeline()
