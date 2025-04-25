"""
explain_model_helpers.py

Utility helpers for SHAP explainability, plotting, and evaluation of the
Transformer-based sepsis-prediction model.

The module purposefully remains a *single file* so that it can be dropped into
existing projects without changing their import paths.  The internal structure
has, however, been modernised and sectioned to improve readability.

Sections
--------
1. Configuration & logging
2. Low-level tensor/data helpers
3. Model wrapper (``ShapModel``)
4. SHAP post-processing helpers
5. Evaluation helpers
6. Plotting utilities
"""

# =============================================================================
# 1. Configuration & logging
# =============================================================================
import copy
import logging
import pickle
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import torch
import yaml
from custom_dataset import collate_fn
from final_pipeline import get_loss_fn
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from training import validation_loop

__all__ = [
    "ExplainConfig",
    "ShapModel",
    "strip_padding",
    "remove_padding",  # backward-compat alias
    "predict_per_patient",
    "get_patient_predictions",  # backward-compat alias
    "prepare_shap_values",
    "get_pred_threshold",
    "patient_heatmap",
    "global_heatmap",
    "global_importance_plot",
]

logger = logging.getLogger(__name__)
if not logger.handlers:
    # Default basic configuration – caller can override via logging API
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s"
    )


class DebugLevel(Enum):
    """Verbosity levels used across this module."""

    OFF = auto()
    QUICK = auto()
    MEDIUM = auto()


@dataclass(frozen=True)
class ExplainConfig:
    """Centralised configuration shared by helpers in this file."""

    seed: int = 42
    debug: DebugLevel = DebugLevel.OFF
    pad_value: float = 0.0
    batch_size: int = 256
    max_len: int = 400  # maximum sequence length fed to collate_fn
    top_k_features: int = 10  # default number of features to visualise
    window_size: int = (
        30  # default size (in hours) for sliding window in global heatmap
    )


# Instantiate a *module-level* config object that downstream code can mutate
CONFIG = ExplainConfig()

# Preserve original boolean flags for full backwards compatibility -------------
QUICK_DEBUG = CONFIG.debug == DebugLevel.QUICK
MEDIUM_DEBUG = CONFIG.debug == DebugLevel.MEDIUM
SEED = CONFIG.seed

# Set torch's random seed for reproducibility
if torch.cuda.is_available():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
else:
    torch.manual_seed(SEED)

# =============================================================================
# 2. Low-level tensor/data helpers
# =============================================================================


def to_batch_major_order(batch) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert ``collate_fn`` output into (T, B, F) *and* mask (T, B)."""

    xs, _ys, mask = batch  # we do not need ``ys`` for SHAP
    xs_t = xs.permute(1, 0, 2)  # (T, B, F)
    mask_t = mask.permute(1, 0)  # (T, B)
    return xs_t, mask_t


def collate_fn_wrapper(batch):
    """Thin wrapper around the project-specific ``collate_fn`` with a fixed max_len."""

    return collate_fn(batch, max_len=CONFIG.max_len)


# -----------------------------------------------------------------------------
# Padding helpers
# -----------------------------------------------------------------------------


def strip_padding(patient_data: np.ndarray, time_index: np.ndarray):
    """Remove right-hand side padding rows (all-zero timesteps) in *patient_data*.

    Parameters
    ----------
    patient_data
        2-D array shaped (T, F).
    time_index
        Array of associated timesteps/hours.

    Returns
    -------
    valid_time_index, valid_patient_data
        Truncated arrays where trailing padding rows were discarded.
    """

    non_zero_timesteps = np.where(~(patient_data == 0).all(axis=1))[0]

    if len(non_zero_timesteps) > 0:
        last_timestep = non_zero_timesteps[-1] + 1
        return time_index[:last_timestep], patient_data[:last_timestep]

    return time_index, patient_data


# Alias for backwards compatibility
remove_padding = strip_padding  # noqa: E305


# -----------------------------------------------------------------------------
# SHAP post-processing helpers
# -----------------------------------------------------------------------------


def apply_shap_mask(shap_vals: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove the singleton output dimension and apply the *padding* mask."""

    shap_vals_squeezed = np.squeeze(shap_vals, -1)  # (B, T, F)
    return shap_vals_squeezed * mask[:, :, np.newaxis]


# =============================================================================
# 3. Model wrapper – ``ShapModel``
# =============================================================================


class ShapModel(nn.Module):
    """Lightweight nn.Module wrapper so that SHAP can query the base network."""

    def __init__(
        self,
        base_model: nn.Module,
        background_data,
        device: torch.device,
        pad_value: float | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.base = base_model
        self.pad_value = pad_value if pad_value is not None else CONFIG.pad_value

        # Ensure deterministic behaviour (e.g., disable dropout)
        self.base.eval()

        # Background batch used by GradientExplainer --------------------------
        bg_loader = DataLoader(
            background_data.dataset,
            batch_size=len(background_data.dataset),
            shuffle=False,
            generator=torch.Generator().manual_seed(SEED),
            collate_fn=collate_fn_wrapper,
            drop_last=False,
            num_workers=12,
        )
        self.bg_batch = next(iter(bg_loader))
        self.bg_xs, self.bg_mask = to_batch_major_order(self.bg_batch)
        self.explainer = shap.GradientExplainer(self, self.bg_xs)

    # ---------------------------------------------------------------------
    # nn.Module interface
    # ---------------------------------------------------------------------
    def forward(self, xs: torch.Tensor):
        """Perform a forward pass compatible with SHAP's expectations."""

        xs = xs.to(self.device)  # ensure correct device placement

        # Reconstruct boolean mask where True denotes non-padding timesteps
        mask = xs.abs().sum(dim=-1) != (self.pad_value * xs.size(-1))
        mask = mask.transpose(0, 1)  # (B, T) -> (T, B)

        logits = self.base(xs.transpose(0, 1), mask=mask)  # (B,) or (B, 1)

        # SHAP requires a 2-D output: (B, 1)
        return logits.unsqueeze(-1) if logits.dim() == 1 else logits

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def get_shap_values(self, data):
        """Compute SHAP values in mini-batches (padding aware)."""

        loader = DataLoader(
            data.dataset,
            batch_size=CONFIG.batch_size,
            shuffle=False,
            collate_fn=collate_fn_wrapper,
            drop_last=False,
            num_workers=12,
        )

        shap_values_lst: list[np.ndarray] = []
        masks_lst: list[np.ndarray] = []

        for idx, batch in enumerate(tqdm(loader, desc="SHAP batches")):
            xs, mask = to_batch_major_order(batch)
            shap_values_lst.append(self.explainer.shap_values(xs))
            masks_lst.append(mask.cpu().numpy())

            if QUICK_DEBUG:
                logger.info("Early exit after first batch due to QUICK_DEBUG.")
                break
            if MEDIUM_DEBUG and idx >= 1:
                logger.info("Early exit after two batches due to MEDIUM_DEBUG.")
                break

        shap_vals = np.concatenate(shap_values_lst, axis=0)
        masks = np.concatenate(masks_lst, axis=0)
        shap_vals = apply_shap_mask(shap_vals, masks)
        self.shap_vals = shap_vals
        self.masks = masks
        return shap_vals, masks

    def save(self, folder: str):
        if self.shap_vals is None or self.masks is None:
            raise ValueError("SHAP values and masks must be computed first")
        path = Path(folder)
        if not path.exists():
            path.mkdir(parents=True)
        np.save(path / "shap_vals.npy", self.shap_vals)
        np.save(path / "masks.npy", self.masks)
        with open(path / "explainer.pkl", "wb") as f:
            pickle.dump(self.explainer, f)


# =============================================================================
# 4. Evaluation helpers (predictions, threshold selection, config I/O)
# =============================================================================


def get_config(root: str | Path, folder_name: str):
    """Load ``xperiment.yml`` for a given training run."""

    cfg_path = (
        Path(root) / "models" / "model_B" / "results" / folder_name / "xperiment.yml"
    )
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {cfg_path}")
    with cfg_path.open("r") as file:
        return yaml.safe_load(file)


# -----------------------------------------------------------------------------
# Predictions per patient
# -----------------------------------------------------------------------------


def predict_per_patient(model_wrapper, data, device: torch.device, pred_threshold):
    """Return ``patient_ids``, ``y_true``, ``y_pred``, ``y_prob`` arrays (order preserved)."""

    loader = DataLoader(
        data.dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        drop_last=False,
    )

    patient_ids = data.dataset.unique_patient_ids
    y_trues, y_probs, y_preds = [], [], []

    model = model_wrapper.model  # underlying Transformer
    model.eval()

    with torch.no_grad():
        for xs, ys, mask in loader:
            xs, ys, mask = xs.to(device), ys.to(device), mask.to(device)
            logits = model(xs, mask=mask)  # (B, 1) or (B,)
            probs = torch.sigmoid(logits.squeeze(-1))
            preds = (probs >= pred_threshold).type(torch.int32).cpu().numpy()

            y_trues.extend(ys.cpu().numpy())
            y_preds.extend(preds)
            y_probs.extend(probs.cpu().numpy())

    return patient_ids, np.array(y_trues), np.array(y_preds), np.array(y_probs)


# Backwards-compat alias -------------------------------------------------------
get_patient_predictions = predict_per_patient


# -----------------------------------------------------------------------------
# SHAP preparation helper
# -----------------------------------------------------------------------------


def prepare_shap_values(
    explainer, test_xs: torch.Tensor, test_mask: torch.Tensor
) -> np.ndarray:
    """Wrapper around ``explainer.shap_values`` that removes padding rows."""

    raw_vals = explainer.shap_values(test_xs)
    return apply_shap_mask(raw_vals, test_mask.cpu().numpy())


# -----------------------------------------------------------------------------
# Threshold helper
# -----------------------------------------------------------------------------


def get_pred_threshold(model, val_data, device, config):
    """Determine optimal prediction threshold (F-beta) on the validation set."""

    val_loader = DataLoader(
        val_data.dataset,
        batch_size=CONFIG.batch_size,
        shuffle=False,
        collate_fn=collate_fn_wrapper,
        drop_last=False,
    )
    loss_fn = get_loss_fn(config, val_data, device)
    (
        _val_loss,
        _val_acc,
        _val_prec,
        _val_rec,
        _f2_score,
        _full_y_pred,
        _full_y_true,
        best_thr,
    ) = validation_loop(model.model, val_loader, loss_fn, device)
    return best_thr


# =============================================================================
# 5. SHAP aggregation helpers
# =============================================================================


def masked_average(shap_vals: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Mean SHAP per timestep across patients, aware of *padding* rows."""

    shap_vals = copy.deepcopy(shap_vals)
    # Boolean mask True where at least one feature differs from 0 (within tol)
    non_pad_mask = np.any(np.abs(shap_vals) > tol, axis=2)  # (B, T)

    counts = non_pad_mask.sum(axis=0)  # (T,)
    summed = (shap_vals * non_pad_mask[..., None]).sum(axis=0)  # (T, F)

    with np.errstate(divide="ignore", invalid="ignore"):
        avg = np.divide(summed, counts[:, None], where=counts[:, None] != 0)

    return np.nan_to_num(avg)


# =============================================================================
# 6. Plotting utilities (Matplotlib / Seaborn)
# =============================================================================


def patient_heatmap(
    shap_vals: np.ndarray,
    feature_names,
    time_index,
    idx: int,
    patient_id: str,
    y_true: float,
    y_pred: float,
    y_prob: float,
    *,
    only_target_relevant: bool = True,
    show: bool = True,
):
    """Plot a SHAP heat-map for a *single* patient and optionally return the figure."""

    # Truncate padding --------------------------------------------------------
    patient_data = copy.deepcopy(shap_vals[idx])  # (T, F)
    valid_time_index, valid_patient_data = strip_padding(patient_data, time_index)

    # Focus on SHAP values relevant to the *target class* ---------------------
    if only_target_relevant:
        valid_patient_data = np.clip(
            valid_patient_data,
            a_min=0 if y_true == 1 else None,
            a_max=0 if y_true == 0 else None,
        )

    # Top-k most relevant features -------------------------------------------
    mean_importance = np.abs(valid_patient_data).mean(axis=0)
    top_features = np.argsort(mean_importance)[-CONFIG.top_k_features :]
    top_feature_names = np.array(feature_names)[top_features]

    # Plot --------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(max(10, valid_time_index.shape[0] * 0.1), 8),
        constrained_layout=True,
    )
    sns.heatmap(
        valid_patient_data[:, top_features].T,
        cmap="vlag",
        center=0,
        yticklabels=top_feature_names,
        xticklabels=valid_time_index,
        ax=ax,
    )
    ax.set_title(
        (
            f"Patient {patient_id} | y_true={int(y_true)} | "
            f"y_pred={int(y_pred)} | y_prob={y_prob:.2f}"
        ),
        fontsize=14,
    )
    ax.set_xlabel("ICU LOS (hours)")
    ax.set_ylabel("Feature")

    if show:
        plt.show()
    return fig


# -----------------------------------------------------------------------------
# Global heatmap (all patients of a given predicted class)
# -----------------------------------------------------------------------------


def global_heatmap(
    shap_vals: np.ndarray,
    feature_names,
    time_index,
    target: int,
    y_preds,
    top_features: list[int] | np.ndarray | None = None,
    *,
    only_target_relevant: bool = True,
    window_size: int | None = None,
    show: bool = True,
):
    """Global heat-map of SHAP values for *predicted* class ``target``."""

    window_size = window_size or CONFIG.window_size

    shap_vals = copy.deepcopy(shap_vals)
    max_idx = len(shap_vals) - 1

    target_indices = np.where(y_preds == target)[0]
    target_indices = target_indices[target_indices <= max_idx]

    if len(target_indices) == 0:
        logger.warning("No patients were predicted as class %s", target)
        return None

    target_shap_vals = copy.deepcopy(shap_vals[target_indices])

    if only_target_relevant:
        target_shap_vals = np.clip(
            target_shap_vals,
            a_min=0 if target == 1 else None,
            a_max=0 if target == 0 else None,
        )

    mean_shap_vals = masked_average(target_shap_vals)  # (T, F)
    time_importance = np.abs(mean_shap_vals).mean(axis=1)  # (T,)

    # ------------------------------------------------------------------
    # Simpler behaviour: always show the *first* ``window_size`` timesteps.
    # ------------------------------------------------------------------
    window_start = 0
    window_end = int(min(window_size, len(time_importance)))

    logger.info(
        "Using first %d timesteps (0 – %d)", window_end - window_start, window_end
    )

    window_time_index = time_index[window_start:window_end]
    window_patient_shap_vals = target_shap_vals[:, window_start:window_end, :]
    average_shap_vals = masked_average(window_patient_shap_vals)  # (T_window, F)

    top_feature_names = np.array(feature_names)[top_features]

    # Plot --------------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(max(10, len(window_time_index) * 0.1), 8),
        constrained_layout=True,
    )
    sns.heatmap(
        average_shap_vals[:, top_features].T,
        cmap="vlag",
        center=0,
        yticklabels=top_feature_names,
        xticklabels=window_time_index,
        ax=ax,
    )
    ax.set_title(
        (
            f"Global SHAP for class {target} – top {window_size}-hr window "
            f"(n={len(target_indices)})"
        ),
        fontsize=14,
    )
    ax.set_xlabel("ICU LOS (hours)")
    ax.set_ylabel("Feature")

    if show:
        plt.show()
    return fig


# -----------------------------------------------------------------------------
# Global feature-importance bar plot
# -----------------------------------------------------------------------------


def global_importance_plot(
    shap_vals: np.ndarray,
    feature_names,
    time_index,
    target: int,
    y_preds,
    *,
    show: bool = True,
):
    """Bar plot with *global* feature importance for predicted class ``target``."""

    max_idx = len(shap_vals) - 1
    target_indices = np.where(y_preds == target)[0]
    target_indices = target_indices[target_indices <= max_idx]

    # target_shap_vals = np.clip(
    #     copy.deepcopy(shap_vals[target_indices]),
    #     a_min=0 if target == 1 else None,
    #     a_max=0 if target == 0 else None,
    # )
    target_shap_vals = copy.deepcopy(shap_vals)

    logger.debug("Aggregating %d patient explanations", len(target_indices))

    feature_averages = []
    for patient_shap in target_shap_vals:
        _, valid_patient_data = strip_padding(patient_shap, time_index)
        feature_averages.append(valid_patient_data.mean(axis=0))

    global_feature_importance = np.array(feature_averages).mean(axis=0)

    top_features = np.argsort(np.abs(global_feature_importance))[
        -CONFIG.top_k_features :
    ]
    top_feature_names = np.array(feature_names)[top_features]

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    cmap = cm.get_cmap("coolwarm")
    color = cmap(0.9) if target == 1 else cmap(0.1)

    ax.barh(top_feature_names, global_feature_importance[top_features], color=color)
    ax.set_title(f"Global Feature Importance for Class {target}")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")

    if show:
        plt.show()

    return fig, top_features[::-1]
