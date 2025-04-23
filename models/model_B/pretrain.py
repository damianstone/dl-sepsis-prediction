# models/model_B/pretrain.py
"""Self‑supervised masked‑value modelling (Time‑BERT style) for the
TransformerTimeSeries encoder.

Running this script:
    python -m models.model_B.pretrain

The script will:
1. Build a DataLoader that returns padded sequences + attention masks
   (labels are ignored).
2. Randomly mask a proportion of the (time‑step, feature) cells.
3. Train the encoder to reconstruct the original values only for the
   masked positions (MSE loss).
4. Save a checkpoint called ``pretrained_encoder.pth`` under
   ``models/model_B/saved``.  This checkpoint can later be loaded in the
   fine‑tuning pipeline via ``load_pretrained_encoder``.
"""
# mypy compat
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports (relative path).  The surrounding package might not be a
# true module when executed as a script; we therefore use a try/except
# import guard so that both ``python pretrain.py`` and
# ``python -m models.model_B.pretrain`` work.
try:
    from .architectures import TransformerTimeSeries
    from .custom_dataset import SepsisPatientDataset
except ImportError:  # pragma: no cover
    from architectures import TransformerTimeSeries  # type: ignore
    from custom_dataset import SepsisPatientDataset  # type: ignore


# -----------------------------------------------------------------------------
# Utility: masked‑value collation
# -----------------------------------------------------------------------------


def collate_fn_masked(
    batch, mask_ratio: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate patients into batch‑first tensors and apply random masking.

    Parameters
    ----------
    batch : list
        Each element is the tuple returned by ``SepsisPatientDataset.__getitem__``
        → (X, y).
    mask_ratio : float, optional
        Fraction of *cells* to mask.  Default is 0.15 (15 %).

    Returns
    -------
    masked_X : Tensor, shape (seq_len, batch_size, feature_dim)
        Input with some cells replaced by 0 (mask token handled inside the
        model).
    attention_mask : Bool Tensor, shape (seq_len, batch_size)
        True for real data, False for padding.
    value_mask : Bool Tensor, shape (seq_len, batch_size, feature_dim)
        True at locations that were masked and should contribute to the loss.
    target_values : Tensor, shape (seq_len, batch_size, feature_dim)
        The ground‑truth unmasked values (only used where ``value_mask`` is
        True).
    """
    # --- reuse the original padding logic (ignoring labels) ---
    X_batch = [x for x, _ in batch]

    max_len = max(x.shape[0] for x in X_batch)
    feature_dim = X_batch[0].shape[1]

    padded_X = torch.zeros(len(X_batch), max_len, feature_dim)
    attention_mask = torch.ones(len(X_batch), max_len)  # 1 = valid, 0 = pad

    for i, x in enumerate(X_batch):
        padded_X[i, : x.shape[0], :] = x
        attention_mask[i, x.shape[0] :] = 0

    # Convert to seq_len‑first as expected by the encoder
    padded_X = padded_X.transpose(0, 1)  # (S, B, F)
    attention_mask = attention_mask.transpose(0, 1)  # (S, B)

    # ------------------------------------------------------------------
    # Random masking per *cell* (time‑step, feature)
    # ------------------------------------------------------------------
    value_mask = torch.rand_like(padded_X, dtype=torch.float32) < mask_ratio
    value_mask &= attention_mask.unsqueeze(-1).bool()  # ensure we do not mask pad

    target_values = padded_X.clone()  # ground truth for the masked cells

    # Replace masked positions with zeros (mask token added in model)
    padded_X.masked_fill_(value_mask, 0.0)

    return padded_X, attention_mask.bool(), value_mask.bool(), target_values


# -----------------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------------


def masked_pretrain(
    model: TransformerTimeSeries,
    dataset: SepsisPatientDataset,
    val_dataset: SepsisPatientDataset,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    mask_ratio: float = 0.15,
    patience: int = 5,
    device: torch.device | str | None = None,
    save_path: Path | str | None = None,
) -> Path:
    """Train the masked reconstruction model and persist encoder weights.

    Returns the path to the saved *encoder* checkpoint.
    Early stopping: if `val_dataset` is provided the function monitors the
    validation reconstruction loss and stops when it has not improved for
    `patience` consecutive epochs.
    """
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    model = model.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_fn_masked(batch, mask_ratio=mask_ratio),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_masked(batch, mask_ratio=mask_ratio),
        drop_last=False,
    )

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch+1}/{epochs}")
        total_loss = 0.0
        total_masks = 0

        for masked_X, attn_mask, value_mask, target in pbar:
            masked_X = masked_X.to(device)
            attn_mask = attn_mask.to(device)
            value_mask = value_mask.to(device)
            target = target.to(device)

            pred = model.forward_reconstruction(masked_X, attn_mask)

            # Mean‑squared reconstruction error on masked cells only
            sq_err = (pred - target) ** 2
            loss = sq_err.masked_select(value_mask).mean()

            optim.zero_grad()
            loss.backward()
            optim.step()

            with torch.no_grad():
                total_loss += loss.item()
                total_masks += 1

            pbar.set_postfix({"loss": loss.item()})

        epoch_loss = total_loss / max(total_masks, 1)

        # ---------------- Validation ----------------

        model.eval()
        val_total, val_count = 0.0, 0
        with torch.no_grad():
            for v_X, v_mask, v_valmask, v_target in val_loader:
                v_X, v_mask, v_valmask, v_target = (
                    v_X.to(device),
                    v_mask.to(device),
                    v_valmask.to(device),
                    v_target.to(device),
                )
                v_pred = model.forward_reconstruction(v_X, v_mask)
                v_sq = (v_pred - v_target) ** 2
                v_loss = v_sq.masked_select(v_valmask).mean()
                val_total += v_loss.item()
                val_count += 1
        val_loss = val_total / max(val_count, 1)
        print(f"Epoch {epoch+1}: train MSE {epoch_loss:.6f} | val MSE {val_loss:.6f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered (no val loss improvement).")
                break

    # ------------------------------------------------------------------
    # Save encoder state_dict (without recon head to keep size small)
    # ------------------------------------------------------------------
    save_dir = (
        Path(save_path).resolve().parent
        if save_path is not None
        else Path(__file__).parent / "saved"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / (
        Path(save_path).name if save_path is not None else "pretrained_encoder.pth"
    )

    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved pretrained encoder to {ckpt_path}")
    return ckpt_path


# -----------------------------------------------------------------------------
# Helper to load encoder weights into TransformerTimeSeries
# -----------------------------------------------------------------------------


def load_pretrained_encoder(
    model: "TransformerTimeSeries", ckpt_path: str | Path, strict: bool = False
) -> None:
    """Load pretrained encoder weights into a `TransformerTimeSeries` model.

    Only layers that match (embedding, positional encoder params, encoder)
    will be loaded; the linear classification head remains randomly
    initialised.
    """
    state_dict = torch.load(ckpt_path, map_location="cpu")

    # Filter keys relevant to supervised model
    filtered = {
        k: v
        for k, v in state_dict.items()
        if k.startswith("embedding")
        or k.startswith("positional_encoder")
        or k.startswith("encoder")
    }

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Missing keys: {missing}\nUnexpected keys: {unexpected}")
    print(f"Loaded pretrained encoder from {ckpt_path}")


# -----------------------------------------------------------------------------
# Command‑line entry point
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    # Simple debug run on a fake dataset
    print("Running quick self‑test of pretraining script…")

    # Fake data: 100 patients, each 10‑20 timesteps, 5 features
    rng = np.random.default_rng(0)
    pts = []
    labels = []
    ids = []
    for pid in range(100):
        n_steps = rng.integers(10, 21)
        for t in range(n_steps):
            pts.append(rng.random(5))
            labels.append(0)  # labels unused
            ids.append(pid)

    dataset = SepsisPatientDataset(
        np.array(pts, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(ids),
        time_index=0,  # dummy
    )

    masked_pretrain(
        dataset=dataset,
        input_dim=5,
        d_model=32,
        n_heads=2,
        n_layers=2,
        batch_size=16,
        epochs=1,
        device="cpu",
    )
