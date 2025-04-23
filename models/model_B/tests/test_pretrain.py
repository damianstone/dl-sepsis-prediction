"""Smoke tests for the masked‑value pre‑training pipeline.

These tests run on tiny synthetic data to verify that
1. The training loop completes without error.
2. A checkpoint file is created.
3. Pretrained weights can be loaded into the supervised model.
"""

from pathlib import Path

import numpy as np
import torch

from models.model_B.architectures import TransformerTimeSeries
from models.model_B.custom_dataset import SepsisPatientDataset
from models.model_B.pretrain import load_pretrained_encoder, train_masked_model


def _create_dummy_dataset(num_patients: int = 8, feat_dim: int = 7):
    rng = np.random.default_rng(42)
    pts, labels, ids = [], [], []
    for pid in range(num_patients):
        n_steps = rng.integers(4, 8)
        for _ in range(n_steps):
            pts.append(rng.random(feat_dim))
            labels.append(0.0)
            ids.append(pid)
    return SepsisPatientDataset(
        np.array(pts, dtype=np.float32),
        np.array(labels, dtype=np.float32),
        np.array(ids),
        time_index=0,  # any column works for synthetic data
    )


def test_pretrain_and_load(tmp_path: Path):
    """End‑to‑end smoke test for pretraining + weight loading."""

    dataset = _create_dummy_dataset()
    ckpt_path = tmp_path / "encoder.pth"

    # Pre‑train for 1 epoch with very small model to keep CI fast
    saved = train_masked_model(
        dataset=dataset,
        input_dim=7,
        d_model=16,
        n_heads=2,
        n_layers=1,
        batch_size=4,
        epochs=1,
        device="cpu",
        save_path=ckpt_path,
    )

    assert saved.exists(), "Checkpoint file was not created"

    # Create a supervised model and load encoder weights
    model = TransformerTimeSeries(
        input_dim=7, d_model=16, n_heads=2, n_layers=1, dropout=0.1
    )
    load_pretrained_encoder(model, saved)

    # Verify that embedding weights are now identical
    pretrained_weights = torch.load(saved, map_location="cpu")["embedding.weight"]
    assert torch.allclose(model.embedding.weight, pretrained_weights)
