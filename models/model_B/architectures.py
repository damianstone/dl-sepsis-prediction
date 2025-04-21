import numpy as np
import torch
from torch import nn


class TransformerClassifier(nn.Module):
    """
    num_heads = more heads capture different attention but increase computation
    num_layers = more make the model deeper but can overfit if too high
    """

    def __init__(self, input_dim, num_heads=2, num_layers=2, drop_out=0.1):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, dropout=drop_out, batch_first=True
        )
        # stacks multiple encoder layers (num_layers controls depth)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # linear layer to map the output to a single value (binary classification)
        self.linear_layer = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x, mask=None):
        # mask to ignore padding in self-attention
        z = self.encoder(x, src_key_padding_mask=mask)
        return self.linear_layer(z[:, -1, :])  # use last time step for classification


# -------------------- Martin-style PositionalEncoding (batch‑first) --------------------
class PositionalEncoding(nn.Module):
    """Batch‑first positional encoding identical to Martin's implementation but adjusted
    to accept tensors in the shape (batch, seq_len, d_model)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create constant positional encoding matrix with shape (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer with shape (1, max_len, d_model) so it is moved with .to(device)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor of same shape as *x* with positional encodings added.
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# -------------------- Martin-style TransformerTimeSeries --------------------
class TransformerTimeSeries(nn.Module):
    """Time‑series transformer identical in spirit to Martin's reference model.

    ‑ Accepts input in the shape (sequence_length, batch_size, feature_dim)
      (same as our collate_fn output) but internally converts it to batch‑first
      before feeding it to a *batch_first* TransformerEncoder.
    ‑ Returns raw logits for each time step: (sequence_length, batch_size).
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # 1) Input projection  (feature_dim → d_model)
        self.input_proj = nn.Linear(input_dim, d_model)

        # 2) Positional encoding (batch‑first)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # 3) Transformer encoder (batch_first=True so we can keep tensors batch‑first inside)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        # 4) Final linear decoder produces one logit per time step
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:  # type: ignore
        """Forward pass.

        Args:
            x: Tensor of shape (seq_len, batch, feature_dim)
            mask: Bool Tensor of shape (seq_len, batch) where **1/True means valid**
        Returns:
            Logits of shape (seq_len, batch)
        """
        # Convert to batch‑first: (batch, seq_len, feature_dim)
        x = x.transpose(0, 1)

        # Input projection & positional encoding
        x = self.input_proj(x)  # (batch, seq_len, d_model)
        x = self.pos_encoder(x)  # (batch, seq_len, d_model)

        # Prepare src_key_padding_mask for the encoder: True == padding.
        # NOTE: On Apple M‑series (MPS backend) nested‑tensor based padding masks
        # are not yet implemented and raise `NotImplementedError`.  To keep the
        # code runnable on MPS we simply skip the mask in that case; padding
        # tokens are still ignored later when we compute the loss & metrics.
        if mask is not None and x.device.type != "mps":
            src_key_padding_mask = (~mask.bool()).transpose(0, 1)  # (batch, seq_len)
        else:
            src_key_padding_mask = None

        # Transformer encoder
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # (batch, seq_len, d_model)

        # Time‑step-wise logits
        logits = self.decoder(x).squeeze(-1)  # (batch, seq_len)

        # Return to original orientation (seq_len, batch)
        return logits.transpose(0, 1)
