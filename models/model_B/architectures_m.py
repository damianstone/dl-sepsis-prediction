import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding, shaped to add directly onto
    a (seq_len, batch, d_model) tensor.
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # create positional encodings once in log space
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # reshape to (max_len, 1, d_model) so we can add to (seq_len, batch, d_model)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape (seq_len, batch_size, d_model)
        Returns:
            x + positional encodings, same shape
        """
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class TransformerTimeSeries(nn.Module):
    """
    Per‑step Transformer with max‑over‑time patient aggregation on eval.

    Args:
        input_dim: # of raw features per time‑step
        d_model:  Transformer embedding size
        nhead:     # of attention heads
        num_layers:# of TransformerEncoder layers
        dropout:   dropout both in pos‑enc and inside layers
    """

    def __init__(
        self,
        input_dim: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        # project raw features → model dimension
        print("SETTING UP M TRANSFORMER")
        self.embedding = nn.Linear(input_dim, d_model)

        # add timestep info
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # build the encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # per‑step logit head
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        # ensure mask is boolean
        if mask is not None:
            mask_bool = mask.bool()
        else:
            mask_bool = None
        """
        Args:
            x:    (seq_len, batch_size, input_dim)
            mask: (seq_len, batch_size) boolean with True=valid, False=pad
        Returns:
            train: (seq_len, batch) logits
            eval:  (batch,) patient‑level score = max‑over‑time
        """
        # embed + pos‑encode
        x = self.embedding(x)  # → (seq_len, batch, d_model)
        x = self.pos_encoder(x)  # → (seq_len, batch, d_model)

        # prepare padding mask for the transformer: True=pad
        if mask is not None:
            # mask: (seq_len, batch) → (batch, seq_len)

            src_key_padding_mask = ~mask_bool.transpose(0, 1)
        else:
            src_key_padding_mask = None

        # transformer expects (seq_len, batch, d_model)
        x = self.transformer_encoder(
            x, src_key_padding_mask=src_key_padding_mask
        )  # → (seq_len, batch, d_model)

        # per‑step logits
        logits = self.decoder(x).squeeze(-1)  # → (seq_len, batch)

        if not self.training:
            # eval mode: max‑over‑time per patient
            if mask is not None:
                # mask out pads to -inf so they never win
                logits = logits.masked_fill(~mask_bool, float("-inf"))
            # → (batch,)
            patient_score, _ = logits.max(dim=0)
            return patient_score

        # training mode: return all time‑step logits
        return logits
