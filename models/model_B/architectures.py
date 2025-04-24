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


class PositionalEncoding(nn.Module):
    """
    adds a unique signal to each patient record based on its time step
    so the transformer can learn by the order of records
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # empty positional encoding table for all time steps and features
        pe = torch.zeros(max_len, d_model)
        # add the time step index
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )

        # we use sine and cosine to represent the time and position of each patient event (record)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: Tensor of shape (seq_len, batch_size, d_model)
        """
        # number of time steps
        seq_len = x.size(0)
        # extract positional encodings for these time steps: (seq_len, d_model)
        pe = self.pe[0, :seq_len, :]  # (seq_len, d_model)
        # add a batch dimension for broadcasting: (seq_len, 1, d_model)
        pe = pe.unsqueeze(1)
        # broadcast add to the input tensor
        x = x + pe  # (seq_len, batch_size, d_model)
        return self.dropout(x)


# --- Attention-based pooling ---
class AttentionPooling(nn.Module):
    """
    Attention-based pooling: learns to weight each time step of the sequence.
    """

    def __init__(self, d_model):
        super().__init__()
        # project each time-step embedding to a scalar score
        self.attn_proj = nn.Linear(d_model, 1)

    def forward(self, x, mask=None):
        """
        x: Tensor of shape (seq_len, batch_size, d_model)
        mask: optional Bool Tensor of shape (seq_len, batch_size) where True=valid
        """
        # compute raw attention scores for each time step
        scores = self.attn_proj(x)  # (seq_len, batch_size, 1)

        if mask is not None:
            mask_bool = mask.bool()
            # mask out padding positions: set their scores to -inf
            scores = scores.masked_fill(~mask_bool.unsqueeze(-1), float("-inf"))

        # normalize scores across time dimension
        weights = torch.softmax(scores, dim=0)  # (seq_len, batch_size, 1)

        # weighted sum of time-step embeddings
        pooled = (weights * x).sum(dim=0)  # (batch_size, d_model)
        return pooled


class TransformerTimeSeries(nn.Module):
    """
    input_dim = number of features in the dataset
    d_model options = 64, 128, 256
    n_heads = 2 or 4 for multi attention
    positional encoding = patients records should be in order
    """

    def __init__(self, input_dim=1, d_model=64, n_heads=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # Shared modules
        self.embedding = nn.Linear(in_features=input_dim, out_features=d_model)
        self.positional_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dropout=dropout, batch_first=False
        )
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)

        # Heads
        self.linear_layer = nn.Linear(
            in_features=d_model, out_features=1
        )  # classification
        self.reconstruction_head = nn.Linear(
            in_features=d_model, out_features=input_dim
        )  # pretraining

        # Pooling for classification
        self.attn_pool = AttentionPooling(d_model)

        # Learnable mask token for masked‑value modelling
        self.mask_token = nn.Parameter(torch.zeros(input_dim))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, x, mask=None):
        """Embed + positionally encode + transformer encode.

        Parameters
        ----------
        x : Tensor (S, B, F)
        mask : Bool Tensor (S, B) with True for valid entries.
        """
        x = self.embedding(x)
        x = self.positional_encoder(x)

        enc_mask = None
        if mask is not None:
            enc_mask = (~mask.bool()).transpose(0, 1)  # batch, seq_len

        h = self.encoder(x, src_key_padding_mask=enc_mask)
        return h

    # ------------------------------------------------------------------
    # Pretraining forward
    # ------------------------------------------------------------------

    def forward_reconstruction(self, x, mask=None):
        """Reconstruct original feature values for masked‑value modelling."""
        # Replace explicit zeros (masked positions) with learnable token
        token = self.mask_token.view(1, 1, -1)
        masked_positions = x == 0
        x = torch.where(masked_positions, token, x)

        h = self._encode(x, mask)
        return self.reconstruction_head(h)

    # ------------------------------------------------------------------
    # Classification forward (unchanged external interface)
    # ------------------------------------------------------------------

    def forward(self, x, mask=None):
        orig_mask = mask  # keep True=valid for pooling
        # x: (sequence_length, batch_size, feature_dim)
        x = self.embedding(x)  # (seq_len, batch_size, d_model)
        x = self.positional_encoder(x)  # (seq_len, batch_size, d_model)

        # prepare masks
        if orig_mask is not None:
            # encoder mask: True = padding
            enc_mask = (~orig_mask.bool()).transpose(0, 1)  # (batch_size, seq_len)
        else:
            enc_mask = None

        x = self.encoder(
            x, src_key_padding_mask=enc_mask
        )  # (seq_len, batch_size, d_model)

        # Pool over the sequence dimension (now dimension 0) to get one representation per batch element
        x = self.attn_pool(
            x, orig_mask
        )  # AttentionPooling expects True=valid (seq_len, batch)
        x = self.linear_layer(x).squeeze(-1)  # (batch_size)
        return x
