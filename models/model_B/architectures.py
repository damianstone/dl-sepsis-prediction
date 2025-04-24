import numpy as np
import torch
from torch import nn


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
        # change to (max_len, 1, d_model) to match sequence-first format
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (seq_len, batch_size, d_model)
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerTimeSeries(nn.Module):
    """
    input_dim = number of features in the dataset
    d_model options = 64, 128, 256
    n_heads = 2 or 4 for multi attention
    positional encoding = patients records should be in order
    """

    def __init__(self, input_dim=1, d_model=64, n_heads=2, n_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(in_features=input_dim, out_features=d_model)
        self.positional_encoder = PositionalEncoding(d_model, dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=n_layers)
        self.attention = nn.Linear(d_model, 1)
        self.linear_layer = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, x, mask=None):
        """
        predict if the patient have sepsis looking at all the time steps at once
        1 patient -> multiple time steps -> 1 prediction (global)

        1. input: multiple time steps for all patients in the batch
        2. the transformer processes the entire sequence of time steps for each patient at one, not one time step at a time
        3. the positional encoder ensures the model know the order of records
        4. attention pooling learns which time steps matter most and summarizes the sequence into a single embedding
        5. the model return one prediction per patient
        """
        # x: (sequence_length, batch_size, feature_dim)
        x = self.embedding(x)  # (seq_len, batch_size, d_model)
        x = self.positional_encoder(x)

        if mask is not None:
            mask = mask.transpose(0, 1)

        x = self.encoder(x, src_key_padding_mask=mask)

        # # gives more weight to later measurements
        # seq_len = x.size(0)
        # temporal_bias = torch.exp(torch.arange(seq_len, device=x.device) / seq_len)
        # # Shape: (seq_len, 1, 1)
        # temporal_bias = temporal_bias.unsqueeze(-1).unsqueeze(-1)

        # Attention-weighted pooling
        attention_weights = torch.softmax(self.attention(x), dim=0)
        x = torch.sum(x * attention_weights, dim=0)
        return self.linear_layer(x).squeeze(-1)


class TransformerClassifier(nn.Module):
    """
    Initila testing, we are not using this model
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
