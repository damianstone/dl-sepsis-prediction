import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from torch import nn

class TransformerClassifier(nn.Module):
    """
    num_heads = more heads capture different attention but increase computation
    num_layers = more make the model deeper but can overfit if too high
    """
    def __init__(self, input_dim, num_heads=4, num_layers=2):
        super().__init__()
        # d_model = input_dim (number of features)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        # stacks multiple encoder layers (num_layers controls depth)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # linear layer to map the output to a single value (binary classification)
        self.linear_layer = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
      z = self.encoder(x)  # Output shape: (batch_size, seq_len, features) or (batch_size, features)
      if z.dim() == 3:  # If (batch_size, seq_len, features), take the last timestep
            # NOTE: last timestep -> the most recent ICU data is usually the most relevant for prediction
            z = z[:, -1, :]
      return self.linear_layer(z)