import torch
import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from pathlib import Path
from torch.utils.data import DataLoader

from preprocess import preprocess_data
from custom_dataset import SepsisPatientDataset, collate_fn
from architectures import TransformerClassifier

def find_project_root(marker=".gitignore"):
    """
    walk up from the current working directory until a directory containing the
    specified marker (e.g., .gitignore) is found.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}")

def get_config():
  pass

def save_model():
  pass

def load_model():
  pass

def full_pipeline():
  # TODO: get args: config_name_file
  
  
  # TODO: get config and set up device
  config = get_config()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # TODO: dataset
  X_train, X_test, y_train, y_test, patient_ids_train, patient_ids_test = preprocess_data(
      use_last_processed_data=True,
      data_file_name="big_imputed_sofa",
      train_sample_fraction=0.02
  )

  batch_size = 32
  dataset = SepsisPatientDataset(X_train.values, y_train.values, patient_ids_train.values)
  train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

  # TODO: model, loss function and optimizer
  in_dim = X_train.shape[1] 
  valid_heads = [h for h in range(1, in_dim + 1) if in_dim % h == 0]
  num_heads = valid_heads[-1] 
  print(num_heads)
  model = TransformerClassifier(input_dim=in_dim, num_heads=num_heads).to(device)
  # TODO: get pos_weight
  pos_weight = torch.tensor([2.3], dtype=torch.float32).to(device)
  loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
  optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

  # TODO: training loop
  
  # TODO: save the model creating a number ID which should i + 1 from the last previously saved
  
  # TODO: standarize all metric values from training loop and save them
  
  # TODO: load the last model for testing loop or can i used the just trained model?
  
  # TODO: testing loop
  
  # TODO: standarize all metric values from testing loop and save them
  
  # TODO: plot and save graphs


if __name__ == '__main__':
    print("not implemented yet")