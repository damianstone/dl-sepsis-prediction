from collections import defaultdict

import torch
from torch.utils.data import Dataset

# NOTE: purpose is to return in tensors and convert into sequences format + padding and masking


class SepsisPatientDataset(Dataset):
    def __init__(self, data, labels, patient_ids, time_index):
        """
        stores input data and groups patient records together

        example before grouping:
        patient_ids = ["A", "A", "B", "B", "B"]
        data = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]
        labels = [0, 1, 0, 0, 1]
        """
        self.data = data
        self.labels = labels
        self.patient_ids = patient_ids
        self.time_index = time_index  # index of ICULUS
        self.patient_to_records = self._group_by_patient()

    def _group_by_patient(self):
        """
        groups records by patient so they stay together during training

        example output:
        self.patient_to_records = [
            [([0.1, 0.2], 0), ([0.3, 0.4], 1)],  # patient A
            [([0.5, 0.6], 0), ([0.7, 0.8], 0), ([0.9, 1.0], 1)]  # patient B
        ]
        """
        patient_dict = defaultdict(list)

        for i, pid in enumerate(self.patient_ids):
            patient_dict[pid].append((self.data[i], self.labels[i]))

        # Sort each patient's records by ICULOS (ascending) - important for the positional encoding
        for pid in patient_dict:
            patient_dict[pid].sort(key=lambda x: x[0][self.time_index])

        return list(patient_dict.values())

    def __len__(self):
        """
        returns the number of unique patients in the dataset

        example:
        if there are 100 patients, this returns 100
        """
        return len(self.patient_to_records)

    def __getitem__(self, idx):
        """
        fetches all records for a given patient and converts them to tensors
        assigns a patient-level label (1 if any record has sepsis)

        example output for patient B:
        X = tensor([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])
        y = tensor(1)  # because at least one record has sepsis
        """
        patient_records = self.patient_to_records[idx]
        X = torch.stack(
            [torch.tensor(record[0], dtype=torch.float32) for record in patient_records]
        )
        y = torch.tensor([record[1] for record in patient_records], dtype=torch.float32)

        return X, y.max()


def collate_fn(batch):
    """
    makes sequences the same length by padding shorter ones with zeros and
    creates masks to tell the transformer which values are real data versus padding.

    this function basically just makes the data compatible with the transformer model
    in the following shape:
    (sequence_length, batch_size, feature_dim).

    Before padding, the data is in batch-first format:
      X_batch = [
          tensor([[0.1, 0.2], [0.3, 0.4]]),         # Patient A (2 records)
          tensor([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])  # Patient B (3 records)
      ]

    After padding (still in batch-first format):
      padded_X = [
          [[0.1, 0.2], [0.3, 0.4], [0.0, 0.0]],  # Patient A (padded to 3 records)
          [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]]   # Patient B (no padding needed)
      ]
      attention_mask = [
          [1, 1, 0],  # Patient A (third record is padding)
          [1, 1, 1]   # Patient B (all records are valid)
      ]

    After transposing to match the default transformer configuration:
      padded_X becomes:
        [
          [[0.1, 0.2], [0.5, 0.6]],  # Time step 1 for all patients
          [[0.3, 0.4], [0.7, 0.8]],  # Time step 2 for all patients
          [[0.0, 0.0], [0.9, 1.0]]   # Time step 3 (Patient A padded, Patient B valid)
        ]
      attention_mask becomes:
        [
          [1, 1],  # Time step 1
          [1, 1],  # Time step 2
          [0, 1]   # Time step 3
        ]

    final output shapes:
      padded_X: (sequence_length, batch_size, feature_dim)
      attention_mask: (sequence_length, batch_size)
    """
    X_batch = [x for x, y in batch]
    y_batch = torch.stack([y for _, y in batch])

    max_len = max([x.shape[0] for x in X_batch])
    feature_dim = X_batch[0].shape[1]

    padded_X = torch.zeros(len(X_batch), max_len, feature_dim)
    attention_mask = torch.ones(len(X_batch), max_len)

    for i, x in enumerate(X_batch):
        padded_X[i, : x.shape[0], :] = x
        attention_mask[i, x.shape[0] :] = 0

    # transpose -> (sequence_length, batch_size, feature_dim)
    padded_X = padded_X.transpose(0, 1)
    attention_mask = attention_mask.transpose(0, 1).bool()

    return padded_X, y_batch, attention_mask


"""
dataset = SepsisPatientDataset(X_train.values, y_train.values, patient_ids.values)

batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

for padded_X, y_batch, attention_mask in dataloader:
    outputs = model(padded_X, mask=attention_mask)
    loss = loss_fn(outputs.squeeze(), y_batch)
    loss.backward()
    optimizer.step()
"""
