from collections import defaultdict

import torch
from torch.utils.data import Dataset

"""
    Will this patient develop sepsis at any point during their stay?
"""


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
        returns features and labels for each timestep

        example output for patient B:
        X = tensor([[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]])  # shape: (seq_len, feature_dim)
        y = tensor([0, 0, 1])  # shape: (seq_len,) - labels for each timestep
        """
        patient_records = self.patient_to_records[idx]
        X = torch.stack(
            [torch.tensor(record[0], dtype=torch.float32) for record in patient_records]
        )
        y = torch.tensor([record[1] for record in patient_records], dtype=torch.float32)

        return X, y


def collate_fn(batch):
    X_batch = [x for x, y in batch]
    y_batch = [y for x, y in batch]

    max_len = max([x.shape[0] for x in X_batch])
    feature_dim = X_batch[0].shape[1]

    padded_X = torch.zeros(len(X_batch), max_len, feature_dim)
    padded_y = torch.zeros(len(X_batch), max_len)  # Added: for label padding
    attention_mask = torch.ones(len(X_batch), max_len)

    for i, (x, y) in enumerate(zip(X_batch, y_batch)):
        padded_X[i, : x.shape[0], :] = x
        padded_y[i, : x.shape[0]] = y  # Added: pad labels same as features
        attention_mask[i, x.shape[0] :] = 0

    # transpose -> (sequence_length, batch_size, feature_dim)
    padded_X = padded_X.transpose(0, 1)
    padded_y = padded_y.transpose(0, 1)  # Added: transpose labels too
    attention_mask = attention_mask.transpose(0, 1).bool()

    return padded_X, padded_y, attention_mask


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
