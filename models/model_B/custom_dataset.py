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
        if any time step was positive sepsis we return Y = 1, making all the time steps to the positive
        """
        patient_records = self.patient_to_records[idx]
        X = torch.stack(
            [torch.tensor(record[0], dtype=torch.float32) for record in patient_records]
        )
        y = torch.tensor([record[1] for record in patient_records], dtype=torch.float32)

        return X, y


def collate_fn(batch):
    Xs, ys = zip(*batch)

    # max sequence length
    S = max(x.shape[0] for x in Xs)
    # feature dimension
    F = Xs[0].shape[1]
    # batch size
    B = len(Xs)

    # (seq length, batch size, feature dim)
    padded_X = torch.zeros(S, B, F)
    # (seq length, batch size)
    padded_y = torch.zeros(S, B)
    # (seq length, batch size)
    attention_mask = torch.zeros(S, B)

    for b, (x, y) in enumerate(zip(Xs, ys)):
        L = x.shape[0]
        padded_X[:L, b] = x
        padded_y[:L, b] = y
        attention_mask[:L, b] = 0  # valid data positions = 0
        attention_mask[L:, b] = 1  # padding positions = 1 = True

    return padded_X, padded_y, attention_mask.bool()


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
