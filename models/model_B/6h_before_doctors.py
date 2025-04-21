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
        self.prediction_window = 6
        self.patient_to_records = self._group_by_patient()

    def _group_by_patient(self):
        """
        Groups records by patient and adjusts labels for early sepsis prediction.
        Records are sorted by ICULOS (time) and labels are modified to indicate
        'will develop sepsis within prediction_window hours'.

        Example with prediction_window = 2 hours:
        Input data:
        patient_ids = ["A", "A", "A", "B", "B", "B", "B"]
        data = [
            [0.1, 0.2], # A: ICULOS=1
            [0.3, 0.4], # A: ICULOS=2
            [0.5, 0.6], # A: ICULOS=3, sepsis onset
            [0.7, 0.8], # B: ICULOS=1
            [0.9, 1.0], # B: ICULOS=2
            [1.1, 1.2], # B: ICULOS=3, sepsis onset
            [1.3, 1.4]  # B: ICULOS=4
        ]
        labels = [0, 0, 1, 0, 0, 1, 1]

        Output:
        self.patient_to_records = [
            # Patient A: sepsis at ICULOS=3
            [([0.1, 0.2], 0),     # ICULOS=1: too early to predict
            ([0.3, 0.4], 1),     # ICULOS=2: within 2hr window before sepsis
            ([0.5, 0.6], 1)],    # ICULOS=3: sepsis onset, label stays 1

            # Patient B: sepsis at ICULOS=3
            [([0.7, 0.8], 0),     # ICULOS=1: too early to predict
            ([0.9, 1.0], 1),     # ICULOS=2: within 2hr window before sepsis
            ([1.1, 1.2], 1),     # ICULOS=3: sepsis onset
            ([1.3, 1.4], 1)]     # ICULOS=4: after sepsis onset, stays 1
        ]

        The adjusted labels (1) indicate either:
        1. Record is within prediction_window hours before sepsis onset
        2. Record is at or after sepsis onset
        """
        patient_dict = defaultdict(list)

        # Group records by patient with ICULOS
        for i, pid in enumerate(self.patient_ids):
            iculos_time = self.data[i][self.time_index]  # Get actual ICULOS value
            patient_dict[pid].append((self.data[i], self.labels[i], iculos_time))

        processed_records = []
        for pid in patient_dict:
            # Sort by actual ICULOS values
            records = sorted(patient_dict[pid], key=lambda x: x[2])

            # Initialize labels
            adjusted_labels = [0] * len(records)

            # Find sepsis onset(s)
            sepsis_indices = [
                i for i, (_, label, _) in enumerate(records) if label == 1
            ]

            if sepsis_indices:
                first_sepsis = sepsis_indices[0]
                sepsis_iculos = records[first_sepsis][2]  # Actual ICULOS time at sepsis

                # Mark labels considering actual time differences
                for i in range(len(records)):
                    current_iculos = records[i][2]  # Actual ICULOS time
                    if i < first_sepsis:
                        # Calculate time difference using actual ICULOS values
                        time_diff = sepsis_iculos - current_iculos
                        if 0 <= time_diff <= self.prediction_window:
                            adjusted_labels[i] = 1
                    else:
                        # At or after sepsis onset
                        adjusted_labels[i] = 1

            # Create final records with adjusted labels
            adjusted_records = [
                (data, adj_label)
                for (data, _, _), adj_label in zip(records, adjusted_labels)
            ]
            processed_records.append(adjusted_records)

        return processed_records

    def __len__(self):
        """
        returns the number of unique patients in the dataset

        example:
        if there are 100 patients, this returns 100
        """
        return len(self.patient_to_records)

    def __getitem__(self, idx):
        """
        Fetches all records for a given patient and converts them to tensors.
        Returns the full sequence of adjusted labels to preserve temporal information.

        Example output for patient B with 6-hour prediction window:
        X = tensor([[0.7, 0.8],    # ICULOS=1: too early
                    [0.9, 1.0],    # ICULOS=2: within window
                    [1.1, 1.2],    # ICULOS=3: sepsis onset
                    [1.3, 1.4]])   # ICULOS=4: after onset
        y = tensor([0, 1, 1, 1])   # Actual temporal progression
        """
        patient_records = self.patient_to_records[idx]
        X = torch.stack(
            [torch.tensor(record[0], dtype=torch.float32) for record in patient_records]
        )
        y = torch.tensor([record[1] for record in patient_records], dtype=torch.float32)

        return X, y


def collate_fn(batch):
    """
    Makes sequences the same length by padding shorter ones with zeros and
    creates masks to tell the transformer which values are real data versus padding.

    Args:
        batch: List of tuples (X, y) where:
            X is a tensor of shape (n_timesteps, n_features)
            y is a tensor of shape (n_timesteps,) - will be reduced to single label

    Returns:
        padded_X: Tensor of shape (sequence_length, batch_size, feature_dim)
        y_batch: Tensor of shape (batch_size,) - single label per sequence
        attention_mask: Boolean tensor of shape (sequence_length, batch_size)
    """
    # Separate X and y from the batch
    X_batch = [x for x, y in batch]
    # Take max of y for each sequence (if any timestep is 1, sequence is positive)
    y_batch = torch.tensor([y.max() for _, y in batch], dtype=torch.float32)

    # Get the maximum sequence length in this batch
    max_len = max([x.shape[0] for x in X_batch])
    feature_dim = X_batch[0].shape[1]
    batch_size = len(X_batch)

    # Initialize padded tensors
    padded_X = torch.zeros(batch_size, max_len, feature_dim)
    attention_mask = torch.zeros(batch_size, max_len)

    # Fill in the tensors with actual data
    for i, x in enumerate(X_batch):
        seq_len = x.shape[0]
        padded_X[i, :seq_len, :] = x
        attention_mask[i, :seq_len] = 1

    # Transpose to get (sequence_length, batch_size, feature_dim)
    padded_X = padded_X.transpose(0, 1)
    attention_mask = attention_mask.transpose(0, 1).bool()

    return padded_X, y_batch, attention_mask
