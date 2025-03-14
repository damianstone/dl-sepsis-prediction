import torch
from torch.utils.data import Dataset
from collections import defaultdict

# NOTE: purpose is to return in tensors and convert into sequences format + padding and marking

class SepsisPatientDataset(Dataset):
    def __init__(self, data, labels, patient_ids):
        """
        Args:
        - data: List of patient records (features)
        - labels: List of corresponding labels
        - patient_ids: List of patient IDs (for grouping)
        """
        self.data = data
        self.labels = labels
        self.patient_ids = patient_ids
        self.patient_to_records = self._group_by_patient()

    def _group_by_patient(self):
        """Groups records by patient ID."""
        patient_dict = defaultdict(list)
        for i, pid in enumerate(self.patient_ids):
            patient_dict[pid].append((self.data[i], self.labels[i]))
        return list(patient_dict.values())  # List of patient-specific lists

    def __len__(self):
        return len(self.patient_to_records)

    def __getitem__(self, idx):
        """Fetch patient records, pad sequences, and create a mask."""
        patient_records = self.patient_to_records[idx]

        # Convert to tensor
        X = torch.stack([torch.tensor(record[0], dtype=torch.float32) for record in patient_records])
        y = torch.tensor(patient_records[0][1], dtype=torch.float32)  # Use first record's label

        return X, y

def collate_fn(batch):
    """Pads sequences and generates an attention mask."""
    X_batch = [x for x, y in batch]  # Extract patient sequences
    y_batch = torch.stack([y for _, y in batch])  # Labels

    # Find max sequence length in batch
    max_len = max([x.shape[0] for x in X_batch])
    feature_dim = X_batch[0].shape[1]

    # Create padded tensor
    padded_X = torch.zeros(len(X_batch), max_len, feature_dim)
    attention_mask = torch.ones(len(X_batch), max_len)  # 1 = real data, 0 = padding

    # Copy original values and create mask
    for i, x in enumerate(X_batch):
        padded_X[i, :x.shape[0], :] = x
        attention_mask[i, x.shape[0]:] = 0  # Mark padding as 0 in mask

    return padded_X, y_batch, attention_mask


"""
dataset = SepsisPatientDataset(X_train, y_train, patient_ids)

batch_size = 32  # Adjust as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

for padded_X, y_batch, attention_mask in dataloader:
    outputs = model(padded_X, mask=attention_mask)
    loss = loss_fn(outputs.squeeze(), y_batch)
    loss.backward()
    optimizer.step()

"""