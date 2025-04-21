from collections import defaultdict

import torch
from torch.utils.data import Dataset


class SepsisPatientDataset(Dataset):
    def __init__(self, data, labels, patient_ids, time_index, prediction_window=6):
        """
        prediction_window: hours before clinical diagnosis to predict sepsis
        """
        self.data = data
        self.labels = labels
        self.patient_ids = patient_ids
        self.time_index = time_index  # index of ICULOS
        self.prediction_window = prediction_window
        self.patient_to_records = self._group_by_patient()

    def _group_by_patient(self):
        patient_dict = defaultdict(list)

        for i, pid in enumerate(self.patient_ids):
            # Store ICULOS time with the data and label
            patient_dict[pid].append(
                (self.data[i], self.labels[i], self.data[i][self.time_index])
            )

        # Sort by ICULOS and adjust labels for early prediction
        for pid in patient_dict:
            # Sort by time
            patient_dict[pid].sort(key=lambda x: x[2])  # sort by ICULOS

            records = patient_dict[pid]
            adjusted_labels = [0] * len(records)

            # Find the first sepsis onset
            sepsis_indices = [
                i for i, (_, label, _) in enumerate(records) if label == 1
            ]

            if sepsis_indices:  # If patient develops sepsis
                first_sepsis = sepsis_indices[0]
                sepsis_time = records[first_sepsis][2]  # ICULOS at sepsis onset

                # Mark records within prediction window as positive
                for i, (_, _, time) in enumerate(records):
                    time_to_sepsis = sepsis_time - time
                    if 0 <= time_to_sepsis <= self.prediction_window:
                        adjusted_labels[i] = 1

            # Create new records with adjusted labels
            adjusted_records = [
                (data, adj_label)
                for (data, _, _), adj_label in zip(records, adjusted_labels)
            ]

            patient_dict[pid] = adjusted_records

        return list(patient_dict.values())

    def __getitem__(self, idx):
        """
        Now returns sequences with adjusted labels for early prediction
        """
        patient_records = self.patient_to_records[idx]
        X = torch.stack(
            [torch.tensor(record[0], dtype=torch.float32) for record in patient_records]
        )
        # Use the adjusted labels that account for the prediction window
        y = torch.tensor([record[1] for record in patient_records], dtype=torch.float32)

        # Return the entire sequence and labels
        return (
            X,
            y.max(),
        )  # still using max as we want to predict if sepsis will occur within next 6h
