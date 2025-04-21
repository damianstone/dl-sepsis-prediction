from pathlib import Path

import torch
from torchmetrics import F1Score, FBetaScore, Precision, Recall
from tqdm import tqdm


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
        f"Project root marker '{marker}' not found starting from {current}"
    )


def load_model(xperiment_name, model):
    project_root = find_project_root()
    model_path = Path(f"{project_root}/models/model_B/saved/")
    model_file = model_path / f"{xperiment_name}.pth"
    model.load_state_dict(torch.load(model_file))
    return model


def compute_masked_loss(y_logits, y_batch, attention_mask, loss_fn):
    """
    at every hour, predict whether this patient ever gets sepsis
    doesn't not learn when exactly the patient gets sepsis
    """
    # y_logits = (seq len, batch size)
    # y_batch = (batch size) -> one label per patient
    # valid_mask = (seq len, batch size) -> true for valid positions

    expanded_targets = y_batch.unsqueeze(0)  # 1, batch size
    # (seq len, batch size)
    expanded_targets = expanded_targets.expand(y_logits.size(0), y_logits.size(1))
    # padded positions = True (seq len, batch size)
    valid_mask = ~attention_mask  # make valid positions to be = True

    masked_outputs = y_logits[valid_mask]
    masked_targets = expanded_targets[valid_mask]

    return loss_fn(masked_outputs, masked_targets)


def testing_loop(model, test_loader, loss_fn, device, threshold=0.3):
    test_loss = 0
    model.eval()
    f1_score = F1Score(task="binary").to(device)
    f2_score = FBetaScore(task="binary", beta=2.0).to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)

    all_y_logits, all_y_probs, all_y_pred, all_y_test = [], [], [], []

    with torch.inference_mode():
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)

        for X_batch, y_batch, attention_mask in progress_bar:
            X_batch, y_batch, attention_mask = (
                X_batch.to(device),
                y_batch.to(device),
                attention_mask.to(device),
            )

            # Forward pass
            # (seq_len, batch_size)
            y_logits = model(X_batch, mask=attention_mask)

            valid = ~attention_mask
            # push padded positions to −∞  (never chosen by max)
            masked_logits = y_logits.masked_fill(valid, -1e9)
            patient_level_logits, _ = masked_logits.max(dim=0)
            patient_level_probs = torch.sigmoid(patient_level_logits)
            patient_level_preds = (patient_level_probs >= threshold).float()

            # compute loss
            loss = compute_masked_loss(y_logits, y_batch, attention_mask, loss_fn)

            # Update all metrics with masked predictions and targets
            f1 = f1_score(patient_level_preds, y_batch)
            f2 = f2_score(patient_level_preds, y_batch)
            prec = precision(patient_level_preds, y_batch)
            rec = recall(patient_level_preds, y_batch)

            test_loss += loss.item()

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Precision": f"{prec.item():.4f}",
                    "Recall": f"{rec.item():.4f}",
                    "F1": f"{f1.item():.4f}",
                    "F2": f"{f2.item():.4f}",
                }
            )

            all_y_logits.append(patient_level_logits.cpu())
            all_y_probs.append(patient_level_probs.cpu())
            all_y_pred.append(patient_level_preds.cpu())
            all_y_test.append(y_batch.cpu())

    # final metrics
    final_f1 = f1_score.compute()
    final_f2 = f2_score.compute()
    final_precision = precision.compute()
    final_recall = recall.compute()
    test_loss /= len(test_loader)

    print("\n" + "=" * 50)
    print("              TEST RESULTS              ")
    print("=" * 50)
    print(f"Loss       : {test_loss:.5f}")
    print(f"F1 Score   : {final_f1:.4f}")
    print(f"F2 Score   : {final_f2:.4f}")
    print(f"Precision  : {final_precision:.4f}")
    print(f"Recall     : {final_recall:.4f}")
    print("=" * 50 + "\n")

    return all_y_logits, all_y_probs, all_y_pred, all_y_test


# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    pass
    # import pandas as pd
    # import torch.nn as nn
    # from data.dataset import SepsisPatientDataset, collate_fn
    # from torch.utils.data import DataLoader

    # # get the yml experiment file
    # root = find_project_root()
    # experiment_file = f"{root}/models/model_B/config/time_series_transformer.yml"
    # with open(experiment_file, "r") as f:
    #     config = yaml.safe_load(f)

    # # Load test data
    # data_path = f"{root}/data/processed"
    # X_test = pd.read_csv(f"{data_path}/X_test.csv")
    # y_test = pd.read_csv(f"{data_path}/y_test.csv")
    # patient_ids_test = pd.read_csv(f"{data_path}/patient_ids_test.csv")

    # # Setup device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # # Create model instance
    # model = TransformerTimeSeries(
    #     input_dim=config["model"]["input_dim"],
    #     d_model=config["model"]["d_model"],
    #     nhead=config["model"]["nhead"],
    #     num_layers=config["model"]["num_layers"],
    #     dim_feedforward=config["model"]["dim_feedforward"],
    #     dropout=config["model"]["dropout"],
    # ).to(device)

    # # Create test dataloader
    # batch_size = config["testing"]["batch_size"]
    # dataset = SepsisPatientDataset(
    #     X_test.values,
    #     y_test.values,
    #     patient_ids_test.values,
    #     time_index=X_test.columns.get_loc("ICULOS"),
    # )
    # test_loader = DataLoader(
    #     dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=collate_fn,
    #     drop_last=True,
    # )

    # loss_fn = nn.BCEWithLogitsLoss(pos_weight=config["model"]["pos_weight"])

    # # Load model weights
    # model_path = f"{root}/models/model_B/saved/{config['xperiment']['name']}.pth"
    # print(f"Loading model from: {model_path}")
    # model.load_state_dict(torch.load(model_path))

    # # Run testing loop
    # print("\nStarting testing...")
    # all_y_logits, all_y_probs, all_y_pred, all_y_test = testing_loop(
    #     model=model,
    #     test_loader=test_loader,
    #     loss_fn=loss_fn,
    #     device=device,
    #     # default to 0.5 if not specified
    #     threshold=config["testing"].get("threshold", 0.5),
    # )
