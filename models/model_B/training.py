from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import fbeta_score
from torchmetrics import Accuracy, F1Score, FBetaScore, Precision, Recall
from tqdm import tqdm


def get_f2_score(y_pred, y_true):
    """Compute the F2-score given predictions and true labels.

    Args:
        y_pred (array-like): Predicted binary labels.
        y_true (array-like): True binary labels.

    Returns:
        float: F2-score.
    """
    return fbeta_score(y_true, y_pred, beta=2, zero_division=0)


def get_f1_score(y_pred, y_true):
    """Compute the F1-score given predictions and true labels.

    Args:
        y_pred (array-like): Predicted binary labels.
        y_true (array-like): True binary labels.

    Returns:
        float: F1-score.
    """
    return fbeta_score(y_true, y_pred, beta=1, zero_division=0)


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


def save_model(xperiment_name, model):
    project_root = find_project_root()
    model_path = Path(f"{project_root}/models/model_B/saved/")
    model_path.mkdir(exist_ok=True)
    model_file = model_path / f"{xperiment_name}.pth"
    torch.save(model.state_dict(), model_file)


def load_model(xperiment_name, model):
    project_root = find_project_root()
    model_path = Path(f"{project_root}/models/model_B/saved/")
    model_file = model_path / f"{xperiment_name}.pth"
    model.load_state_dict(torch.load(model_file))
    return model


def delete_model(xperiment_name):
    project_root = find_project_root()
    model_path = Path(f"{project_root}/models/model_B/saved/")
    model_file = model_path / f"{xperiment_name}.pth"
    model_file.unlink()


def print_validation_metrics(
    val_loss, val_acc, val_prec, val_rec, val_f1, val_f2, val_f2_patient
):
    print("\nValidation Metrics:")
    print(f"{'='*40}")
    print(f"Loss:       {val_loss:.2f}")
    print(f"Accuracy:   {val_acc*100:.2f}%")
    print(f"Precision:  {val_prec*100:.2f}%")
    print(f"Recall:     {val_rec*100:.2f}%")
    print(f"F1-Score:   {val_f1*100:.2f}%")
    print(f"F2-Score:   {val_f2*100:.2f}%")
    print(f"F2-PATIENTE Score:   {val_f2_patient*100:.2f}%")


def print_training_metrics(
    phase: str,
    epoch: int,
    epochs: int,
    loss: float,
    acc: float,
    prec: float,
    rec: float,
    f1: float,
    f2: float,
):
    print(
        f"[{phase:<5}] Epoch {epoch:>3}/{epochs:<3} | "
        f"loss {loss:.4f} | "
        f"acc {acc*100:5.1f} | "
        f"prec {prec*100:5.1f} | "
        f"rec {rec*100:5.1f} | "
        f"F1 {f1*100:5.1f} | "
        f"F2 {f2*100:5.1f}"
    )


def get_earliest_hour(probs_masked, patient_pred, threshold):
    cross = (probs_masked >= threshold).float()
    onset = cross.argmax(dim=0)
    onset[patient_pred == 0] = -1
    return onset


def build_time_weights(y_batch, pad_mask, window=3, neg_w=0.3):
    """
    y_batch  : (S, B) 0/1 labels
    pad_mask : (S, B) 1 = padding
    Returns  : weight matrix (S, B)
    """
    S, B = y_batch.shape
    w = torch.ones_like(y_batch, dtype=y_batch.dtype)

    for b in range(B):
        # indices (along seq_len) that are real data for patient b
        valid_idx = (~pad_mask[:, b]).nonzero(as_tuple=False).squeeze(-1)

        if valid_idx.numel() == 0:  # all-padding patient (shouldn’t happen)
            continue

        seq_valid = y_batch[valid_idx, b]  # 0/1 labels without padding

        # does this patient ever become septic?
        onset_rel = (seq_valid == 1).nonzero(as_tuple=False)
        if onset_rel.numel() == 0:
            continue  # never septic → keep weights = 1

        onset = valid_idx[onset_rel[0, 0]].item()  # onset index in full sequence

        # last `window` negative hours before onset
        early_zone = range(max(0, onset - window), onset)

        # down-weight those negatives
        for t in early_zone:
            if y_batch[t, b] == 0:
                w[t, b] = neg_w
    return w


# def compute_masked_loss(y_logits, y_batch, attention_mask, loss_fn):
#     # y_logits = (seq len, batch size)
#     # y_batch = (seq len, batch size)
#     valid_mask = ~attention_mask  # (seq len, batch size) -> true for valid positions
#     masked_outputs = y_logits[valid_mask]
#     masked_targets = y_batch[valid_mask]
#     return loss_fn(masked_outputs, masked_targets)


def compute_masked_loss(logits, targets, pad_mask, loss_fn, window=6, neg_w=0.5):
    """
    logits, targets : (S, B)
    pad_mask        : (S, B) 1 = padding
    window = 6      # look 6 h *before* that label-1
    neg_w  = 0.5    # pay only 10 % of normal penalty - as less more incentive to predict early
    """
    raw = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    weights = build_time_weights(targets, pad_mask, window, neg_w)
    valid = ~pad_mask
    return (raw * weights)[valid].mean()


def validation_loop(model, val_loader, loss_fn, device, threshold):
    model.eval()
    val_loss = 0.0

    f1_hour = F1Score(task="binary").to(device)
    f2_hour = FBetaScore(task="binary", beta=2.0).to(device)
    prec_hour = Precision(task="binary").to(device)
    rec_hour = Recall(task="binary").to(device)
    acc_hour = Accuracy(task="binary").to(device)

    f2_patient = FBetaScore(task="binary", beta=2.0).to(device)

    y_probs_list, y_true_list = [], []

    with torch.inference_mode():
        for X_batch, y_batch, attention_mask in val_loader:
            X_batch, y_batch, attention_mask = (
                X_batch.to(device),
                y_batch.to(device),
                attention_mask.to(device),
            )

            # Forward pass
            y_logits = model(X_batch, mask=attention_mask)
            y_probs = torch.sigmoid(y_logits)
            y_preds = (y_probs >= threshold).float()
            # make padded values = False
            valid = ~attention_mask

            # compute loss
            loss = compute_masked_loss(y_logits, y_batch, attention_mask, loss_fn)
            val_loss += loss.item()

            y_probs_list.append(y_probs[valid].cpu())
            y_true_list.append(y_batch[valid].cpu())

            # hour level metrics
            f1_hour.update(y_preds[valid], y_batch[valid])
            f2_hour.update(y_preds[valid], y_batch[valid])
            prec_hour.update(y_preds[valid], y_batch[valid])
            rec_hour.update(y_preds[valid], y_batch[valid])
            acc_hour.update(y_preds[valid], y_batch[valid])

            # NOTE: patient level metrics
            # any positive hour then patient with sepsis
            patient_pred = y_preds.masked_fill(attention_mask, 0).any(dim=0).float()
            patient_true = y_batch.masked_fill(attention_mask, 0).any(dim=0).float()
            f2_patient.update(patient_pred, patient_true)

    y_probs_all = torch.cat(y_probs_list).numpy()
    y_true_all = torch.cat(y_true_list).numpy()
    best_f2, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (y_probs_all >= thr).astype(int)
        f2 = get_f2_score(y_true_all, preds)
        if f2 > best_f2:
            best_f2, best_thr = f2, thr

    n_batches = len(val_loader)
    val_loss = val_loss / n_batches
    val_acc = acc_hour.compute().item()
    val_prec = prec_hour.compute().item()
    val_rec = rec_hour.compute().item()
    val_f1 = f1_hour.compute().item()
    val_f2 = f2_hour.compute().item()
    val_f2_patient = f2_patient.compute().item()

    print_validation_metrics(
        val_loss, val_acc, val_prec, val_rec, val_f1, val_f2, val_f2_patient
    )

    return val_loss, val_f2, best_thr


def training_loop(
    experiment_name, model, train_loader, val_loader, optimizer, loss_fn, epochs, device
):
    epoch_counter, loss_counter, acc_counter = [], [], []

    patience = 10
    best_f2_score = 0
    epochs_without_improvement = 0
    min_epochs = 5
    threshold = 0.5

    acc_hour = Accuracy(task="binary").to(device)
    prec_hour = Precision(task="binary").to(device)
    rec_hour = Recall(task="binary").to(device)
    f1_hour = F1Score(task="binary").to(device)
    f2_hour = FBetaScore(task="binary", beta=2.0).to(device)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for m in (acc_hour, prec_hour, rec_hour, f1_hour, f2_hour):
            m.reset()

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for X_batch, y_batch, attention_mask in train_loader:
            X_batch, y_batch, attention_mask = (
                X_batch.to(device),
                y_batch.to(device),
                attention_mask.to(device),
            )

            # X_batch = (seq, batch size, feature dim)
            # Y_batch = (batch size)

            # Forward pass
            y_logits = model(X_batch, mask=attention_mask)
            y_probs = torch.sigmoid(y_logits)
            y_preds = (y_probs >= 0.5).float()
            valid = ~attention_mask

            # compute loss
            loss = compute_masked_loss(y_logits, y_batch, attention_mask, loss_fn)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc_hour.update(y_preds[valid], y_batch[valid])
            prec_hour.update(y_preds[valid], y_batch[valid])
            rec_hour.update(y_preds[valid], y_batch[valid])
            f1_hour.update(y_preds[valid], y_batch[valid])
            f2_hour.update(y_preds[valid], y_batch[valid])

            epoch_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        epoch_loss /= len(train_loader)
        acc_h = acc_hour.compute().item()
        prec_h = prec_hour.compute().item()
        rec_h = rec_hour.compute().item()
        f1_h = f1_hour.compute().item()
        f2_hour.compute().item()

        # store for future plots
        epoch_counter.append(epoch + 1)
        loss_counter.append(epoch_loss)
        acc_counter.append(acc_h)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.5f} | F1: {f1_h*100:.2f}% | Accuracy: {acc_h*100:.2f}% | Precision: {prec_h*100:.2f}% | Recall: {rec_h*100:.2f}%"
        )

        # validation loop + early stopping
        if val_loader is not None:
            val_loss, val_f2, best_thr = validation_loop(
                model, val_loader, loss_fn, device, threshold
            )
            threshold = best_thr
            if epoch >= min_epochs:
                if val_f2 > best_f2_score:
                    best_f2_score = val_f2
                    epochs_without_improvement = 0
                    save_model(experiment_name, model)
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print("early stopping triggered")
                        break

    model = load_model(experiment_name, model)
    res = {
        "epoch_counter": epoch_counter,
        "loss_counter": loss_counter,
        "acc_counter": acc_counter,
        "best_f2_score": best_f2_score,
        "best_threshold": threshold,
        "model": model,
    }
    return res


# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    print("not implemented")
    # train(model, train_loader, optimizer, loss_fn, epochs, device, threshold_update_n_batches)
