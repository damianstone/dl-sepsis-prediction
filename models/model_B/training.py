from pathlib import Path

import torch
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


def compute_masked_loss(y_logits, y_batch, attention_mask, loss_fn):
    # y_logits = (seq len, batch size)
    # y_batch = (seq len, batch size)
    # valid_mask = (seq len, batch size) -> true for valid positions
    valid_mask = ~attention_mask
    masked_outputs = y_logits[valid_mask]
    masked_targets = y_batch[valid_mask]
    return loss_fn(masked_outputs, masked_targets)


def validation_loop(model, val_loader, loss_fn, device, threshold):
    model.eval()
    val_loss = 0.0

    f1_hour = F1Score(task="binary").to(device)
    f2_hour = FBetaScore(task="binary", beta=2.0).to(device)
    prec_hour = Precision(task="binary").to(device)
    rec_hour = Recall(task="binary").to(device)
    acc_hour = Accuracy(task="binary").to(device)

    f2_patient = FBetaScore(task="binary", beta=2.0).to(device)

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
            # # padding values = 0
            # # push padded logits to −∞ so they never win the max
            # masked_logits = y_logits.masked_fill(attention_mask, -1e9)
            # # max‑pool over time axis → one logit per patient
            # patient_logits = masked_logits.max(dim=0).values
            # patient_prob = torch.sigmoid(patient_logits)
            # patient_pred = (patient_prob >= threshold).float()
            # # ground‑truth per patient: 1 if any hour label == 1
            # patient_true = y_batch.max(dim=0).values
            # f2_patient.update(patient_pred, patient_true)

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

    return val_loss, val_f2_patient


def training_loop(
    experiment_name, model, train_loader, val_loader, optimizer, loss_fn, epochs, device
):
    epoch_counter, loss_counter, acc_counter = [], [], []

    patience = 10
    best_f2_score = 0
    epochs_without_improvement = 0
    min_epochs = 20
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
            y_preds = (y_probs >= threshold).float()
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
            val_loss, val_f2_patient = validation_loop(
                model, val_loader, loss_fn, device, threshold
            )
            if epoch >= min_epochs:
                if val_f2_patient > best_f2_score:
                    best_f2_score = val_f2_patient
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
        "model": model,
    }
    return res


# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    print("not implemented")
    # train(model, train_loader, optimizer, loss_fn, epochs, device, threshold_update_n_batches)
