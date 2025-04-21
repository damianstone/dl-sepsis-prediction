from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a simple linear warmup + linear decay scheduler.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # after warmup, decay linearly to zero
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


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
    val_loss, val_acc, val_prec, val_rec, full_y_pred, full_y_true
):
    print("\nValidation Metrics:")
    print(f"{'='*40}")
    print(f"Loss:       {val_loss:.2f}")
    print(f"Accuracy:   {val_acc*100:.2f}%")
    print(f"Precision:  {val_prec*100:.2f}%")
    print(f"Recall:     {val_rec*100:.2f}%")
    val_f1 = get_f1_score(full_y_pred, full_y_true)
    val_f2 = get_f2_score(full_y_pred, full_y_true)
    print(f"F1-Score:   {val_f1*100:.2f}%")
    print(f"F2-Score:   {val_f2*100:.2f}%")


def validation_loop(model, val_loader, loss_fn, device, threshold):
    """Run model on validation set and compute patient‑level metrics.

    The model returns per‑time‑step logits with shape (seq_len, batch).
    For patient‑level prediction we aggregate with **max over time** which
    corresponds to logical OR after sigmoid, boosting recall (same strategy
    Martin used).
    """

    model.eval()
    val_loss = 0.0

    full_y_pred, full_y_true = [], []

    with torch.no_grad():
        for X_batch, y_batch, attention_mask in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            attention_mask = attention_mask.to(device)

            # ---- forward ----
            y_logits_step = model(X_batch, mask=attention_mask)  # (seq_len, batch)

            # mask out padded positions so they do not influence the max
            if attention_mask is not None:
                valid_mask = attention_mask.bool()
                y_logits_step = y_logits_step.masked_fill(~valid_mask, float("-inf"))

            # Patient‑level aggregation (max over time)
            y_logits_patient, _ = y_logits_step.max(dim=0)  # (batch,)

            y_probs_patient = torch.sigmoid(y_logits_patient)
            y_pred_patient = (y_probs_patient >= threshold).float()

            # ----- loss (patient‑level) -----
            loss = loss_fn(y_logits_patient, y_batch.float())

            # store
            full_y_pred.append(y_pred_patient.cpu())
            full_y_true.append(y_batch.float().cpu())
            val_loss += loss.item()

    n_batches = len(val_loader)
    val_loss = val_loss / n_batches if n_batches else 0.0

    # Concatenate tensors and convert to numpy arrays for metric computation
    full_y_pred = torch.cat(full_y_pred).numpy()
    full_y_true = torch.cat(full_y_true).numpy()

    val_acc = accuracy_score(full_y_true, full_y_pred)
    val_prec = precision_score(full_y_true, full_y_pred, zero_division=0)
    val_rec = recall_score(full_y_true, full_y_pred, zero_division=0)
    f2_score = get_f2_score(full_y_pred, full_y_true)

    return val_loss, val_acc, val_prec, val_rec, f2_score, full_y_pred, full_y_true


def training_loop(
    experiment_name, model, train_loader, val_loader, optimizer, loss_fn, epochs, device
):
    epoch_counter, loss_counter, acc_counter = [], [], []

    # set up LR scheduler: 10% of total steps as warmup
    warmup_epochs = 10
    warmup_steps = int(warmup_epochs * len(train_loader))
    total_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    patience = 10  # if the validation doesn't improve after K (patience) checks
    best_f2_score = 0
    epochs_without_improvement = 0
    min_epochs = 50
    threshold = 0.5

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_y_pred, epoch_y_true = [], []
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for X_batch, y_batch, attention_mask in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)  # (batch,)
            attention_mask = attention_mask.to(device)  # (seq_len, batch)

            # ---- forward ----
            y_logits = model(X_batch, mask=attention_mask)
            # y_logits: (seq_len, batch)

            # Build per‑step labels
            y_labels = y_batch.unsqueeze(0).expand_as(y_logits)  # (seq_len, batch)

            # Mask padding steps
            valid_mask = attention_mask.bool()
            logits_flat = y_logits[valid_mask]  # (#valid,)
            labels_flat = y_labels[valid_mask].float()  # (#valid,)

            # ---- loss ----
            loss = loss_fn(logits_flat, labels_flat)

            # ---- backward ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # ---- track metrics ----
            epoch_loss += loss.item()
            preds_flat = (torch.sigmoid(logits_flat) >= threshold).float()
            epoch_y_pred.append(preds_flat.cpu())
            epoch_y_true.append(labels_flat.cpu())

        epoch_loss /= len(train_loader)
        epoch_y_pred = torch.cat(epoch_y_pred).numpy()
        epoch_y_true = torch.cat(epoch_y_true).numpy()
        epoch_acc = accuracy_score(epoch_y_true, epoch_y_pred)
        epoch_prec = precision_score(epoch_y_true, epoch_y_pred, zero_division=0)
        epoch_rec = recall_score(epoch_y_true, epoch_y_pred, zero_division=0)
        epoch_counter.append(epoch + 1)
        loss_counter.append(epoch_loss)
        acc_counter.append(epoch_acc)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.5f} | Accuracy: {epoch_acc*100:.2f}% | Precision: {epoch_prec*100:.2f}% | Recall: {epoch_rec*100:.2f}%"
        )

        if val_loader is not None and epoch % 1 == 0:
            val_loss, val_acc, val_prec, val_rec, f2_score, full_y_pred, full_y_true = (
                validation_loop(model, val_loader, loss_fn, device, threshold)
            )
            print_validation_metrics(
                val_loss, val_acc, val_prec, val_rec, full_y_pred, full_y_true
            )
            if epoch >= min_epochs:
                if f2_score > best_f2_score:
                    best_f2_score = f2_score
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
