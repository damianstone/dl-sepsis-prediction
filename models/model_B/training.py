from pathlib import Path

import numpy as np
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
    return model_file


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


def validation_loop(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0

    full_y_pred, full_y_true, full_y_probs = [], [], []
    with torch.no_grad():
        for X_batch, y_batch, attention_mask in val_loader:
            X_batch, y_batch, attention_mask = (
                X_batch.to(device),
                y_batch.to(device),
                attention_mask.to(device),
            )

            y_logits = model(X_batch, mask=attention_mask)
            y_probs = torch.sigmoid(y_logits)

            loss = loss_fn(y_logits.squeeze(), y_batch.float())

            full_y_true.append(y_batch.float().cpu())
            full_y_probs.append(y_probs.squeeze().cpu())
            val_loss += loss.item()

    n_batches = len(val_loader)
    val_loss = val_loss / n_batches

    # Concatenate tensors and convert to numpy arrays
    full_y_probs = torch.cat(full_y_probs).numpy()
    full_y_true = torch.cat(full_y_true).numpy()

    best_f2, best_thr = float("-inf"), 0.5
    for thr in np.linspace(0.01, 0.99, 99):
        preds = (full_y_probs >= thr).astype(int)
        f2 = get_f2_score(full_y_true, preds)
        if f2 > best_f2:
            best_f2, best_thr = f2, thr

    full_y_pred = (full_y_probs >= best_thr).astype(int)

    # Compute metrics on the full validation set
    val_acc = accuracy_score(full_y_true, full_y_pred)
    val_prec = precision_score(full_y_true, full_y_pred, zero_division=0)
    val_rec = recall_score(full_y_true, full_y_pred, zero_division=0)
    f2_score = get_f2_score(full_y_pred, full_y_true)

    return (
        val_loss,
        val_acc,
        val_prec,
        val_rec,
        f2_score,
        full_y_pred,
        full_y_true,
        best_thr,
    )


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
    min_epochs = 10
    best_threshold = 0.5

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_y_pred, epoch_y_true = [], []
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for X_batch, y_batch, attention_mask in train_loader:
            X_batch, y_batch, attention_mask = (
                X_batch.to(device),
                y_batch.to(device),
                attention_mask.to(device),
            )

            # Forward pass
            y_logits = model(X_batch, mask=attention_mask)
            y_probs = torch.sigmoid(y_logits)
            y_preds = (y_probs >= 0.5).float()

            # compute loss
            loss = loss_fn(y_logits.squeeze(), y_batch.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_count += 1

            # update epoch metrics and store predictions
            epoch_loss += loss.item()
            epoch_y_pred.append(y_preds.squeeze().cpu())
            epoch_y_true.append(y_batch.float().cpu())

            progress_bar.set_postfix({"Loss": loss.item()})

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
            (
                val_loss,
                val_acc,
                val_prec,
                val_rec,
                f2_score,
                full_y_pred,
                full_y_true,
                best_threshold,
            ) = validation_loop(model, val_loader, loss_fn, device)
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
        "best_threshold": best_threshold,
    }
    return res


# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    print("not implemented")
    # train(model, train_loader, optimizer, loss_fn, epochs, device, threshold_update_n_batches)
