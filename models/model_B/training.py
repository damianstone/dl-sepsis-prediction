from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, fbeta_score, precision_score, recall_score
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
    model.eval()
    val_loss = 0.0

    full_y_pred, full_y_true = [], []
    with torch.no_grad():
        for X_batch, y_batch, attention_mask in val_loader:
            X_batch, y_batch, attention_mask = (
                X_batch.to(device),
                y_batch.to(device),
                attention_mask.to(device),
            )

            y_logits = model(X_batch, mask=attention_mask)
            y_probs = torch.sigmoid(y_logits)
            y_preds = (y_probs >= threshold).float()

            loss = loss_fn(y_logits.squeeze(), y_batch.float())

            full_y_pred.append(y_preds.squeeze().cpu())
            full_y_true.append(y_batch.float().cpu())

            val_loss += loss.item()

    n_batches = len(val_loader)
    val_loss = val_loss / n_batches

    # Concatenate tensors and convert to numpy arrays
    full_y_pred = torch.cat(full_y_pred).numpy()
    full_y_true = torch.cat(full_y_true).numpy()
    # Compute metrics on the full validation set
    val_acc = accuracy_score(full_y_true, full_y_pred)
    val_prec = precision_score(full_y_true, full_y_pred, zero_division=0)
    val_rec = recall_score(full_y_true, full_y_pred, zero_division=0)

    return val_loss, val_acc, val_prec, val_rec, full_y_pred, full_y_true


def training_loop(
    experiment_name, model, train_loader, val_loader, optimizer, loss_fn, epochs, device
):
    epoch_counter, loss_counter, acc_counter = [], [], []

    patience = 10  # if the validation doesn't improve after K (patience) checks
    best_loss = float("inf")
    epochs_without_improvement = 0
    min_epochs = 10
    threshold = 0.5

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
            y_preds = (y_probs >= threshold).float()

            # compute loss
            loss = loss_fn(y_logits.squeeze(), y_batch.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

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

        if val_loader is not None and epoch % 2 == 0:
            val_loss, val_acc, val_prec, val_rec, full_y_pred, full_y_true = (
                validation_loop(model, val_loader, loss_fn, device, threshold)
            )
            print_validation_metrics(
                val_loss, val_acc, val_prec, val_rec, full_y_pred, full_y_true
            )
            if epoch >= min_epochs:
                if val_loss < best_loss:
                    best_loss = val_loss
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
        "best_loss": best_loss,
        "model": model,
    }
    return res


# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    print("not implemented")
    # train(model, train_loader, optimizer, loss_fn, epochs, device, threshold_update_n_batches)


"""
def validate_and_adjust_threshold(
        model,
        val_loader,
        loss_fn,
        device,
        t_precision,
        t_recall):
    model.eval()
    total_loss, total_acc, total_prec, total_rec = 0, 0, 0, 0
    n_batches = 0
    all_y_probs, all_y_true = [], []

    with torch.no_grad():
        for X_batch, y_batch, attention_mask in val_loader:
            X_batch, y_batch, attention_mask = (X_batch.to(
                device), y_batch.to(device), attention_mask.to(device))
            y_logits = model(X_batch, mask=attention_mask)
            loss = loss_fn(y_logits.squeeze(), y_batch.float())
            y_probs = torch.sigmoid(y_logits)

            # store probabilities and true labels for threshold adjustment
            all_y_probs.append(y_probs.cpu())
            all_y_true.append(y_batch.cpu())

            # compute metrics with default threshold 0.5
            y_preds = (y_probs >= 0.5).float()
            total_loss += loss.item()
            # using t_precision instance here for simplicity
            total_acc += t_precision(y_preds.squeeze(), y_batch.float()).item()
            total_prec += t_precision(y_preds.squeeze(), y_batch.float()).item()
            total_rec += t_recall(y_preds.squeeze(), y_batch.float()).item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    avg_prec = total_prec / n_batches
    avg_rec = total_rec / n_batches

    # Concatenate stored predictions and true labels
    # all_y_probs = torch.cat(all_y_probs, dim=0)
    # all_y_true  = torch.cat(all_y_true, dim=0)

    # NOTE::  using FB score as it weights recall more heavily than precision
    # thresholds=[0.3, 0.5]
    # best_threshold = 0.5
    # best_fbeta = 0
    # beta = 2 # recall will be 9x more important
    # for t in thresholds:
    #     y_preds_t = (all_y_probs >= t).float()
    #     prec = t_precision(y_preds_t.squeeze(), all_y_true.float())
    #     rec = t_recall(y_preds_t.squeeze(), all_y_true.float())
    #     fbeta = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-8)
    #     if fbeta.item() > best_fbeta:
    #         best_fbeta = fbeta.item()
    #         best_threshold = t

    print(
        f"validation -> loss: {avg_loss:.5f} | accuracy: {avg_acc:.2f}% | precision: {avg_prec:.2f}% | recall: {avg_rec:.2f}%")
    # print(f"Adjusted validation threshold to: {best_threshold:.2f} with FB: {best_fbeta:.4f}")
    return avg_loss, avg_acc, avg_prec, avg_rec, None

"""
