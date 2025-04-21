import torch
from torchmetrics import Accuracy
from tqdm import tqdm


def testing_loop(model, test_loader, loss_fn, device, threshold: float = 0.3):
    """Evaluate *model* on *test_loader* with patient‑level aggregation.

    The model returns per‑time‑step logits `(seq_len, batch)`. We convert these to a
    single patient‑level logit via **max over time** (equivalent to OR after
    sigmoid).
    """

    test_loss, test_acc = 0.0, 0.0
    model.eval()

    t_accuracy = Accuracy(task="binary").to(device)

    # Store predictions and labels for downstream analysis
    all_y_logits, all_y_probs, all_y_pred, all_y_test = [], [], [], []

    with torch.inference_mode():
        progress = tqdm(test_loader, desc="Testing", leave=False)

        for X_batch, y_batch, attention_mask in progress:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            attention_mask = attention_mask.to(device)

            # ---- forward ----
            y_logits_step = model(X_batch, mask=attention_mask)  # (seq_len, batch)

            # mask out padded positions so they cannot dominate the max
            if attention_mask is not None:
                valid_mask = attention_mask.bool()
                y_logits_step = y_logits_step.masked_fill(~valid_mask, float("-inf"))

            # patient‑level aggregation
            y_logits_patient, _ = y_logits_step.max(dim=0)  # (batch,)
            y_probs_patient = torch.sigmoid(y_logits_patient)
            y_pred_patient = (y_probs_patient >= threshold).float()

            # ---- loss & accuracy (patient level) ----
            loss = loss_fn(y_logits_patient, y_batch.float())
            acc = t_accuracy(y_pred_patient, y_batch.float())

            test_loss += loss.item()
            test_acc += acc.item()

            progress.set_postfix({"Loss": loss.item(), "Acc": acc.item()})

            # ---- store results ----
            all_y_logits.append(y_logits_patient.cpu())
            all_y_probs.append(y_probs_patient.cpu())
            all_y_pred.append(y_pred_patient.cpu())
            all_y_test.append(y_batch.cpu())

    if len(test_loader) > 0:
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

    print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")

    return all_y_logits, all_y_probs, all_y_pred, all_y_test


# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    print("not implemented")
    # testing_loop(model, train_loader, optimizer, loss_fn, epochs, device)
