import torch
from torch import nn
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall
from architectures import TransformerClassifier


def adjust_threshold(y_probs, y_batch, t_precision, t_recall):
    """Dynamically adjusts the threshold by maximizing F1-score over different values."""
    best_thresh, best_f1 = 0.5, 0
    for t in torch.arange(0.1, 0.9, 0.1):
        y_preds = (y_probs >= t).float()
        
        prec, rec = t_precision(y_preds.squeeze(), y_batch), t_recall(y_preds.squeeze(), y_batch)
        f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh


def training_loop(model, train_loader, optimizer, loss_fn, epochs, device, threshold_update_n_batches):
    epoch_counter, loss_counter, acc_counter = [], [], []
    t_accuracy = Accuracy(task='binary').to(device)
    t_precision = Precision(task='binary').to(device)
    t_recall = Recall(task='binary').to(device)

    threshold = 0.5
    best_threshold = threshold
    best_f1 = 0 
    N = threshold_update_n_batches

    for epoch in range(epochs):
        model.train()
        epoch_loss, epoch_acc, epoch_prec, epoch_rec = 0, 0, 0, 0
        batch_count = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

        for X_batch, y_batch, attention_mask in train_loader:
            X_batch, y_batch, attention_mask = (
                X_batch.to(device),
                y_batch.to(device),
                attention_mask.to(device)
            )

            # Forward pass
            y_logits = model(X_batch, mask=attention_mask)
            y_probs = torch.sigmoid(y_logits)
            y_preds = (y_probs >= threshold).float()
            # compute loss
            loss = loss_fn(y_logits.squeeze(), y_batch.float())
            
            # metrics
            acc = t_accuracy(y_preds.squeeze(), y_batch.float())
            prec = t_precision(y_preds.squeeze(), y_batch.float())
            rec = t_recall(y_preds.squeeze(), y_batch.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # adjust threshold every N batches
            if batch_count % N == 0:
                new_threshold = float(adjust_threshold(y_probs, y_batch, t_precision, t_recall))
                new_f1 = float(2 * (prec * rec) / (prec + rec + 1e-8))

                if new_f1 > best_f1:
                    best_f1 = new_f1
                    best_threshold = new_threshold

                threshold = new_threshold

            batch_count += 1

            # update epoch metrics
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_prec += prec.item()
            epoch_rec += rec.item()

            progress_bar.set_postfix(
                {"Loss": loss.item(), "Acc": acc.item(), "Prec": prec.item(), "Rec": rec.item()})

            epoch_loss /= len(train_loader)
        
        epoch_acc /= len(train_loader)
        epoch_prec /= len(train_loader)
        epoch_rec /= len(train_loader)
        epoch_counter.append(epoch + 1)
        loss_counter.append(epoch_loss)
        acc_counter.append(epoch_acc)
            
        print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.5f} | Accuracy: {epoch_acc:.2f}% | Precision: {epoch_prec:.2f}% | Recall: {epoch_rec:.2f}%")

    # TODO: how to return the best threshold
    return epoch_counter, loss_counter, acc_counter, best_threshold

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    print("not implemented")
    # train(model, train_loader, optimizer, loss_fn, epochs, device, threshold_update_n_batches)
