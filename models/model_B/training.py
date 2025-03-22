import torch
from torch import nn
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall
from architectures import TransformerClassifier
import numpy as np

# NOTE: old
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

def adjust_threshold_val(model, val_loader, device, t_precision, t_recall):
    # adjust threshold on validation set by iterating over possible thresholds and selecting the one that maximizes f1
    model.eval()
    thresholds = np.linspace(0, 1, 3)  # thresholds from 0 to 1 in steps of 0.05
    best_threshold = 0.5
    best_f1 = 0
    with torch.no_grad():
        for t in thresholds:
            f1_total = 0
            n_batches = 0
            for X_batch, y_batch, attention_mask in val_loader:
                X_batch, y_batch, attention_mask = (
                    X_batch.to(device),
                    y_batch.to(device),
                    attention_mask.to(device)
                )
                y_logits = model(X_batch, mask=attention_mask)
                y_probs = torch.sigmoid(y_logits)
                y_preds = (y_probs >= t).float()
                prec = t_precision(y_preds.squeeze(), y_batch.float())
                rec = t_recall(y_preds.squeeze(), y_batch.float())
                f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
                f1_total += f1.item()
                n_batches += 1
            avg_f1 = f1_total / n_batches if n_batches > 0 else 0
            if avg_f1 > best_f1:
                best_f1 = avg_f1
                best_threshold = t
    print(f"validation threshold adjusted to: {best_threshold:.2f} with f1: {best_f1:.4f}")
    return best_threshold

def validate_loop(model, val_loader, loss_fn, device, threshold):
    model.eval()
    val_loss, val_acc, val_prec, val_rec = 0, 0, 0, 0
    t_accuracy = Accuracy(task='binary').to(device)
    t_precision = Precision(task='binary').to(device)
    t_recall = Recall(task='binary').to(device)
    
    with torch.no_grad():
        for X_batch, y_batch, attention_mask in val_loader:
            X_batch, y_batch, attention_mask = (
                X_batch.to(device),
                y_batch.to(device),
                attention_mask.to(device)
            )
            y_logits = model(X_batch, mask=attention_mask)
            y_probs = torch.sigmoid(y_logits)
            y_preds = (y_probs >= threshold).float()
            loss = loss_fn(y_logits.squeeze(), y_batch.float())
            acc = t_accuracy(y_preds.squeeze(), y_batch.float())
            prec = t_precision(y_preds.squeeze(), y_batch.float())
            rec = t_recall(y_preds.squeeze(), y_batch.float())
            
            val_loss += loss.item()
            val_acc += acc.item()
            val_prec += prec.item()
            val_rec += rec.item()
    
    n_batches = len(val_loader)
    return val_loss / n_batches, val_acc / n_batches, val_prec / n_batches, val_rec / n_batches

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
            X_batch, y_batch, attention_mask = (X_batch.to(device), y_batch.to(device), attention_mask.to(device))
            y_logits = model(X_batch, mask=attention_mask)
            loss = loss_fn(y_logits.squeeze(), y_batch.float())
            y_probs = torch.sigmoid(y_logits)
            
            # store probabilities and true labels for threshold adjustment
            all_y_probs.append(y_probs.cpu())
            all_y_true.append(y_batch.cpu())
            
            # compute metrics with default threshold 0.5
            y_preds = (y_probs >= 0.5).float()
            total_loss += loss.item()
            total_acc += t_precision(y_preds.squeeze(), y_batch.float()).item()  # using t_precision instance here for simplicity
            total_prec += t_precision(y_preds.squeeze(), y_batch.float()).item()
            total_rec += t_recall(y_preds.squeeze(), y_batch.float()).item()
            n_batches += 1

    avg_loss = total_loss / n_batches
    avg_acc  = total_acc / n_batches
    avg_prec = total_prec / n_batches
    avg_rec  = total_rec / n_batches

    # Concatenate stored predictions and true labels
    all_y_probs = torch.cat(all_y_probs, dim=0)
    all_y_true  = torch.cat(all_y_true, dim=0)
    
    # TODO::  weighs recall more heavily than precision
    thresholds=[0.3, 0.5]
    best_threshold = 0.5
    best_fbeta = 0
    beta = 2  # f2 score, giving more weight to recall
    for t in thresholds:
        y_preds_t = (all_y_probs >= t).float()
        prec = t_precision(y_preds_t.squeeze(), all_y_true.float())
        rec = t_recall(y_preds_t.squeeze(), all_y_true.float())
        fbeta = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec + 1e-8)
        if fbeta.item() > best_fbeta:
            best_fbeta = fbeta.item()
            best_threshold = t

    print(f"validation -> loss: {avg_loss:.5f} | accuracy: {avg_acc:.2f}% | precision: {avg_prec:.2f}% | recall: {avg_rec:.2f}%")
    print(f"Adjusted validation threshold to: {best_threshold:.2f} with F2: {best_fbeta:.4f}")
    return avg_loss, avg_acc, avg_prec, avg_rec, best_threshold

def training_loop(
    model, 
    train_loader,
    val_loader, 
    optimizer, 
    loss_fn, 
    epochs, 
    device, 
    threshold_update_n_batches):
    # tracking shit
    epoch_counter, loss_counter, acc_counter = [], [], []
    t_accuracy = Accuracy(task='binary').to(device)
    t_precision = Precision(task='binary').to(device)
    t_recall = Recall(task='binary').to(device)

    # for early stopping and threshold selection
    patience = 5
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    min_delta = 0.001  # minimum required improvement
    min_epochs = 50    # don't check early stopping before this epoch
    
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
            # if N > 0 and batch_count % N == 0:
            #     new_threshold = float(adjust_threshold(y_probs, y_batch, t_precision, t_recall))
            #     new_f1 = float(2 * (prec * rec) / (prec + rec + 1e-8))

            #     if new_f1 > best_f1:
            #         best_f1 = new_f1
            #         best_threshold = new_threshold

            #     threshold = new_threshold

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
        
        # TODO: run evalution every 2-3 epochs
        if val_loader is not None and epoch % 10 == 0:
            # best_threshold = adjust_threshold_val(model, val_loader, device, t_precision, t_recall)
            # threshold = best_threshold 
            # val_loss, val_acc, val_prec, val_rec = validate_loop(model, val_loader, loss_fn, device, threshold)
            # print(f"validation -> loss: {val_loss:.5f} | accuracy: {val_acc:.2f}% | precision: {val_prec:.2f}% | recall: {val_rec:.2f}%")

            val_loss, val_acc, val_prec, val_rec, best_threshold = validate_and_adjust_threshold(
                model, val_loader, loss_fn, device, t_precision, t_recall
            )
            threshold = best_threshold
            
            # TODO: fix early stopping to not be that early
            if epoch >= min_epochs:
                if best_val_loss - val_loss > min_delta:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= patience:
                        print("early stopping triggered")
                        break
                
    return epoch_counter, loss_counter, acc_counter, best_threshold

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    print("not implemented")
    # train(model, train_loader, optimizer, loss_fn, epochs, device, threshold_update_n_batches)
