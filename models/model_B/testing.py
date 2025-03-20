import torch
from torch import nn
from tqdm import tqdm
from torchmetrics import Accuracy

def testing_loop(model, test_loader, loss_fn, device, threshold=0.3):
    model.eval()
    test_loss, test_acc = 0, 0

    # Initialize metrics
    t_accuracy = Accuracy(task='binary').to(device)

    # Store predictions and labels
    all_y_logits, all_y_probs, all_y_pred, all_y_test = [], [], [], []

    with torch.inference_mode():
        progress_bar = tqdm(test_loader, desc="Testing", leave=False)
        
        for X_batch, y_batch, attention_mask in test_loader:
            X_batch, y_batch, attention_mask = (
                X_batch.to(device),
                y_batch.to(device),
                attention_mask.to(device)
            )

            # Forward pass
            y_logits = model(X_batch, mask=attention_mask)
            y_probs = torch.sigmoid(y_logits)
            y_pred = (y_probs >= threshold).float()

            # Compute loss and accuracy
            loss = loss_fn(y_logits.squeeze(), y_batch.float())
            acc = t_accuracy(y_pred.squeeze(), y_batch.float())

            test_loss += loss.item()
            test_acc += acc.item()

            progress_bar.set_postfix({"Loss": loss.item(), "Acc": acc.item()})

            # Store results
            all_y_logits.append(y_logits.cpu())
            all_y_probs.append(y_probs.cpu())
            all_y_pred.append(y_pred.cpu())
            all_y_test.append(y_batch.cpu())

    # Compute average metrics
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    print(f"Test Loss: {test_loss:.5f} | Test Accuracy: {test_acc:.2f}%")

    return all_y_logits, all_y_probs, all_y_pred, all_y_test

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    print("not implemented")
    # testing_loop(model, train_loader, optimizer, loss_fn, epochs, device)
