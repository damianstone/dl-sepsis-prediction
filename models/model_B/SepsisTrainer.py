import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall, F1Score

class SepsisTrainer:
    def __init__(self, model, optimizer, loss_fn, train_loader, val_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.best_threshold = 0.5
        
        self.t_acc = Accuracy(task='binary').to(device)
        self.t_prec = Precision(task='binary').to(device)
        self.t_rec = Recall(task='binary').to(device)
        self.t_f1 = F1Score(task='binary').to(device)

    def train_epoch(self):
        """Entrena el modelo durante una época."""
        self.model.train()
        epoch_loss, epoch_acc, epoch_prec, epoch_rec = 0, 0, 0, 0
        progress_bar = tqdm(self.train_loader, desc="Entrenando", leave=False)
        for X_batch, y_batch in progress_bar:
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            self.optimizer.zero_grad()
            y_logits = self.model(X_batch)
            loss = self.loss_fn(y_logits, y_batch.unsqueeze(1).float())
            loss.backward()
            self.optimizer.step()

            # En entrenamiento usamos threshold = 0.5
            y_probs = torch.sigmoid(y_logits)
            y_preds = (y_probs >= 0.5).float()
            acc = self.t_acc(y_preds, y_batch.unsqueeze(1).float())
            prec = self.t_prec(y_preds, y_batch.unsqueeze(1).float())
            rec = self.t_rec(y_preds, y_batch.unsqueeze(1).float())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_prec += prec.item()
            epoch_rec += rec.item()

            progress_bar.set_postfix({"Loss": loss.item(), "Acc": acc.item(), 
                                        "Prec": prec.item(), "Rec": rec.item()})
        n_batches = len(self.train_loader)
        return epoch_loss/n_batches, epoch_acc/n_batches, epoch_prec/n_batches, epoch_rec/n_batches

    def evaluate(self, threshold=None):
        """Evalúa el modelo en validación usando el threshold especificado (o el mejor ajustado)."""
        self.model.eval()
        if threshold is None:
            threshold = self.best_threshold
        epoch_loss, epoch_acc, epoch_prec, epoch_rec, epoch_f1 = 0, 0, 0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                y_logits = self.model(X_batch)
                loss = self.loss_fn(y_logits, y_batch.unsqueeze(1).float())
                y_probs = torch.sigmoid(y_logits)
                y_preds = (y_probs >= threshold).float()
                acc = self.t_acc(y_preds, y_batch.unsqueeze(1).float())
                prec = self.t_prec(y_preds, y_batch.unsqueeze(1).float())
                rec = self.t_rec(y_preds, y_batch.unsqueeze(1).float())
                f1 = self.t_f1(y_preds, y_batch.unsqueeze(1).float())
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                epoch_prec += prec.item()
                epoch_rec += rec.item()
                epoch_f1 += f1.item()
        n_batches = len(self.val_loader)
        return (epoch_loss/n_batches, epoch_acc/n_batches, 
                epoch_prec/n_batches, epoch_rec/n_batches, epoch_f1/n_batches)

    def adjust_threshold(self):
        """
        Ajusta el threshold iterando sobre un rango y eligiendo aquel que
        maximice la métrica F1 en el conjunto de validación.
        """
        self.model.eval()
        thresholds = np.linspace(0, 1, 21)  # de 0 a 1 en pasos de 0.05
        best_threshold = 0.5
        best_f1 = 0
        with torch.no_grad():
            for t in thresholds:
                f1_total, n_batches = 0, 0
                for X_batch, y_batch in self.val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    y_logits = self.model(X_batch)
                    y_probs = torch.sigmoid(y_logits)
                    y_preds = (y_probs >= t).float()
                    f1 = self.t_f1(y_preds, y_batch.unsqueeze(1).float())
                    f1_total += f1.item()
                    n_batches += 1
                avg_f1 = f1_total / n_batches if n_batches > 0 else 0
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_threshold = t
        self.best_threshold = best_threshold
        print(f"Threshold ajustado a: {best_threshold:.2f} con F1: {best_f1:.4f}")
        return best_threshold

    def train(self, epochs, adjust_every=5):
        """
        Entrena el modelo por varias épocas.
        Cada 'adjust_every' épocas se ajusta el threshold en validación.
        """
        for epoch in range(epochs):
            train_loss, train_acc, train_prec, train_rec = self.train_epoch()
            val_loss, val_acc, val_prec, val_rec, val_f1 = self.evaluate()
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train -> Loss: {train_loss:.5f} | Acc: {train_acc*100:.2f}% | " +
                  f"Prec: {train_prec*100:.2f}% | Rec: {train_rec*100:.2f}%")
            print(f"  Val   -> Loss: {val_loss:.5f} | Acc: {val_acc*100:.2f}% | " +
                  f"Prec: {val_prec*100:.2f}% | Rec: {val_rec*100:.2f}% | F1: {val_f1*100:.2f}%")
            if (epoch+1) % adjust_every == 0:
                self.adjust_threshold()
            print("-"*50)

# Ejemplo de uso:
# Asumiendo que ya tienes definidos: model, train_loader, val_loader, device, etc.
# pos_weight = torch.tensor([2], dtype=torch.float32).to(device)
# loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
#
# trainer = SepsisTrainer(model, optimizer, loss_fn, train_loader, val_loader, device)
# trainer.train(epochs=100, adjust_every=5)
