import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve

def plot_roc_curve(y_true, y_probs):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    return fig

def plot_precision_recall_curve(y_true, y_probs):
    precision, recall, _ = precision_recall_curve(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(6,6))
    ax.plot(recall, precision, label="Precision-Recall Curve")
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend()
    return fig

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,6))
    disp = ConfusionMatrixDisplay(cm)
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title("Confusion Matrix")
    return fig

def plot_attention_heatmap(attention_weights, feature_names):
    avg_attention = np.mean(attention_weights, axis=0)  
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(avg_attention.reshape(1, -1), annot=True, cmap="viridis", xticklabels=feature_names, ax=ax)
    ax.set_title("Feature Importance (Attention Heatmap)")
    ax.set_yticks([])
    return fig

def plot_feature_importance(model, feature_names):
    importance = model.linear_layer.weight.detach().cpu().numpy().flatten()

    fig, ax = plt.subplots(figsize=(8,6))
    ax.barh(feature_names, importance)
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Features")
    ax.set_title("Feature Importance from Model Weights")
    return fig

def plot_loss_per_epoch(loss_counter):
    """Generates and saves the loss per epoch plot."""
    fig, ax = plt.subplots()
    ax.plot(range(1, len(loss_counter) + 1), loss_counter, marker='o', linestyle='-')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss per Epoch")
    ax.grid(True)
    return fig

def save_plots(
  root,
  xperiment_name,
  loss_counter,
  y_test, 
  y_probs, 
  y_pred, 
  model, 
  feature_names, 
  attention_weights):
    save_path = f"{root}/models/model_B/results/{xperiment_name}"
    os.makedirs(save_path, exist_ok=True)

    for plot_func, args, filename in [
        (plot_roc_curve, (y_test, y_probs), "roc_curve.png"),
        (plot_precision_recall_curve, (y_test, y_probs), "precision_recall_curve.png"),
        (plot_confusion_matrix, (y_test, y_pred), "confusion_matrix.png"),
        (plot_feature_importance, (model, feature_names), "feature_importance.png"),
        (plot_loss_per_epoch, (loss_counter,), "loss_per_epoch.png"),
    ]:
        fig = plot_func(*args)  
        fig.savefig(os.path.join(save_path, filename), bbox_inches='tight') 
        plt.close(fig)