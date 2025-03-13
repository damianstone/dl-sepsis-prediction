import torch
import os
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, classification_report

FEATURE_NAMES = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BaseExcess',
                                'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos',
                                'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 'Glucose',
                                'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total',
                                'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets',
                                'Age', 'Gender', 'HospAdmTime', 'ICULOS', "SOFA"]

def save_eval_csv(df, eval):
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    save_path = f"{project_root}/models/model_B/results/{eval}"
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(f"{project_root}/models/model_B/results/{eval}/{eval}.csv", index=False)

########################################## DATASET #############################################################
def get_original_dataset_tensors(data_path="imputed_sofa.parquet", save_path="dataset_tensors.pth"):
    """
    normal 80/20 split from a dataset
    """
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    tensor_ds_path = f"{project_root}/dataset/{save_path}"
    if os.path.exists(tensor_ds_path):
        data = torch.load(tensor_ds_path)
        print("from saved dataset")
        return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    imputed_df = pd.read_parquet(f"{project_root}/dataset/{data_path}")

    X = imputed_df.drop(columns=['SepsisLabel']).values
    y = imputed_df['SepsisLabel'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(
        X_test, dtype=torch.float32)
    y_train, y_test = torch.tensor(
        y_train, dtype=torch.long), torch.tensor(y_test, dtype=torch.long)

    path = f"{project_root}/dataset/{save_path}"
    torch.save({"X_train": X_train, "X_test": X_test,
               "y_train": y_train, "y_test": y_test}, path)

    return X_train, X_test, y_train, y_test

def get_full_balanced_dataset_tensors(data_path="balanced_dataset.parquet", save_path="full_balanced_tensors.pth"):
    current_dir = os.getcwd()
    project_root = os.path.abspath(os.path.join(current_dir, "../.."))
    tensor_ds_path = f"{project_root}/dataset/{save_path}"
    if os.path.exists(tensor_ds_path):
        data = torch.load(tensor_ds_path)
        print("from saved dataset")
        return data["X_train"], data["X_test"], data["y_train"], data["y_test"]

    imputed_df = pd.read_parquet(f"{project_root}/dataset/{data_path}")

    X = imputed_df.drop(columns=['SepsisLabel']).values
    y = imputed_df['SepsisLabel'].values

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)

    path = f"{project_root}/dataset/{save_path}"
    torch.save({"X_train": X_train,
               "y_train": y_train}, path)

    return X_train, y_train

def get_train_val_loaders(X_train, y_train, batch_size=32, val_split=0.2):
    dataset = TensorDataset(X_train, y_train)
    n_total = len(dataset)
    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    train_dataset, val_dataset = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def display_balance_statistics(df):
    patient_df = df.groupby("patient_id")["SepsisLabel"].max().reset_index()
    counts = patient_df["SepsisLabel"].value_counts()
    total_patients = counts.sum()
    print("Patient-level balance statistics:")
    print("Total patients:", total_patients)
    for label, count in counts.items():
        perc = (count / total_patients) * 100
        print(f"Label {label}: {count} patients ({perc:.2f}%)")
    if len(counts) >= 2:
        imbalance_ratio = counts.max() / counts.min()
        print(f"Imbalance ratio (majority/minority): {imbalance_ratio:.2f}")


########################################## PLOTS #############################################################

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

def save_plots(y_test, y_probs, y_pred, model, feature_names, attention_weights, eval):
    notebook_dir = os.getcwd()
    root = os.path.abspath(os.path.join(notebook_dir, "../.."))
    save_path = f"{root}/models/model_B/results/{eval}"
    os.makedirs(save_path, exist_ok=True)

    for plot_func, args, filename in [
        (plot_roc_curve, (y_test, y_probs), "roc_curve.png"),
        (plot_precision_recall_curve, (y_test, y_probs), "precision_recall_curve.png"),
        (plot_confusion_matrix, (y_test, y_pred), "confusion_matrix.png"),
        (plot_feature_importance, (model, feature_names), "feature_importance.png"),
    ]:
        fig = plot_func(*args)  
        fig.savefig(os.path.join(save_path, filename), bbox_inches='tight') 
        plt.close(fig)

########################################## METRICS #############################################################
def save_metrics(y_test, 
                 y_probs, 
                 y_pred, 
                 eval):
    notebook_dir = os.getcwd()
    root = os.path.abspath(os.path.join(notebook_dir, "../.."))
    save_path = f"{root}/models/model_B/results/{eval}"
    os.makedirs(save_path, exist_ok=True)

    # Compute precision, recall, thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)  # Avoid division by zero
    best_threshold = thresholds[np.argmax(f1_scores)] if len(thresholds) > 0 else 0.5

    # Compute overall metrics
    auc_score = roc_auc_score(y_test, y_probs)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "AUC": round(float(auc_score), 4),
        "Best Threshold": round(float(best_threshold), 4),
        "Precision at Best Threshold": round(float(precision[np.argmax(f1_scores)]), 4),
        "Recall at Best Threshold": round(float(recall[np.argmax(f1_scores)]), 4),
        "F1 Score": round(float(f1), 4),
        "Classification Report": report,
    }

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {save_path}/metrics.json")