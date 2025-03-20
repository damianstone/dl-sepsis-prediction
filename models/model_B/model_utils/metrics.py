import os
import json
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, roc_auc_score, f1_score, classification_report, recall_score


def save_metrics(
    root,
    xperiment_name,
    accuracy,
    y_test,
    y_probs,
    y_pred,
    best_threshold):
    save_path = f"{root}/models/model_B/results/{xperiment_name}"
    os.makedirs(save_path, exist_ok=True)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_probs)
    f1 = f1_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": round(float(accuracy), 3),
        "best_threshold": best_threshold,
        "AUC": round(float(auc_score), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1_score": round(float(f1), 3),
        "classification_report": report,
    }

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {save_path}/metrics.json")
