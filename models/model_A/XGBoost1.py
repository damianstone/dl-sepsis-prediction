import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
import glob
from sklearn.metrics import roc_auc_score

# Load dataset
data_path = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\dataset\raw_combined_sofa24.parquet"
df = pd.read_parquet(data_path)
X = df.drop(columns=["SepsisLabel", "patient_id"])
y = df["SepsisLabel"]

# Output folder base
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "train_outputs")
os.makedirs(base_dir, exist_ok=True)

# Evaluation function
def recall_f1_eval(y_pred, dtrain, recall_weight=0.6):
    y_true = dtrain.get_label()
    y_pred_label = (y_pred > 0.5).astype(int)
    recall = recall_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)
    return "recall_f1_combo", recall_weight * recall + (1 - recall_weight) * f1

# Custom loss
def slse_loss(alpha=0.5):
    def loss(preds, dtrain):
        y_true = dtrain.get_label()
        eps = 1e-7
        grad = alpha * (preds - y_true) + (1 - alpha) * (np.log(preds + 1) - np.log(y_true + 1)) / (preds + 1)
        hess = alpha + (1 - alpha) * ((1 - np.log(preds + 1) + np.log(y_true + 1)) / ((preds + 1)**2))
        return grad, hess
    return loss

# Cross-validation
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold + 1}")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

    params = {
        "objective": "binary:logistic",
        "scale_pos_weight": scale_pos_weight,
        "max_depth": 10,
        "eta": 0.05,
        "gamma": 0.5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "lambda": 1,
        "alpha": 0.05
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=400,
        evals=[(dtest, "eval")],
        early_stopping_rounds=50,
        obj=slse_loss(alpha=0.6),
        feval=lambda y_pred, dtrain: recall_f1_eval(y_pred, dtrain, recall_weight=0.6),
        maximize=True,
        verbose_eval=20
    )

    y_probs = bst.predict(dtest)
    auc = roc_auc_score(y_test, y_probs)
    print(f"Fold {fold + 1}: AUROC = {auc:.4f}")
    
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    print(f" Fold {fold + 1}: Best F1-threshold = {best_thresh:.2f}")

    y_pred = (y_probs >= best_thresh).astype(int)
    report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T

    # Output path per fold
    existing = glob.glob(os.path.join(base_dir, "train_*"))
    existing_nums = [int(x.split("_")[-1]) for x in existing if x.split("_")[-1].isdigit()]
    next_id = max(existing_nums) + 1 if existing_nums else 1
    output_dir = os.path.join(base_dir, f"train_{next_id}")
    os.makedirs(output_dir, exist_ok=True)

    report_df.to_csv(os.path.join(output_dir, "report.csv"))

    thresholds_full = list(thresholds) + [1.0]
    precision = list(precision)[:len(thresholds_full)]
    recall = list(recall)[:len(thresholds_full)]
    f1_scores = 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall) + 1e-8)
    f1_scores = list(f1_scores)[:len(thresholds_full)]

    metrics_table = pd.DataFrame({
        "threshold": thresholds_full,
        "precision": precision,
        "recall": recall,
        "f1": f1_scores
    })
    
    metrics_table.to_csv(os.path.join(output_dir, "threshold_metrics.csv"), index=False)

    joblib.dump(bst, os.path.join(output_dir, "xgboost_model.pkl"))

    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision, label='PR Curve')
    plt.axvline(x=recall[best_idx], color='red', linestyle='--',
                label=f"Recall = {recall[best_idx]:.2f}")
    plt.plot(recall[best_idx], precision[best_idx], 'ro')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Fold {fold + 1} - PR Curve")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pr_curve.png"))
    plt.close()

    print(f"Fold {fold + 1} results saved to: {output_dir}")
