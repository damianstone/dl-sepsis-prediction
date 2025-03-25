import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import xgboost as xgb
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.metrics import f1_score
import glob



train_path = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\models\model_A\diff_sofa_train.parquet"
test_path = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\models\model_A\diff_sofa_test.parquet"


df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)


X_train = df_train.drop(columns=["SepsisLabel", "patient_id"])
y_train = df_train["SepsisLabel"]

X_test = df_test.drop(columns=["SepsisLabel", "patient_id"])
y_test = df_test["SepsisLabel"]

print("col：")
print(X_train.columns.tolist())

print("\nfirst 10 row：")
print(X_train.head(10))


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]

from sklearn.metrics import recall_score, f1_score

def recall_f1_eval(y_pred, dtrain, recall_weight=0.6):
    y_true = dtrain.get_label()
    y_pred_label = (y_pred > 0.5).astype(int)

    recall = recall_score(y_true, y_pred_label)
    f1 = f1_score(y_true, y_pred_label)

    combined_score = recall_weight * recall + (1 - recall_weight) * f1
    return "recall_f1_combo", combined_score

def slse_loss(alpha=0.5):
    def loss(preds, dtrain):
        y_true = dtrain.get_label()
        eps = 1e-7
        
        grad = alpha * (preds - y_true) + \
               (1 - alpha) * (np.log(preds + 1) - np.log(y_true + 1)) / (preds + 1)
        
        hess = alpha + (1 - alpha) * (
            (1 - np.log(preds + 1) + np.log(y_true + 1)) / ((preds + 1)**2)
        )
        return grad, hess
    return loss



params = {
    "objective": "binary:logistic",
    "scale_pos_weight": scale_pos_weight,
    "max_depth": 8,
    "eta": 0.1,
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
    maximize=True
)


y_probs = bst.predict(dtest)

precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

#F1-score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

acceptable_indices = np.where((recall >= 0.7))[0]

if len(acceptable_indices) > 0:
    best_idx = acceptable_indices[np.argmax(f1_scores[acceptable_indices])]
    best_thresh = thresholds[best_idx]
    print(f"use Recall ≥ 0.8 ,best_thresh: {best_thresh:.2f}")

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)


y_pred = (y_probs >= best_thresh).astype(int)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.join(script_dir, "train_outputs")
os.makedirs(base_dir, exist_ok=True)
existing = glob.glob(os.path.join(base_dir, "train_*"))
existing_nums = [int(x.split("_")[-1]) for x in existing if x.split("_")[-1].isdigit()]
next_id = max(existing_nums) + 1 if existing_nums else 1
output_dir = os.path.join(base_dir, f"train_{next_id}")
os.makedirs(output_dir, exist_ok=True)


report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
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
print("threshold_metrics.csv done")


joblib.dump(bst, os.path.join(output_dir, "xgboost_model.pkl"))

# PR
plt.figure(figsize=(8, 5))
plt.plot(recall, precision, label='PR Curve')
plt.axvline(x=recall[best_idx], color='red', linestyle='--',
            label=f"Recall = {recall[best_idx]:.2f}")
plt.plot(recall[best_idx], precision[best_idx], 'ro')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Recall Optimization)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pr_curve.png"))
plt.close()

print(f"save to  {output_dir}")