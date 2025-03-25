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


# STEP 1: 指定 parquet 文件路径
train_path = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\models\model_A\diff_sofa_train.parquet"
test_path = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\models\model_A\diff_sofa_test.parquet"

# STEP 2: 读取 parquet 文件为 DataFrame
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# STEP 3: 拆分特征与标签
X_train = df_train.drop(columns=["SepsisLabel", "patient_id"])
y_train = df_train["SepsisLabel"]

X_test = df_test.drop(columns=["SepsisLabel", "patient_id"])
y_test = df_test["SepsisLabel"]

print("特征列名：")
print(X_train.columns.tolist())

print("\n训练集前两行数据：")
print(X_train.head(10))

# STEP 4: 构建 DMatrix（供 XGBoost 使用）
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


params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "scale_pos_weight": scale_pos_weight,
    "max_depth": 20,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "lambda": 1,
    "alpha": 0.05
}

# -----------------------------
# STEP 4: 模型训练
# -----------------------------
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=400,
    evals=[(dtest, "eval")],
    early_stopping_rounds=20,
    feval=lambda y_pred, dtrain: recall_f1_eval(y_pred, dtrain, recall_weight=0.5),
    maximize=True
)

# -----------------------------
# STEP 5: 模型预测
# -----------------------------
y_probs = bst.predict(dtest)

precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

# 计算 F1-score
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

acceptable_indices = np.where((recall >= 0.8) & (precision >= 0.2))[0]

if len(acceptable_indices) > 0:
    best_idx = acceptable_indices[np.argmax(f1_scores[acceptable_indices])]
    best_thresh = thresholds[best_idx]
    print(f"使用 Recall ≥ 0.8 & Precision ≥ 0.2 条件下的最佳阈值: {best_thresh:.2f}")
else:
    # 如果没有符合条件的点，退而求其次：选 F1-score 最大的
    best_idx = np.argmax(f1_scores)
    best_thresh = thresholds[best_idx]
    print(f"没有同时满足 Recall≥0.8 和 Precision≥0.2 的点，使用最大 F1-score 阈值: {best_thresh:.2f}")

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

# 保存报告
report_df = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T
report_df.to_csv(os.path.join(output_dir, "report.csv"))

# 保证 precision/recall/f1 与 threshold 对齐
thresholds_full = list(thresholds) + [1.0]
precision = list(precision)[:len(thresholds_full)]
recall = list(recall)[:len(thresholds_full)]
f1_scores = 2 * (np.array(precision) * np.array(recall)) / (np.array(precision) + np.array(recall) + 1e-8)
f1_scores = list(f1_scores)[:len(thresholds_full)]

# 保存指标表
metrics_table = pd.DataFrame({
    "threshold": thresholds_full,
    "precision": precision,
    "recall": recall,
    "f1": f1_scores
})
metrics_table.to_csv(os.path.join(output_dir, "threshold_metrics.csv"), index=False)
print("阈值指标表已保存：threshold_metrics.csv")

# 保存模型
joblib.dump(bst, os.path.join(output_dir, "xgboost_model.pkl"))

# PR曲线绘图
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

print(f"模型与报告保存至 {output_dir}")