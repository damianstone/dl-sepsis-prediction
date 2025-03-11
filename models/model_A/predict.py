import pandas as pd
import joblib
import os
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime


base_dir = "C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A"
model_path ="C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/result/20250311_185309/xgboost_model.pkl"


if not os.path.exists(model_path):
    raise FileNotFoundError(f"未找到模型: {model_path}")


model = joblib.load(model_path)

print(f"成功加载模型: {model_path}")

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

predict_output_dir = os.path.join(base_dir, "result/predict", timestamp)

os.makedirs(predict_output_dir, exist_ok=True)

data_path = "C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/processed_data.csv"

df = pd.read_csv(data_path)

def calculate_sofa(row):
    sofa = 0
    if row.get('FiO2', 0) > 0:
        pao2_fio2 = row.get('SaO2', 1) / row.get('FiO2', 1)
        if pao2_fio2 < 100: sofa += 4
        elif pao2_fio2 < 200: sofa += 3
        elif pao2_fio2 < 300: sofa += 2
        elif pao2_fio2 < 400: sofa += 1

    if row.get('Platelets', 9999) < 20: sofa += 4
    elif row.get('Platelets', 9999) < 50: sofa += 3
    elif row.get('Platelets', 9999) < 100: sofa += 2
    elif row.get('Platelets', 9999) < 150: sofa += 1

    if row.get('Bilirubin_total', 0) >= 12: sofa += 4
    elif row.get('Bilirubin_total', 0) >= 6: sofa += 3
    elif row.get('Bilirubin_total', 0) >= 2: sofa += 2
    elif row.get('Bilirubin_total', 0) >= 1.2: sofa += 1

    if row.get('MAP', 999) < 70: sofa += 1

    if row.get('Creatinine', 0) >= 5: sofa += 4
    elif row.get('Creatinine', 0) >= 3.5: sofa += 3
    elif row.get('Creatinine', 0) >= 2: sofa += 2
    elif row.get('Creatinine', 0) >= 1.2: sofa += 1

    return sofa

df['SOFA'] = df.apply(calculate_sofa, axis=1)

X_new = df.drop(columns=['patient_id', 'ICULOS', 'SepsisLabel'], errors='ignore')

predictions = model.predict(X_new)

predictions_df = pd.DataFrame({'Sepsis_Prediction': predictions})

output_path = os.path.join(predict_output_dir, "predictions.csv")

predictions_df.to_csv(output_path, index=False)

print(f"预测结果已保存: {output_path}")


if 'SepsisLabel' in df.columns:
    y_true = df['SepsisLabel']
    y_pred = predictions

    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)

    # ✅ 只提取关键指标
    metrics_df = pd.DataFrame(report).transpose()[['precision', 'recall', 'f1-score']]
    metrics_df.loc['accuracy'] = [accuracy, accuracy, accuracy]  # 追加 Accuracy

    # ✅ 保存预测指标
    metrics_path = os.path.join(predict_output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=True)

    print(f"预测完成，指标已保存: {metrics_path}")
    print(metrics_df)
