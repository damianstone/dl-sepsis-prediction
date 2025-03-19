import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
from datetime import datetime



base_output_dir = r'C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/result'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(base_output_dir, timestamp)
os.makedirs(output_dir, exist_ok=True)

X_train = pd.read_csv("C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/train/X_train.csv")
X_test = pd.read_csv("C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/test/X_test.csv")
y_train = pd.read_csv("C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/train/y_train.csv")
y_test = pd.read_csv("C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/test/y_test.csv")

# SOFA calculation based on Sepsis-3
def calculate_sofa(row):
    sofa = 0
    if row['FiO2'] > 0:
        pao2_fio2 = row['SaO2'] / row['FiO2']
        if pao2_fio2 < 100: sofa += 4
        elif pao2_fio2 < 200: sofa += 3
        elif pao2_fio2 < 300: sofa += 2
        elif pao2_fio2 < 400: sofa += 1

    if row['Platelets'] < 20: sofa += 4
    elif row['Platelets'] < 50: sofa += 3
    elif row['Platelets'] < 100: sofa += 2
    elif row['Platelets'] < 150: sofa += 1

    if row['Bilirubin_total'] >= 12: sofa += 4
    elif row['Bilirubin_total'] >= 6: sofa += 3
    elif row['Bilirubin_total'] >= 2: sofa += 2
    elif row['Bilirubin_total'] >= 1.2: sofa += 1

    if row['MAP'] < 70: sofa += 1

    if row['Creatinine'] >= 5: sofa += 4
    elif row['Creatinine'] >= 3.5: sofa += 3
    elif row['Creatinine'] >= 2: sofa += 2
    elif row['Creatinine'] >= 1.2: sofa += 1

    return sofa

X_train['SOFA'] = X_train.apply(calculate_sofa, axis=1)
X_test['SOFA'] = X_test.apply(calculate_sofa, axis=1)


scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.1,
    'max_depth': 20,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 0,
    'eval_metric': 'logloss',
    'scale_pos_weight': scale_pos_weight
}


model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'), index=True)

plt.figure(figsize=(10, 7))
xgb.plot_importance(model, max_num_features=15)
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance.png", dpi=300)
plt.close()



print(f"finish")
