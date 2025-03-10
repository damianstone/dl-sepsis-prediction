import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import joblib

# Output directory with timestamp
base_output_dir = r'C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/result'
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = os.path.join(base_output_dir, timestamp)
os.makedirs(output_dir, exist_ok=True)

# Load data
data_path = r'C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/processed_data.csv'
df = pd.read_csv(data_path)

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

df['SOFA'] = df.apply(calculate_sofa, axis=1)
df_latest = df.sort_values(['patient_id', 'ICULOS']).groupby('patient_id').last().reset_index()

X = df_latest.drop(columns=['SepsisLabel', 'patient_id', 'ICULOS'])
y = df_latest['SepsisLabel']
X['Gender'] = X['Gender'].map({'Female': 0, 'Male': 1})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

params = {
    'objective': 'binary:logistic',
    'learning_rate': 0.1,
    'max_depth': 4,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss',
    'scale_pos_weight': scale_pos_weight
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save results in unique directory
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = f"{base_output_dir}/{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Save classification report to CSV
report_df = pd.DataFrame(report).transpose()
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(output_dir, 'classification_report.csv'), index=True)

# Feature importance
plt.figure(figsize=(10, 7))
xgb.plot_importance(model, max_num_features=15)
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.savefig(f"{output_dir}/feature_importance.png", dpi=300)
plt.close()

# Tree visualization
plt.figure(figsize=(30, 15))
xgb.plot_tree(model, num_trees=0, rankdir='LR')
plt.title('Visualization of Tree 0')
plt.tight_layout()
plt.savefig(f"{output_dir}/tree_0.png", dpi=300)
plt.close()

# Node distribution
leaf_counts = [model.get_booster().trees_to_dataframe().query(f'Tree=={i}').shape[0] for i in range(params['n_estimators'])]
plt.figure(figsize=(10, 6))
sns.histplot(leaf_counts, bins=20)
plt.xlabel('Number of Nodes per Tree')
plt.ylabel('Frequency')
plt.title('Distribution of Nodes Across Trees')
plt.tight_layout()
plt.savefig(f"{output_dir}/nodes_distribution.png", dpi=300)
plt.close()

# Decision rules CSV
tree_df = model.get_booster().trees_to_dataframe()
tree_df.query('Tree==0').to_csv(f"{output_dir}/tree_0_decision_rules.csv", index=False)

# Save classification report to CSV
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f"{output_dir}/classification_report.csv", index=True)
# Save trained model to the current output directory
model_path = os.path.join(output_dir, 'xgboost_model.pkl')
joblib.dump(model, model_path)