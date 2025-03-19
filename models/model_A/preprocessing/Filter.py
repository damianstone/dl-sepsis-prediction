import pandas as pd
from sklearn.model_selection import train_test_split

data_path = "C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/processed_data.csv"

data = pd.read_csv(data_path)

sepsis_patient_ids = data.loc[data['SepsisLabel'] == 1, 'patient_id'].unique()
sepsis_data = data[data['patient_id'].isin(sepsis_patient_ids)]

num_negative_patients = int(len(sepsis_patient_ids) * 0.1)
negative_patient_ids = data.loc[data['SepsisLabel'] == 0, 'patient_id'].unique()
selected_negative_patient_ids = pd.Series(negative_patient_ids).sample(n=num_negative_patients, random_state=0).values
negative_data = data[data['patient_id'].isin(selected_negative_patient_ids)]


filtered_data = pd.concat([sepsis_data, negative_data]).sort_values(by=['patient_id', 'ICULOS'])


X = filtered_data.drop(columns=['SepsisLabel'])  
y = filtered_data['SepsisLabel']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.to_csv("C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/train/X_train.csv", index=False)
X_test.to_csv("C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/test/X_test.csv", index=False)
y_train.to_csv("C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/train/y_train.csv", index=False)
y_test.to_csv("C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/test/y_test.csv", index=False)


print("done")
