import pandas as pd


data_path = "C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/processed_data.csv"
output_path = "C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/filtered_data.csv"


data = pd.read_csv(data_path)


sepsis_patient_ids = data.loc[data['SepsisLabel'] == 1, 'patient_id'].unique()
sepsis_data = data[data['patient_id'].isin(sepsis_patient_ids)]


num_negative_patients = int(len(sepsis_patient_ids) * 0.1)
negative_patient_ids = data.loc[data['SepsisLabel'] == 0, 'patient_id'].unique()


selected_negative_patient_ids = pd.Series(negative_patient_ids).sample(n=num_negative_patients, random_state=0).values


negative_data = data[data['patient_id'].isin(selected_negative_patient_ids)]
filtered_data = pd.concat([sepsis_data, negative_data]).sort_values(by=['patient_id', 'ICULOS'])

filtered_data.to_csv(output_path, index=False)

print(f"原始数据行数: {len(data)}")
print(f"筛选后数据行数: {len(filtered_data)}")
print(f"新数据集已保存: {output_path}")
