import pandas as pd


input_path = "C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/dataset/imputed_combined_data.csv"
output_path = "C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/models/model_A/processed_data.csv"

data = pd.read_csv(input_path)
print(f"colums before ：{len(data.columns)}")

data_cleaned = data.drop(columns=["Unit1", "Unit2", "cluster_id","dataset"], errors='ignore')


data_cleaned.to_csv(output_path, index=False)
print(f"colums after：{len(data_cleaned.columns)}")

print(f"finish, new file path：{output_path}")
