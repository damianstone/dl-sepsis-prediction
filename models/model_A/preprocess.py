import pandas as pd
import os

def add_time_diff_features(df, group_col="patient_id", time_col="ICULOS", deltas=[1, 3, 5]):
    df = df.sort_values(by=[group_col, time_col])
    df_with_diff = df.copy()

    dynamic_features = [
        'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 'BaseExcess', 'HCO3',
        'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride',
        'Creatinine', 'Bilirubin_direct', 'Glucose', 'Lactate', 'Magnesium', 'Phosphate',
        'Potassium', 'Bilirubin_total', 'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen',
        'Platelets', 'SOFA'
    ]

    for delta in deltas:
        for feature in dynamic_features:
            col_name = f"{feature}_diff_{delta}h"
            df_with_diff[col_name] = df_with_diff.groupby(group_col)[feature].diff(periods=delta)

    return df_with_diff

train_path = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\dataset\small_imputed_sofa_train.parquet"
test_path = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\dataset\small_imputed_sofa_test.parquet"

# 输出目标路径（模型 A 文件夹）
output_base = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\models\model_A"
os.makedirs(output_base, exist_ok=True)

train_out_path = os.path.join(output_base, "diff_sofa_train.parquet")
test_out_path = os.path.join(output_base, "diff_sofa_test.parquet")

# 加载数据
df_train = pd.read_parquet(train_path)
df_test = pd.read_parquet(test_path)

# 增强差分特征
df_train_enhanced = add_time_diff_features(df_train)
df_test_enhanced = add_time_diff_features(df_test)

# 保存增强版数据
df_train_enhanced.to_parquet(train_out_path)
df_test_enhanced.to_parquet(test_out_path)

print("增强完成，已保存到模型 A 目录：")
print(train_out_path)
print(test_out_path)