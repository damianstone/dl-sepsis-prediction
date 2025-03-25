import pandas as pd

# 读取 CSV 文件
file_path = "C:/Users/Administrator/Desktop/ds/dl-sepsis-prediction/dataset/training_setA/p000001.csv"
df = pd.read_csv(file_path)

# 显示前几行数据
print(df.head())

# 查看数据的基本信息
print(df.info())
