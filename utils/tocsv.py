import pandas as pd

imputed_file = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\dataset\imputed_combined_data.parquet"
raw_file = r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\dataset\raw_combined_data.parquet"

df_imputed = pd.read_parquet(imputed_file)
df_raw = pd.read_parquet(raw_file)

df_imputed.to_csv(
    r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\dataset\imputed_combined_data.csv",
    index=False,
)
df_raw.to_csv(
    r"C:\Users\Administrator\Desktop\ds\dl-sepsis-prediction\dataset\raw_combined_data.csv",
    index=False,
)

print("finish")
