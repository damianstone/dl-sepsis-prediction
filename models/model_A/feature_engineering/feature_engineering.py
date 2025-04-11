import pandas as pd
import numpy as np
from pathlib import Path

#medical

def score_by_value(value, thresholds):
    for threshold, score in thresholds:
        if value >= threshold:
            return score
    return 0

def calculate_sofa(row):
    sofa = 0
    if row.get("MAP", 100) < 70:
        sofa += 1
    sofa += score_by_value(row.get("Creatinine", 0), [(5, 4), (3.5, 3), (2, 2), (1.2, 1)])
    sofa += score_by_value(row.get("Platelets", float("inf")), [(20, 4), (50, 3), (100, 2), (150, 1)])
    sofa += score_by_value(row.get("Bilirubin_total", 0), [(12, 4), (6, 3), (2, 2), (1.2, 1)])
    if row.get("FiO2", 0) > 0 and pd.notna(row.get("SaO2", None)):
        pao2_fio2 = row["SaO2"] / row["FiO2"]
        sofa += score_by_value(pao2_fio2, [(100, 4), (200, 3), (300, 2), (400, 1)])
    return sofa

def calculate_news(row):
    news = 0
    news += score_by_value(row.get("HR", 0), [(131, 3), (130, 2), (110, 1), (90, 0), (50, 1), (40, 3)])
    news += score_by_value(row.get("Resp", 0), [(24, 3), (21, 2), (11, 0), (9, 1), (8, 3)])
    news += score_by_value(row.get("Temp", 0), [(39.1, 2), (38, 1), (36, 1), (35, 3)])
    sbp = row.get("SBP", row.get("MAP", 100))
    news += score_by_value(sbp, [(90, 3), (100, 2), (110, 1)])
    news += score_by_value(row.get("O2Sat", 0), [(85, 3), (91, 2), (93, 1)])
    if row.get("FiO2", 0) > 0.21:
        news += 2
    return news

def calculate_qsofa(row):
    qsofa = 0
    if row.get("SBP", 120) <= 100:
        qsofa += 1
    if row.get("Resp", 0) >= 22:
        qsofa += 1
    return qsofa


def aggregate_window_features(df, cols, suffix):
    stats = {}
    for col in cols:
        series = df[col].dropna()
        stats[f"{col}_mean_{suffix}"] = series.mean()
        stats[f"{col}_std_{suffix}"] = series.std()
        stats[f"{col}_min_{suffix}"] = series.min()
        stats[f"{col}_max_{suffix}"] = series.max()
        stats[f"{col}_last_{suffix}"] = series.iloc[-1] if not series.empty else np.nan

        is_missing = df[col].isna()
        stats[f"{col}_missing_count_{suffix}"] = is_missing.sum()
        stats[f"{col}_missing_rate_{suffix}"] = is_missing.mean()
    return stats

def aggregate_scores(df, suffix="global"):
    sofa_scores = df.apply(calculate_sofa, axis=1)
    news_scores = df.apply(calculate_news, axis=1)
    qsofa_scores = df.apply(calculate_qsofa, axis=1)

    return {
        f"SOFA_mean_{suffix}": sofa_scores.mean(),
        f"SOFA_max_{suffix}": sofa_scores.max(),
        f"SOFA_last_{suffix}": sofa_scores.iloc[-1] if not sofa_scores.empty else np.nan,
        f"NEWS_mean_{suffix}": news_scores.mean(),
        f"NEWS_max_{suffix}": news_scores.max(),
        f"NEWS_last_{suffix}": news_scores.iloc[-1] if not news_scores.empty else np.nan,
        f"qSOFA_mean_{suffix}": qsofa_scores.mean(),
        f"qSOFA_max_{suffix}": qsofa_scores.max(),
        f"qSOFA_last_{suffix}": qsofa_scores.iloc[-1] if not qsofa_scores.empty else np.nan
    }

def compute_missingness_summary(df, cols):
    summary = {}
    for col in cols:
        series = df[col]
        is_missing = series.isna()
        summary[f"{col}_missing_count_global"] = is_missing.sum()

        last_seen = None
        intervals = []
        for i, val in enumerate(series):
            if pd.notna(val):
                last_seen = i
            elif last_seen is not None:
                intervals.append(i - last_seen)
            else:
                intervals.append(-1)
        valid_intervals = [v for v in intervals if v >= 0]
        summary[f"{col}_missing_interval_mean_global"] = np.mean(valid_intervals) if valid_intervals else -1
    return summary



def generate_multiwindow_features(df, feature_cols, window_size=6):
    patient_rows = []

    for pid, group in df.groupby("patient_id"):
        row = {"patient_id": pid}
        group = group.sort_values("ICULOS").reset_index(drop=True)
        max_time = group["ICULOS"].max()

        row.update(aggregate_window_features(group, feature_cols, suffix="global"))
        row.update(compute_missingness_summary(group, feature_cols))
        row.update(aggregate_scores(group, suffix="global"))

        window_idx = 1
        for start in range(0, int(max_time) + 1, window_size):
            window_df = group[(group["ICULOS"] >= start) & (group["ICULOS"] < start + window_size)]
            if window_df.shape[0] < 2:
                continue
            suffix = f"6h{window_idx}"
            row.update(aggregate_window_features(window_df, feature_cols, suffix=suffix))
            window_idx += 1

        # Demographics
        last_row = group.iloc[-1]
        row["Age"] = last_row.get("Age", np.nan)
        row["Gender"] = last_row.get("Gender", np.nan)
        row["ICULOS_last"] = last_row.get("ICULOS", np.nan)
        row["SepsisLabel_patient"] = group["SepsisLabel"].max()

        patient_rows.append(row)

    return pd.DataFrame(patient_rows)

def run_multiwindow_feature_engineering(compress=False):
    input_path = Path("dataset/XGBoost/feature_engineering/balanced_dataset_filtered.parquet")
    output_path = Path("dataset/XGBoost/feature_engineering/after_feature_engineering.parquet")
    preview_path = Path("dataset/XGBoost/feature_engineering/patient_preview.csv")
    shap_path = Path("models/model_A/outputs/shap/shap_features_filtered.csv")

    df = pd.read_parquet(input_path)
    if "ICULOS" not in df.columns and "ICULOS" in df.index.names:
        df = df.reset_index()

    shap_df = pd.read_csv(shap_path)
    top_shap_features = (
        shap_df.sort_values(by="mean_shap_positive", ascending=False)
        .head(20)["feature"]
        .tolist()
    )

    print(f"Top 20 physiological features selected from SHAP:\n{top_shap_features}")

    if compress:
        df_feat = generate_multiwindow_features(df, top_shap_features, window_size=6)
        df_feat.to_parquet(output_path, index=False)
        df_feat.head(10).to_csv(preview_path, index=False)
        print(f"\nSaved patient-level multi-window features to: {output_path}")
        print(f"Preview saved to: {preview_path}")
    else:
        preview_path = Path("dataset/patient_preview_timeseries.csv")
        df = df.sort_values(['patient_id', 'ICULOS']).copy()
        output_path = Path("dataset/XGBoost/feature_engineering/after_feature_engineering_timeseries.parquet")
        df['SOFA'] = df.apply(calculate_sofa, axis=1)
        df['NEWS'] = df.apply(calculate_news, axis=1)
        df['qSOFA'] = df.apply(calculate_qsofa, axis=1)
        patient_scores = []
        for pid, group in df.groupby("patient_id"):
            group = group.sort_values("ICULOS").copy()
            scores = aggregate_scores(group, suffix="global")
            for k, v in scores.items():
                group[k] = v
            group["SepsisLabel_patient"] = group["SepsisLabel"].max()
            patient_scores.append(group)

        df = pd.concat(patient_scores)
        time_window_sizes = [3, 6, 12]
        feature_dict = {}
        for col in top_shap_features:
            if col in df.columns:
                for window in time_window_sizes:
                    feature_dict[f'{col}_MA_{window}h'] = df.groupby('patient_id')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean())
                    feature_dict[f'{col}_SD_{window}h'] = df.groupby('patient_id')[col].transform(
                    lambda x: x.rolling(window, min_periods=1).std())
                feature_dict[f'{col}_Delta'] = df.groupby('patient_id')[col].diff()

        df = pd.concat([df, pd.DataFrame(feature_dict, index=df.index)], axis=1)

        df.to_parquet(output_path, index=False)
        df.head(40).to_csv(preview_path, index=False)
        print(f"Saved time-series feature-enhanced data to: {output_path}")
        print(f"Preview saved to: {preview_path}")



if __name__ == "__main__":
    run_multiwindow_feature_engineering()