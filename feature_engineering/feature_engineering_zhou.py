from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(marker=".gitignore"):
    """
    walk up from the current working directory until a directory containing the
    specified marker (e.g., .gitignore) is found.
    """
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError(
        f"Project root marker '{marker}' not found starting from {current}"
    )


def score_by_value(value, thresholds):
    for threshold, score in thresholds:
        if value >= threshold:
            return score
    return 0


def calculate_sofa(df, row):
    sofa = 0
    if row.get("MAP", 100) < 70:
        sofa += 1
    sofa += score_by_value(
        row.get("Creatinine", 0), [(5, 4), (3.5, 3), (2, 2), (1.2, 1)]
    )
    sofa += score_by_value(
        row.get("Platelets", float("inf")), [(20, 4), (50, 3), (100, 2), (150, 1)]
    )
    sofa += score_by_value(
        row.get("Bilirubin_total", 0), [(12, 4), (6, 3), (2, 2), (1.2, 1)]
    )
    if row.get("FiO2", 0) > 0 and pd.notna(row.get("SaO2", None)):
        pao2_fio2 = row["SaO2"] / row["FiO2"]
        sofa += score_by_value(pao2_fio2, [(100, 4), (200, 3), (300, 2), (400, 1)])
    return sofa


def calculate_news(row):
    news = 0
    news += score_by_value(
        row.get("HR", 0), [(131, 3), (130, 2), (110, 1), (90, 0), (50, 1), (40, 3)]
    )
    news += score_by_value(
        row.get("Resp", 0), [(24, 3), (21, 2), (11, 0), (9, 1), (8, 3)]
    )
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
    return stats


def aggregate_scores(df, suffix="global"):
    sofa_scores = df.apply(calculate_sofa, axis=1)
    news_scores = df.apply(calculate_news, axis=1)
    qsofa_scores = df.apply(calculate_qsofa, axis=1)

    return {
        f"SOFA_mean_{suffix}": sofa_scores.mean(),
        f"SOFA_max_{suffix}": sofa_scores.max(),
        f"SOFA_last_{suffix}": (
            sofa_scores.iloc[-1] if not sofa_scores.empty else np.nan
        ),
        f"NEWS_mean_{suffix}": news_scores.mean(),
        f"NEWS_max_{suffix}": news_scores.max(),
        f"NEWS_last_{suffix}": (
            news_scores.iloc[-1] if not news_scores.empty else np.nan
        ),
        f"qSOFA_mean_{suffix}": qsofa_scores.mean(),
        f"qSOFA_max_{suffix}": qsofa_scores.max(),
        f"qSOFA_last_{suffix}": (
            qsofa_scores.iloc[-1] if not qsofa_scores.empty else np.nan
        ),
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
        summary[f"{col}_missing_interval_mean_global"] = (
            np.mean(valid_intervals) if valid_intervals else -1
        )
    return summary


def preprocess_data(imputed_file, raw_file, output_file):

    df_imputed = pd.read_parquet(imputed_file)
    df_raw = pd.read_parquet(raw_file)

    selected_cols = ["HR", "O2Sat", "SBP", "MAP", "Resp"]

    all_rows = []

    for pid, group in df_imputed.groupby("patient_id"):
        group = group.sort_values("ICULOS").copy()

        group["SOFA"] = group.apply(lambda row: calculate_sofa(group, row), axis=1)
        group["NEWS"] = group.apply(calculate_news, axis=1)
        group["qSOFA"] = group.apply(calculate_qsofa, axis=1)

        scores = aggregate_scores(group, suffix="global")
        for k, v in scores.items():
            group[k] = v

        for i in range(len(group)):
            if i >= 6:
                window = group.iloc[i - 6 : i]
                stats = aggregate_window_features(window, selected_cols, suffix="6h")
            else:
                stats = {
                    f"{col}_{stat}_6h": 0
                    for col in selected_cols
                    for stat in ["mean", "std", "min", "max", "last"]
                }
            for k, v in stats.items():
                group.iloc[
                    i,
                    (
                        group.columns.get_loc(k)
                        if k in group.columns
                        else len(group.columns)
                    ),
                ] = v

        all_rows.append(group)

    df_final = pd.concat(all_rows)

    missing_rows = []
    for pid, group in df_raw.groupby("patient_id"):
        summary = compute_missingness_summary(group, selected_cols)
        summary["patient_id"] = pid
        missing_rows.append(summary)
    df_missing = pd.DataFrame(missing_rows)

    df_final = pd.merge(df_final, df_missing, on="patient_id", how="left")

    df_final.drop(
        columns=["Unit1", "Unit2", "cluster_id", "dataset"],
        errors="ignore",
        inplace=True,
    )

    print("Any NaN:", df_final.isna().sum().sum() > 0)
    print("New columns:", list(df_final.columns))
    print("Total number of features:", df_final.shape[1])

    df_final.to_parquet(output_file, index=False)
    print(f"Full time-series feature set saved to: {output_file}")

    print(f"Preprocessed data saved to {output_file}")


if __name__ == "__main__":
    root = find_project_root()
    RAW_DATASET = f"{root}/dataset/raw_combined_data.parquet"
    INPUT_DATASET = f"{root}/dataset/Fully_imputed_dataset.parquet"
    OUTPUT_DATASET = f"{root}/dataset/V2_preprocessed.parquet"
    preprocess_data(INPUT_DATASET, RAW_DATASET, OUTPUT_DATASET)
