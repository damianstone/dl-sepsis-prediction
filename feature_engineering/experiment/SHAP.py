from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from scipy.special import expit
from sklearn.metrics import f1_score, fbeta_score, recall_score, roc_auc_score


def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError("Project root marker not found.")


def run_shap_on_data(parquet_file="no_sampling_train.parquet", output_tag="default"):
    root = find_project_root()
    input_path = root / "dataset" / "final_datasets" / parquet_file
    df = pd.read_parquet(input_path)

    if "patient_id" not in df.columns and "patient_id" in df.index.names:
        df = df.reset_index()

    label_col = (
        "SepsisLabel_patient" if "SepsisLabel_patient" in df.columns else "SepsisLabel"
    )

    exclude_prefixes = ["SepsisLabel_", "patient_id"]
    exclude_exact = [label_col]
    features = df.drop(
        columns=[
            col
            for col in df.columns
            if col in exclude_exact or any(col.startswith(p) for p in exclude_prefixes)
        ],
        errors="ignore",
    ).fillna(-1)

    labels = df[label_col].astype(int)

    model = xgb.XGBClassifier(n_estimators=400, max_depth=6, eval_metric="logloss")
    model.fit(features, labels)

    y_pred_prob = model.predict_proba(features)[:, 1]
    y_pred_label = (y_pred_prob >= 0.5).astype(int)
    print(f"\nModel Evaluation:")
    print(f"Recall: {recall_score(labels, y_pred_label):.4f}")
    print(f"F1 Score: {f1_score(labels, y_pred_label):.4f}")
    print(f"F2 Score: {fbeta_score(labels, y_pred_label, beta=2):.4f}")
    print(f"AUROC: {roc_auc_score(labels, y_pred_prob):.4f}")

    explainer = shap.Explainer(model)
    shap_values = explainer(features)

    output_dir = root / "models" / "model_A" / "outputs" / "shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_path = output_dir / f"shap_summary_{output_tag}.png"
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, features, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()

    shap_matrix = shap_values.values
    mean_abs_shap = np.abs(shap_matrix).mean(axis=0)
    feature_names = shap_values.feature_names
    shap_df_raw = pd.DataFrame(shap_matrix, columns=feature_names)
    shap_df_raw["predicted_label"] = y_pred_label

    grouped = shap_df_raw.groupby("predicted_label").mean().T
    grouped = grouped.rename(columns={0: "mean_shap_negative", 1: "mean_shap_positive"})
    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    shap_df = shap_df.set_index("feature").join(grouped).reset_index()
    shap_df = shap_df.sort_values(by="mean_abs_shap", ascending=False)

    table_path = output_dir / f"shap_features_{output_tag}.csv"
    shap_df.to_csv(table_path, index=False)

    print(f"SHAP summary plot saved to: {plot_path}")
    print(f"SHAP importance table saved to: {table_path}")

    top_positive_features = (
        shap_df.sort_values(by="mean_shap_positive", ascending=False)
        .head(20)["feature"]
        .tolist()
    )

    top_positive_indices = [features.columns.get_loc(f) for f in top_positive_features]
    shap_values_top_pos = shap.Explanation(
        values=shap_values.values[:, top_positive_indices],
        base_values=shap_values.base_values,
        data=features.iloc[:, top_positive_indices],
        feature_names=[features.columns[i] for i in top_positive_indices],
    )

    positive_plot_path = output_dir / f"top20_positive_shap.png"
    plt.figure(figsize=(12, 10))
    shap.summary_plot(
        shap_values_top_pos, features[top_positive_features], max_display=20, show=False
    )
    plt.tight_layout()
    plt.savefig(positive_plot_path, dpi=300)
    plt.close()
    print(f"Top 20 positive SHAP plot saved to: {positive_plot_path}")

    return shap_values, features, shap_df


def export_single_shap_to_csv(shap_values, features, sample_index, output_path):
    row = shap_values[sample_index]
    feature_names = shap_values.feature_names
    feature_values = features.iloc[sample_index]
    shap_scores = row.values

    df = pd.DataFrame(
        {
            "feature": feature_names,
            "value": feature_values.values,
            "shap_value": shap_scores,
        }
    )

    base_row = pd.DataFrame(
        [
            {
                "feature": "base_value (E[f(x)])",
                "value": "",
                "shap_value": row.base_values,
            }
        ]
    )

    logit_prediction = row.base_values + np.sum(row.values)
    probability = expit(logit_prediction)
    final_row = pd.DataFrame(
        [
            {
                "feature": "final_prediction_probability",
                "value": "",
                "shap_value": probability,
            }
        ]
    )

    df = pd.concat([df, base_row, final_row], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"SHAP CSV for sample {sample_index} saved to: {output_path}")


if __name__ == "__main__":
    shap_values, features, shap_df = run_shap_on_data()
    sample_index = 0
    output_path = Path("single_sample_shap.csv")
    export_single_shap_to_csv(shap_values, features, sample_index, output_path)
