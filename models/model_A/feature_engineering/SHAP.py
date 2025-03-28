import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.metrics import recall_score, f1_score, fbeta_score, roc_auc_score
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.special import expit

def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError("Project root marker not found.")

def run_shap_on_data(parquet_file="features_with_missing_interval.parquet"):
    root = find_project_root()
    input_path = root / "dataset" / "XGBoost" / "feature_engineering" / parquet_file
    df = pd.read_parquet(input_path)

    if not isinstance(df.index, pd.MultiIndex):
        df.set_index(["patient_id", "ICULOS"], inplace=True)

    df = df.groupby(level=0).filter(lambda x: len(x) >= 6)

    exclude = ["SepsisLabel", "HospAdmTime", "patient_id"]
    features = df.drop(columns=[col for col in exclude if col in df.columns], errors="ignore")
    features = features.fillna(-1)

    labels = df["SepsisLabel"].astype(int)
    
    # Experimental model
    model = xgb.XGBClassifier(n_estimators=400, max_depth=6, eval_metric="logloss")
    model.fit(features, labels)
    
    y_pred_prob = model.predict_proba(features)[:, 1]
    y_pred_label = (y_pred_prob >= 0.5).astype(int)

    recall = recall_score(labels, y_pred_label)
    f1 = f1_score(labels, y_pred_label)
    f2 = fbeta_score(labels, y_pred_label, beta=2)
    auroc = roc_auc_score(labels, y_pred_prob)

    print(f"\nModel Evaluation:")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"F2 Score: {f2:.4f}")
    print(f"AUROC: {auroc:.4f}")


    # SHAP
    explainer = shap.Explainer(model)
    shap_values = explainer(features)

    output_dir = root / "models" /"model_A" /  "outputs"/ "shap"
    output_dir.mkdir(parents=True, exist_ok=True)

    # summary plot
    plt.figure(figsize=(24, 24))
    shap.summary_plot(shap_values, features, max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary_plot.png", dpi=300)
    plt.close()
    print(f"SHAP plots saved to: {output_dir}")
    
    sample_index = 100
    csv_output = output_dir / f"shap_values_sample{sample_index}.csv"
    
    shap_matrix = np.abs(shap_values.values)
    mean_abs_shap = shap_matrix.mean(axis=0)
    feature_names = shap_values.feature_names

    shap_importance = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False)

    top20_df = shap_importance.head(20)
    top20_path = output_dir / "top20_features_by_shap.csv"
    top20_df.to_csv(top20_path, index=False)

    print(f"Top 20 features saved to: {top20_path}")

    return shap_values, features, sample_index, csv_output

def export_single_shap_to_csv(shap_values, features, sample_index, output_path):
    row = shap_values[sample_index]
    feature_names = shap_values.feature_names
    feature_values = features.iloc[sample_index]
    shap_scores = row.values

    df = pd.DataFrame({
        "feature": feature_names,
        "value": feature_values.values,
        "shap_value": shap_scores
    })

    base_row = pd.DataFrame([{
        "feature": "base_value (E[f(x)])",
        "value": "",
        "shap_value": row.base_values
    }])

    logit_prediction = row.base_values + np.sum(row.values)
    probability = expit(logit_prediction)
    final_row = pd.DataFrame([{
        "feature": "final_prediction_probability",
        "value": "",
        "shap_value": probability
    }])

    df = pd.concat([df, base_row, final_row], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"SHAP CSV for sample {sample_index} saved to: {output_path}")


if __name__ == "__main__":
    shap_values, features, sample_index, output_path = run_shap_on_data()
    export_single_shap_to_csv(shap_values, features, sample_index, output_path)