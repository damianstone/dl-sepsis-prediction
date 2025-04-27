from pathlib import Path

import pandas as pd
import xgboost as xgb
from plots import save_all_xgb_plots
from sklearn.metrics import (
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError("Project root marker not found.")


def load_top_features(n=20):
    root = find_project_root()
    shap_path = (
        root / "models" / "model_A" / "outputs" / "shap" / "shap_features_default.csv"
    )
    shap_df = pd.read_csv(shap_path)
    shap_df_sorted = shap_df.sort_values(by="mean_shap_positive", ascending=False)
    top_features = shap_df_sorted["feature"].tolist()[:n]
    print("\n Selected Features Used for XGBoost:")
    for f in top_features:
        print(f" - {f}")

    return top_features


def load_all_features():
    root = find_project_root()
    data_path = root / "dataset" / "final_datasets" / "no_sampling_train.parquet"
    df = pd.read_parquet(data_path)
    feature_cols = [
        col
        for col in df.columns
        if col not in ["patient_id", "SepsisLabel", "SepsisLabel_patient"]
    ]
    return feature_cols


def get_next_predict_dir(base_dir):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    existing = sorted(base_path.glob("predict_*"))
    ids = [
        int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()
    ]
    next_id = max(ids) + 1 if ids else 1
    predict_dir = base_path / f"predict_{next_id}"
    predict_dir.mkdir(exist_ok=False)
    return predict_dir


def evaluate_on_test_set(model_path, features, save_dir):
    root = find_project_root()
    test_path = root / "dataset" / "final_datasets" / "test.parquet"
    df = pd.read_parquet(test_path)
    if "patient_id" not in df.columns:
        df.reset_index(inplace=True)

    df = df[features + ["SepsisLabel"]]
    X_test = df[features]
    y_test = df["SepsisLabel"]

    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dtest = xgb.DMatrix(X_test)
    y_probs = booster.predict(dtest)
    y_pred = (y_probs >= 0.5).astype(int)

    print(f"\n[TEST EVALUATION for model: {model_path.name}]")
    print("AUROC:", roc_auc_score(y_test, y_probs))
    print("F1:", f1_score(y_test, y_pred))
    print("F2:", fbeta_score(y_test, y_pred, beta=2))
    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))

    save_all_xgb_plots(
        y_true=y_test,
        y_pred=y_pred,
        y_probs=y_probs,
        save_dir=save_dir,
        booster=booster,
        feature_names=features,
    )


if __name__ == "__main__":
    # features = load_all_features()
    features = load_top_features(n=40)
    root = find_project_root()

    specified_train_name = "train_3"
    train_dir = root / "models" / "model_A" / "train_outputs" / specified_train_name
    best_model_path = train_dir / "best_xgb_model.ubj"

    test_output_base = root / "models" / "model_A" / "test_outputs"
    predict_dir = get_next_predict_dir(test_output_base)

    evaluate_on_test_set(best_model_path, features, save_dir=predict_dir)
