import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score, recall_score, precision_score
from plots import save_all_xgb_plots

def find_project_root(marker=".gitignore"):
    current = Path.cwd()
    for parent in [current] + list(current.parents):
        if (parent / marker).exists():
            return parent.resolve()
    raise FileNotFoundError("Project root marker not found.")

def load_top_features(n=30):
    root = find_project_root()
    top_path = root / "models" / "model_A" / "outputs" / "shap" / "top_features_by_shap.csv"
    top_features = pd.read_csv(top_path)["feature"].tolist()
    

    if "SOFA" not in top_features:
        top_features.append("SOFA")

    return top_features[:n] + (["SOFA"] if "SOFA" not in top_features[:n] else [])

def get_next_predict_dir(base_dir):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    existing = sorted(base_path.glob("predict_*"))
    ids = [int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()]
    next_id = max(ids) + 1 if ids else 1
    predict_dir = base_path / f"predict_{next_id}"
    predict_dir.mkdir(exist_ok=False)
    return predict_dir

def evaluate_on_test_set(model_path, features, save_dir):
    root = find_project_root()
    test_path = root / "dataset" / "XGBoost" / "feature_engineering" / "test_balanced.parquet"
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
    y_pred = (y_probs >= 0.3).astype(int)

    print(f"\n[TEST EVALUATION for model: {model_path.name}]")
    print("AUROC:", roc_auc_score(y_test, y_probs))
    print("F1:", f1_score(y_test, y_pred))
    print("F2:", fbeta_score(y_test, y_pred, beta=2))
    print("Recall:", recall_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))

    save_all_xgb_plots(y_true=y_test, y_pred=y_pred, y_probs=y_probs, save_dir=save_dir, booster=booster, feature_names=features)

if __name__ == "__main__":
    features = load_top_features(n=50)
    root = find_project_root()


    specified_train_name = "train_64"
    train_dir = root / "models" / "model_A" / "train_outputs" / specified_train_name
    best_model_path = train_dir / "best_xgb_model.ubj"


    test_output_base = root / "models" / "model_A" / "test_outputs"
    predict_dir = get_next_predict_dir(test_output_base)

    evaluate_on_test_set(best_model_path, features, save_dir=predict_dir)
