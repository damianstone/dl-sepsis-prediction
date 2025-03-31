import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import (classification_report, roc_auc_score,f1_score, recall_score, precision_score)
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from plots import save_all_xgb_plots
from analyze_thresholds import analyze_thresholds



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


def load_all_features():
    root = find_project_root()
    path = root / "dataset" / "XGBoost" / "feature_engineering" / "train_balanced.parquet"
    df = pd.read_parquet(path)

    if not isinstance(df.index, pd.MultiIndex):
        df.set_index(["patient_id", "ICULOS"], inplace=True)

    exclude = ["SepsisLabel", "patient_id","HospAdmTime", "dataset", "Unit1", "Unit2"]
    all_features = [col for col in df.columns if col not in exclude]
    return all_features

def load_data(features):
    root = find_project_root()
    data_path = root / "dataset" / "XGBoost" / "feature_engineering" / "train_balanced.parquet"
    df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.MultiIndex):
        df.set_index(["patient_id", "ICULOS"], inplace=True)

    X = df[features].fillna(-1)
    y = df["SepsisLabel"]
    return X, y


def get_next_output_dir(base_dir):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    existing = sorted(base_path.glob("train_*"))
    ids = [int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()]
    next_id = max(ids) + 1 if ids else 1
    output_dir = base_path / f"train_{next_id}"
    output_dir.mkdir(exist_ok=False)
    return output_dir

def train_xgboost(X, y, params, n_splits=3):
    from sklearn.metrics import fbeta_score
    import shutil

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    root = find_project_root()
    output_base = get_next_output_dir(root / "models" / "model_A" / "train_outputs")

    fold_results = []
    fold_dirs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f"\nFold {fold}")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        bst = xgb.train(params=params,dtrain=dtrain,num_boost_round=1000,evals=[(dtest, "eval")],early_stopping_rounds=50,verbose_eval=50)

        y_probs = bst.predict(dtest)
        y_pred = (y_probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_probs)
        f1 = f1_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print(f"Fold {fold} - AUROC: {auc:.4f}, F1: {f1:.4f}, F2: {f2:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

        fold_results.append({
            "fold": fold,
            "AUROC": auc,
            "F1": f1,
            "F2": f2,
            "Recall": recall,
            "Precision": precision
        })

        fold_dir = output_base / f"fold{fold}"
        fold_dir.mkdir()
        fold_dirs.append(fold_dir)

        #visualizations
        pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.to_csv(fold_dir / "report.csv")

        model_path = fold_dir / "xgb_model.ubj "
        bst.save_model(str(model_path))
      
        save_all_xgb_plots(y_true=y_test,y_pred=y_pred,y_probs=y_probs,save_dir=fold_dir,booster=bst,feature_names=X.columns.tolist())
        
        threshold_csv_path = fold_dir / "threshold_analysis.csv"
        analyze_thresholds(y_true=y_test, y_probs=y_probs, save_path=threshold_csv_path)


    df_summary = pd.DataFrame(fold_results)
    best_idx = df_summary["F2"].idxmax()
    df_summary["is_best"] = [i == best_idx for i in range(len(df_summary))]

    avg_row = pd.DataFrame([{
        "fold": "mean",
        "AUROC": df_summary["AUROC"].mean(),
        "F1": df_summary["F1"].mean(),
        "F2": df_summary["F2"].mean(),
        "Recall": df_summary["Recall"].mean(),
        "Precision": df_summary["Precision"].mean(),
        "is_best": ""}])
    
    summary_all = pd.concat([df_summary, avg_row], ignore_index=True)
    summary_all.to_csv(output_base / "summary.csv", index=False)

    best_model_path = output_base / "best_xgb_model.ubj "
    shutil.copy(fold_dirs[best_idx] / "xgb_model.ubj", best_model_path)

    shutil.copy(fold_dirs[best_idx] / "xgb_pr_curve.png", output_base / "best_pr_curve.png")
    print(f"Best model from Fold {df_summary.loc[best_idx, 'fold']} saved to {best_model_path}")


def main():
    features = load_top_features(n=50)
    #features = load_all_features()
    X, y = load_data(features)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 4,
        "min_child_weight": 10,
        "lambda": 5,
        "alpha": 1
    }

    train_xgboost(X, y, params=params)

if __name__ == "__main__":
    main()
