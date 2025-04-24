from pathlib import Path

import pandas as pd
import xgboost as xgb
from analyze_thresholds import analyze_thresholds
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


def evaluate_patient_level(model_path, test_path, save_dir):
    df = pd.read_parquet(test_path)
    df.set_index(["patient_id", "ICULOS"], inplace=True)
    feature_cols = df.columns.difference(["SepsisLabel"])
    X = df[feature_cols].fillna(-1)
    y_time = df["SepsisLabel"]

    dtest = xgb.DMatrix(X)
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    y_probs = booster.predict(dtest)
    y_pred = (y_probs >= 0.5).astype(int)

    df_pred = pd.DataFrame({
        "y_true": y_time,
        "y_prob": y_probs,
        "y_pred": y_pred
    }, index=df.index)


    df_patient = df_pred.groupby(level="patient_id").agg(
        {"y_true": "max", "y_prob": "max", "y_pred": "max"}
    )

    print("\n[EVALUATION ON PATIENT LEVEL]")
    print("AUROC:", roc_auc_score(df_patient["y_true"], df_patient["y_prob"]))
    print("F1:", f1_score(df_patient["y_true"], df_patient["y_pred"]))
    print("F2:", fbeta_score(df_patient["y_true"], df_patient["y_pred"], beta=2))
    print("Recall:", recall_score(df_patient["y_true"], df_patient["y_pred"]))
    print("Precision:", precision_score(df_patient["y_true"], df_patient["y_pred"]))

    save_all_xgb_plots(
        y_true=df_patient["y_true"],
        y_pred=df_patient["y_pred"],
        y_probs=df_patient["y_prob"],
        save_dir=save_dir,
        booster=booster,
        feature_names=feature_cols.tolist(),
    )

    threshold_csv_path = save_dir / "threshold_analysis.csv"
    analyze_thresholds(
        y_true=df_patient["y_true"], y_probs=df_patient["y_prob"], save_path=threshold_csv_path
    )


if __name__ == "__main__":
    root = find_project_root()
    test_path = root / "dataset" / "final_datasets" / "test.parquet"
    model_path = root / "models" / "model_A" / "train_outputs" / "train_9" / "best_xgb_model.ubj"
    save_dir = get_next_predict_dir(root / "models" / "model_A" / "test_outputs")
    evaluate_patient_level(model_path, test_path, save_dir)
