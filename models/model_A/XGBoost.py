import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from analyze_thresholds import analyze_thresholds
from plots import save_all_xgb_plots
from sklearn.metrics import (
    classification_report,
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
        root
        / "models"
        / "model_A"
        / "outputs"
        / "shap"
        / "shap_features_engineered.csv"
    )
    shap_df = pd.read_csv(shap_path)

    shap_df_sorted = shap_df.sort_values(by="mean_shap_positive", ascending=False)
    top_features = shap_df_sorted["feature"].tolist()[:n]

    sofa_features = ["SOFA_mean_global", "SOFA_max_global", "SOFA_last_global"]
    for sofa in sofa_features:
        if sofa not in top_features:
            top_features.append(sofa)

    print("\n Selected Features Used for XGBoost:")
    for f in top_features:
        print(f" - {f}")

    return top_features


def load_data(features):
    root = find_project_root()
    data_path = (
        root / "dataset" / "XGBoost" / "feature_engineering" / "train_balanced.parquet"
    )
    df = pd.read_parquet(data_path)
    if not isinstance(df.index, pd.MultiIndex):
        df.set_index(["patient_id"], inplace=True)

    X = df[features].fillna(-1)
    label_col = (
        "SepsisLabel_patient" if "SepsisLabel_patient" in df.columns else "SepsisLabel"
    )
    y = df[label_col]
    return X, y


def get_next_output_dir(base_dir):
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    existing = sorted(base_path.glob("train_*"))
    ids = [
        int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()
    ]
    next_id = max(ids) + 1 if ids else 1
    output_dir = base_path / f"train_{next_id}"
    output_dir.mkdir(exist_ok=False)
    return output_dir


def train_xgboost(X, y, params, n_splits=3):

    patient_ids = X.index.get_level_values("patient_id")
    patient_labels = (
        pd.DataFrame({"pid": patient_ids, "label": y}).groupby("pid")["label"].max()
    )
    root = find_project_root()
    output_base = get_next_output_dir(root / "models" / "model_A" / "train_outputs")
    fold_results = []
    fold_dirs = []
    pos_ids = patient_labels[patient_labels == 1].index.to_numpy()
    neg_ids = patient_labels[patient_labels == 0].index.to_numpy()

    np.random.seed(42)
    np.random.shuffle(pos_ids)
    np.random.shuffle(neg_ids)
    fold_pos = np.array_split(pos_ids, n_splits)
    fold_neg = np.array_split(neg_ids, n_splits)

    for fold in range(n_splits):
        print(f"\nFold {fold:}")

        test_pos = fold_pos[fold]
        test_neg = fold_neg[fold]
        train_pos = np.concatenate([fold_pos[i] for i in range(n_splits) if i != fold])
        train_neg_full = np.concatenate(
            [fold_neg[i] for i in range(n_splits) if i != fold]
        )
        np.random.shuffle(train_neg_full)
        train_neg = train_neg_full[: len(train_pos) * 4]
        np.random.shuffle(test_neg)
        test_neg = test_neg[: len(test_pos) * 4]

        train_ids = np.concatenate([train_pos, train_neg])
        test_ids = np.concatenate([test_pos, test_neg])

        train_mask = patient_ids.isin(train_ids)
        test_mask = patient_ids.isin(test_ids)

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        print(
            f"Fold {fold} - Train samples: {len(y_train)} (Pos: {y_train.sum()}, Neg: {len(y_train) - y_train.sum()})"
        )
        print(
            f"Fold {fold} - Test samples: {len(y_test)} (Pos: {y_test.sum()}, Neg: {len(y_test) - y_test.sum()})"
        )
        print(
            f"Fold {fold} - Positive rate in train: {y_train.mean():.3f}, test: {y_test.mean():.3f}"
        )

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        bst = xgb.train(
            params=params,
            dtrain=dtrain,
            num_boost_round=1000,
            evals=[(dtest, "eval")],
            early_stopping_rounds=50,
            verbose_eval=50,
        )

        y_probs = bst.predict(dtest)
        y_pred = (y_probs >= 0.5).astype(int)

        auc = roc_auc_score(y_test, y_probs)
        f1 = f1_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)

        print(
            f"Fold {fold} - AUROC: {auc:.4f}, F1: {f1:.4f}, F2: {f2:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}"
        )

        fold_results.append(
            {
                "fold": fold,
                "AUROC": auc,
                "F1": f1,
                "F2": f2,
                "Recall": recall,
                "Precision": precision,
            }
        )

        fold_dir = output_base / f"fold{fold}"
        fold_dir.mkdir()
        fold_dirs.append(fold_dir)

        # visualizations
        pd.DataFrame(classification_report(y_test, y_pred, output_dict=True)).T.to_csv(
            fold_dir / "report.csv"
        )

        model_path = fold_dir / "xgb_model.ubj"
        bst.save_model(str(model_path))

        save_all_xgb_plots(
            y_true=y_test,
            y_pred=y_pred,
            y_probs=y_probs,
            save_dir=fold_dir,
            booster=bst,
            feature_names=X.columns.tolist(),
        )

        threshold_csv_path = fold_dir / "threshold_analysis.csv"
        analyze_thresholds(y_true=y_test, y_probs=y_probs, save_path=threshold_csv_path)

    df_summary = pd.DataFrame(fold_results)
    best_idx = df_summary["F2"].idxmax()
    df_summary["is_best"] = [i == best_idx for i in range(len(df_summary))]

    avg_row = pd.DataFrame(
        [
            {
                "fold": "mean",
                "AUROC": df_summary["AUROC"].mean(),
                "F1": df_summary["F1"].mean(),
                "F2": df_summary["F2"].mean(),
                "Recall": df_summary["Recall"].mean(),
                "Precision": df_summary["Precision"].mean(),
                "is_best": "",
            }
        ]
    )

    summary_all = pd.concat([df_summary, avg_row], ignore_index=True)
    summary_all.to_csv(output_base / "summary.csv", index=False)

    best_model_path = output_base / "best_xgb_model.ubj "
    shutil.copy(fold_dirs[best_idx] / "xgb_model.ubj", best_model_path)

    shutil.copy(
        fold_dirs[best_idx] / "xgb_pr_curve.png", output_base / "best_pr_curve.png"
    )
    print(
        f"Best model from Fold {df_summary.loc[best_idx, 'fold']} saved to {best_model_path}"
    )


def main():
    features = load_top_features(n=20)
    # features = load_all_features()
    X, y = load_data(features)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 4,
        "lambda": 1,
        "alpha": 0.35,
    }

    train_xgboost(X, y, params=params)


if __name__ == "__main__":
    main()
