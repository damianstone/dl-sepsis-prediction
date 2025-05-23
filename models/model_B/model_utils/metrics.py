import json
import os

import sklearn.metrics as m


def save_metrics(root, xperiment_name, y_test, y_probs, y_pred):
    save_path = f"{root}/models/model_B/results/{xperiment_name}"
    os.makedirs(save_path, exist_ok=True)

    accuracy = m.accuracy_score(y_test, y_pred)
    balanced_accuracy = m.balanced_accuracy_score(y_test, y_pred)
    precision = m.precision_score(y_test, y_pred, zero_division=0)
    recall = m.recall_score(y_test, y_pred, zero_division=0)
    auc_score = m.roc_auc_score(y_test, y_probs)
    f1 = m.f1_score(y_test, y_pred, zero_division=0)
    f_beta = m.fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    report = m.classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    metrics = {
        "f1_score": round(float(f1), 3),
        "f2_score": round(float(f_beta), 3),
        "balanced_accuracy": round(float(accuracy), 3),
        "accuracy": round(float(balanced_accuracy), 3),
        "AUC": round(float(auc_score), 3),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "classification_report": report,
    }

    with open(os.path.join(save_path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {save_path}/metrics.json")
