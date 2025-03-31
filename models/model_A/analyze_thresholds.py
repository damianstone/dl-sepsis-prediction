import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, fbeta_score

def analyze_thresholds(
    y_true,
    y_probs,
    save_path="threshold_analysis.csv",
    min_recall=0.55,
    max_fp_rate=0.2
):
    thresholds = np.linspace(0.01, 0.99, 100)
    results = []

    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

        tn_fp = tn + fp
        tp_fn = tp + fn
        fpr = fp / (tn_fp + 1e-6)

        if recall >= min_recall and fpr <= max_fp_rate:
            results.append({
                "threshold": round(thresh, 4),
                "recall": round(recall, 4),
                "precision": round(precision, 4),
                "f2": round(f2, 4),
                "TN(%)": f"{tn / tn_fp:.2%}" if tn_fp else "N/A",
                "FP(%)": f"{fp / tn_fp:.2%}" if tn_fp else "N/A",
                "FN(%)": f"{fn / tp_fn:.2%}" if tp_fn else "N/A",
                "TP(%)": f"{tp / tp_fn:.2%}" if tp_fn else "N/A",
            })
        

    df = pd.DataFrame(results)
    df.to_csv(save_path, index=False)
    print(f"Threshold analysis saved to {save_path}")
