import numpy as np


def patient_utility(y_pred, y_true):
    """
    y_pred, y_true: 1D numpy arrays of 0/1, length T
    Returns the un‐normalized utility U for this patient.
    """
    # find true onset (first index where y_true==1), or None
    idxs = np.where(y_true == 1)[0]
    tsep = int(idxs[0]) if len(idxs) > 0 else None

    U = 0.0
    for t in range(len(y_pred)):
        p = y_pred[t]
        if tsep is None:
            # non‐septic
            if p == 1:
                U -= 0.05
        else:
            dt = t - tsep
            if p == 1:
                # true‐alarm reward: triangle peak at dt=–6, zero at –12 and +3
                if -12 <= dt <= +3:
                    U += 1 - abs(dt + 6) / 6
                elif dt < -12:
                    U += 1 - (12 / 6)  # small reward (<0)
                else:
                    U += 1 - ((dt - 3) / 3)  # late - penalty
            else:
                # miss penalty in [–12,+3]
                if -12 <= dt <= +3:
                    U -= 2.0
                else:
                    U -= 0.1
    return U


def physionet_score(y_preds, y_trues, threshold=0.5):
    """
    y_preds, y_trues: lists of 1D arrays (one per patient)
    threshold: cutoff on model probability -> binary alarm
    Returns normalized utility: sum(U_i) / (n_septic × 1.0)
    """
    utilities = []
    n_septic = 0
    for yp, yt in zip(y_preds, y_trues):
        bp = (yp >= threshold).astype(int)
        utilities.append(patient_utility(bp, yt))
        if yt.max() == 1:
            n_septic += 1
    return np.sum(utilities) / (n_septic or 1)
