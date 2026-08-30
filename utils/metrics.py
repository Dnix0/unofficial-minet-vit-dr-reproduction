import numpy as np
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "qwk": qwk,
        "confusion_matrix": cm
    }

    return metrics