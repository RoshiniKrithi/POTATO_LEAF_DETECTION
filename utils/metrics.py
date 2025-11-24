from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)


def compute_classification_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {
        "accuracy": acc,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }


def per_class_report(y_true: List[int], y_pred: List[int]) -> Dict[int, Dict[str, float]]:
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    report: Dict[int, Dict[str, float]] = {}
    for i in range(len(precision)):
        report[i] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
    return report


def confusion_matrix_array(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    return confusion_matrix(y_true, y_pred)


def try_multiclass_roc_auc(y_true: List[int], y_scores: np.ndarray) -> float | None:
    try:
        return float(roc_auc_score(y_true, y_scores, multi_class="ovr"))
    except Exception:
        return None


