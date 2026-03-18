import numpy as np
from typing import List
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
)

def compute_metrics(
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str]
) -> dict:
    """Compute metrics using sklearn metrics"""
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

    return {
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0
        ),
        "f1_per_class": {
            name: float(score)
            for name, score in zip(class_names, f1_per_class)
        },
    }

def compute_confusion_matrix(
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str]
) -> np.ndarray:
    """Compute confusion matrix using sklearn metrics"""
    return confusion_matrix(y_true, y_pred, class_names)

def full_classification_report(
        y_true: List[int],
        y_pred: List[int],
        class_names: List[str]
) -> dict:
    """Compute full classification report using sklearn metrics"""
    return classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)