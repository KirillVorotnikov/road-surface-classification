import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def compute_metrics(
    y_true: list[int], y_pred: list[int], class_names: list[str]
) -> dict:
    """Compute metrics using sklearn metrics"""
    labels = range(len(class_names))
    f1_per_class = f1_score(
        y_true, y_pred, average=None, zero_division=0, labels=labels
    )

    # Suppress warning for single-class case
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return {
        "balanced_accuracy": balanced_acc,
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(
            y_true, y_pred, average="macro", zero_division=0, labels=labels
        ),
        "f1_weighted": f1_score(
            y_true, y_pred, average="weighted", zero_division=0, labels=labels
        ),
        "f1_per_class": {
            name: float(score) for name, score in zip(class_names, f1_per_class)
        },
    }


def compute_confusion_matrix(
    y_true: list[int], y_pred: list[int], class_names: list[str]
) -> np.ndarray:
    """Compute confusion matrix using sklearn metrics"""
    return confusion_matrix(y_true, y_pred, labels=range(len(class_names)))


def full_classification_report(
    y_true: list[int], y_pred: list[int], class_names: list[str]
) -> dict:
    """Compute full classification report using sklearn metrics"""
    return classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
