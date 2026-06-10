"""Comprehensive model evaluation metrics."""

import logging
from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def model_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_labels: List[str],
) -> Dict[str, Any]:
    """Compute a comprehensive suite of classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        y_prob: Predicted probabilities; either ``(N,)`` for binary or
            ``(N, C)`` for multiclass.
        class_labels: Human-readable class names.

    Returns:
        Dictionary with scalar metrics, confusion matrix, and classification report.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 2:
        roc_score = roc_auc_score(y_true, y_prob[:, 1])
    else:
        roc_score = roc_auc_score(y_true, y_prob)

    metrics: Dict[str, Any] = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="weighted"),
        "Recall": recall_score(y_true, y_pred, average="weighted"),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
        "ROC-AUC": roc_score,
        "Log Loss": log_loss(y_true, y_prob),
        "Confusion Matrix": confusion_matrix(y_true, y_pred),
        "Classification Report": classification_report(y_true, y_pred, target_names=class_labels),
    }
    return metrics


def print_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_labels: List[str],
    model_name: str = "Model",
) -> None:
    """Log and print a formatted report of all metrics."""
    metrics = model_metrics(y_true, y_pred, y_prob, class_labels)

    sep = "=" * 60
    logger.info(sep)
    logger.info("Evaluation Report — %s", model_name)
    logger.info(sep)

    for key, val in metrics.items():
        if key in ("Confusion Matrix", "Classification Report"):
            continue
        logger.info("%s: %.4f", key, val)

    logger.info("Confusion Matrix:\n%s", metrics["Confusion Matrix"])
    logger.info("Classification Report:\n%s", metrics["Classification Report"])
