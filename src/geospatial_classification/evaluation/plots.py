"""Plotting utilities for evaluation."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, roc_auc_score, roc_curve


def plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a ROC curve.

    Args:
        y_true: Ground-truth labels.
        y_prob: Predicted probability for the positive class.
        model_name: Label for the legend.
        ax: Optional matplotlib Axes; created if None.

    Returns:
        The Axes object.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True)
    return ax


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list,
    model_name: str = "Model",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """Plot a confusion matrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_labels, ax=ax, cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — {model_name}")
    return ax


def plot_training_curves(
    history: dict,
    save_path: Optional[str] = None,
) -> None:
    """Plot accuracy and loss curves from a training history dict.

    *history* should contain keys ``train_loss``, ``val_loss``,
    ``train_acc``, and ``val_acc``.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history["train_acc"], label="Train Acc")
    axes[0].plot(history["val_acc"], label="Val Acc")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history["train_loss"], label="Train Loss")
    axes[1].plot(history["val_loss"], label="Val Loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
