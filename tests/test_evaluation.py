"""Unit tests for evaluation utilities."""

import numpy as np

from geospatial_classification.evaluation import model_metrics, plot_roc


class TestMetrics:
    def test_perfect_predictions(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])
        metrics = model_metrics(y_true, y_pred, y_prob, ["non-agri", "agri"])
        assert metrics["Accuracy"] == 1.0
        assert metrics["Precision"] == 1.0
        assert metrics["Recall"] == 1.0
        assert metrics["F1 Score"] == 1.0

    def test_all_wrong_predictions(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])
        y_prob = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
        metrics = model_metrics(y_true, y_pred, y_prob, ["non-agri", "agri"])
        assert metrics["Accuracy"] == 0.0

    def test_1d_probabilities(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])
        metrics = model_metrics(y_true, y_pred, y_prob, ["non-agri", "agri"])
        assert metrics["Accuracy"] == 1.0


class TestPlots:
    def test_plot_roc_runs(self) -> None:
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.3, 0.7])
        ax = plot_roc(y_true, y_prob, "Test Model")
        assert ax is not None
