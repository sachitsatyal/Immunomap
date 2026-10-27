"""Metrics for confidence quality and unknown-cell detection."""

from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from immunomap.models.calibration import UNKNOWN, calibrated_proba


def expected_calibration_error(confidence: Any, correct: Any, n_bins: int = 15) -> float:
    """Average gap between confidence and accuracy, weighted by cells per confidence bin.

    0 means perfectly calibrated: among cells at 90% confidence, 90% are right.
    """
    confidence = np.asarray(confidence, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    bins = np.minimum((confidence * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(ece)


def accuracy_at_coverage(confidence: Any, correct: Any, coverage: float) -> float:
    """Accuracy on the most confident `coverage` fraction of cells."""
    order = np.argsort(-np.asarray(confidence))
    keep = max(1, round(coverage * len(order)))
    return float(np.asarray(correct)[order[:keep]].mean())


def unknown_detection_auroc(confidence: Any, is_unknown: Any) -> float:
    """How well low confidence separates truly unseen cell types from known ones (0.5 = chance)."""
    return float(roc_auc_score(np.asarray(is_unknown), -np.asarray(confidence)))


def calibration_summary(calib: Any, scores: Any, y: Any, n_bins: int = 15) -> dict[str, float]:
    """Calibration and Unknown-flag quality of a fitted Calibration on one set of cells.

    `scores` are the model's raw scores (one column per class in calib.classes); y are
    the true labels. Works for any model whose scores temperature scaling divides.
    """
    classes = np.asarray(calib.classes)
    y = np.asarray(y).astype(str)
    raw = calibrated_proba(scores, 1.0)
    calibrated = calibrated_proba(scores, calib.temperature)
    correct = classes[calibrated.argmax(axis=1)] == y
    labels, confidence = calib.predict(scores)
    called = labels != UNKNOWN
    return {
        "ece_raw": expected_calibration_error(
            raw.max(axis=1), classes[raw.argmax(axis=1)] == y, n_bins
        ),
        "ece_calibrated": expected_calibration_error(confidence, correct, n_bins),
        "coverage": float(called.mean()),
        "accuracy_called": float((labels[called] == y[called]).mean()) if called.any() else 0.0,
        "accuracy_all": float(correct.mean()),
        "accuracy_top90pct": accuracy_at_coverage(confidence, correct, 0.9),
    }
