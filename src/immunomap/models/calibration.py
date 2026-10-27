"""Calibrated confidence and the "Unknown" flag.

Raw classifier scores are over- or under-confident. Temperature scaling fits one number T
on the validation set so that softmax(scores / T) matches how often the model is actually
right. A cell whose calibrated confidence falls below a threshold is reported as "Unknown"
instead of being forced into a cell type.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax, softmax

UNKNOWN = "Unknown"


@dataclass(frozen=True)
class CalibrationConfig:
    target_accuracy: float  # accepted (non-Unknown) cells must reach this accuracy on val
    ece_bins: int
    holdout_types: list[str]  # coarse types removed from training to test unknown detection

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls(**yaml.safe_load(Path(path).read_text()))


def calibrated_proba(scores: Any, temperature: float) -> Any:
    """Softmax of classifier scores divided by the temperature."""
    return softmax(np.asarray(scores, dtype=np.float64) / temperature, axis=1)


def fit_temperature(scores: Any, y_index: Any) -> float:
    """Temperature minimizing the negative log-likelihood of the true classes (fit on val)."""
    z = np.asarray(scores, dtype=np.float64)
    rows = np.arange(len(z))

    def nll(log_t: float) -> float:
        return float(-log_softmax(z / np.exp(log_t), axis=1)[rows, y_index].mean())

    result = minimize_scalar(nll, bounds=(-5.0, 5.0), method="bounded")
    return float(np.exp(result.x))


def choose_threshold(confidence: Any, correct: Any, target_accuracy: float) -> float:
    """Lowest confidence threshold at which the accepted cells reach the target accuracy.

    Cells are accepted from most to least confident; the threshold is the confidence of
    the last cell that keeps running accuracy at or above the target.
    """
    order = np.argsort(-np.asarray(confidence))
    conf = np.asarray(confidence)[order]
    running = np.cumsum(np.asarray(correct)[order]) / np.arange(1, len(conf) + 1)
    ok = np.flatnonzero(running >= target_accuracy)
    return float(conf[ok[-1]]) if len(ok) else 1.0


@dataclass(frozen=True)
class Calibration:
    """Everything the product needs to turn classifier scores into calibrated calls."""

    classes: list[str]
    temperature: float
    threshold: float

    def predict(self, scores: Any) -> tuple[Any, Any]:
        """Labels (with "Unknown" below the threshold) and calibrated confidence."""
        proba = calibrated_proba(scores, self.temperature)
        confidence = proba.max(axis=1)
        labels = np.asarray(self.classes, dtype=object)[proba.argmax(axis=1)]
        labels[confidence < self.threshold] = UNKNOWN
        return labels, confidence

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def fit_calibration(model: Any, scores: Any, y: Any, target_accuracy: float) -> Calibration:
    """Fit temperature and Unknown threshold on validation scores of a fitted classifier."""
    return calibrate_scores(list(model.classes_), scores, y, target_accuracy)


def calibrate_scores(classes: Any, scores: Any, y: Any, target_accuracy: float) -> Calibration:
    """Same as fit_calibration, for any model: `scores` has one column per class in `classes`
    (sorted), e.g. log-probabilities from scANVI."""
    classes = [str(c) for c in classes]
    if classes != sorted(classes):
        raise ValueError("classes must be sorted (score columns in sorted order)")
    y_index = np.searchsorted(np.asarray(classes), np.asarray(y).astype(str))
    temperature = fit_temperature(scores, y_index)
    proba = calibrated_proba(scores, temperature)
    correct = proba.argmax(axis=1) == y_index
    threshold = choose_threshold(proba.max(axis=1), correct, target_accuracy)
    return Calibration(classes=classes, temperature=temperature, threshold=threshold)
