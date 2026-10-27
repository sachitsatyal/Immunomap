import numpy as np
import pytest

from immunomap.evaluation.calibration_metrics import (
    accuracy_at_coverage,
    calibration_summary,
    expected_calibration_error,
    unknown_detection_auroc,
)
from immunomap.models.calibration import (
    UNKNOWN,
    Calibration,
    CalibrationConfig,
    calibrate_scores,
    calibrated_proba,
    choose_threshold,
    fit_temperature,
)


def test_calibrated_proba_rows_sum_to_one_and_temperature_flattens() -> None:
    scores = np.array([[2.0, 0.0, -1.0]])
    sharp, flat = calibrated_proba(scores, 1.0), calibrated_proba(scores, 4.0)
    assert sharp.sum() == pytest.approx(1.0)
    assert flat.max() < sharp.max()


def test_fit_temperature_recovers_overconfidence() -> None:
    rng = np.random.default_rng(0)
    true_logits = rng.normal(0, 2, (20_000, 5))
    p = calibrated_proba(true_logits, 1.0)
    y = np.array([rng.choice(5, p=row) for row in p])
    # A model that is 3x too confident: fitted temperature should undo it
    assert fit_temperature(true_logits * 3, y) == pytest.approx(3.0, rel=0.05)


def test_choose_threshold_meets_target_accuracy() -> None:
    confidence = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 1, 0, 1])  # running accuracy: 1, 1, 0.67, 0.75
    assert choose_threshold(confidence, correct, 0.75) == pytest.approx(0.6)
    assert choose_threshold(confidence, correct, 0.9) == pytest.approx(0.8)
    assert choose_threshold(confidence, np.zeros(4), 0.5) == 1.0  # never reachable


def test_low_confidence_cells_are_called_unknown() -> None:
    calib = Calibration(classes=["B", "T"], temperature=1.0, threshold=0.8)
    labels, confidence = calib.predict(np.array([[5.0, 0.0], [0.1, 0.0]]))
    assert labels.tolist() == ["B", UNKNOWN]
    assert confidence[0] > 0.8 > confidence[1]


def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    confidence = np.full(1000, 0.8)
    correct = np.r_[np.ones(800), np.zeros(200)]
    assert expected_calibration_error(confidence, correct) == pytest.approx(0.0, abs=1e-9)
    assert expected_calibration_error(np.ones(4), np.array([1, 0, 1, 0])) == pytest.approx(0.5)


def test_accuracy_at_coverage_uses_most_confident_cells() -> None:
    confidence = np.array([0.99, 0.9, 0.5, 0.4])
    correct = np.array([1, 1, 0, 0])
    assert accuracy_at_coverage(confidence, correct, 0.5) == 1.0
    assert accuracy_at_coverage(confidence, correct, 1.0) == 0.5


def test_unknown_auroc_is_one_when_unseen_cells_are_least_confident() -> None:
    confidence = np.array([0.95, 0.9, 0.3, 0.2])
    assert unknown_detection_auroc(confidence, np.array([0, 0, 1, 1])) == 1.0


def test_calibration_config_loads_from_repo_yaml() -> None:
    cfg = CalibrationConfig.from_yaml("configs/calibration.yaml")
    assert 0.5 < cfg.target_accuracy < 1.0
    assert "Plasmablast" in cfg.holdout_types


def test_calibrate_scores_works_from_log_probabilities() -> None:
    rng = np.random.default_rng(0)
    y = rng.choice(["B", "NK", "T"], 3000)
    # an overconfident model: right 80% of the time, always ~99% sure
    pred = np.where(rng.random(3000) < 0.8, y, rng.choice(["B", "NK", "T"], 3000))
    proba = np.full((3000, 3), 0.005)
    proba[np.arange(3000), np.searchsorted(["B", "NK", "T"], pred)] = 0.99
    calib = calibrate_scores(["B", "NK", "T"], np.log(proba), y, target_accuracy=0.95)
    assert calib.temperature > 1.0  # overconfidence is softened
    summary = calibration_summary(calib, np.log(proba), y)
    assert summary["ece_calibrated"] < summary["ece_raw"]


def test_calibrate_scores_requires_sorted_classes() -> None:
    with pytest.raises(ValueError):
        calibrate_scores(["T", "B"], np.zeros((2, 2)), ["T", "B"], 0.95)
