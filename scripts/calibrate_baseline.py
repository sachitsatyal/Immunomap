"""Calibrate the baseline's confidence, set the Unknown threshold, test unknown detection.

Per label level: train the baseline, fit the temperature and the Unknown threshold on
val, then report calibration error and coverage on test and on each external study.
Then a held-out experiment: retrain the coarse model WITHOUT the holdout types in
configs/calibration.yaml and measure whether those never-seen cells get flagged Unknown.

Usage:
    uv run python scripts/calibrate_baseline.py
"""

import argparse
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from immunomap.data.io import load_split
from immunomap.evaluation.calibration_metrics import (
    accuracy_at_coverage,
    expected_calibration_error,
    unknown_detection_auroc,
)
from immunomap.models.baseline import LEVELS, BaselineConfig, build_pipeline, labelled_cells
from immunomap.models.calibration import (
    UNKNOWN,
    Calibration,
    CalibrationConfig,
    calibrated_proba,
    choose_threshold,
    fit_temperature,
)
from immunomap.utils.tracking import data_fingerprint, git_commit, setup_experiment

EXPERIMENT = "immunomap-calibration"


def scored(model: Any, X: Any, obs: pd.DataFrame, level: str) -> tuple[Any, Any, Any]:
    """Classifier scores, true labels and study group for cells labelled at this level."""
    idx, y = labelled_cells(obs, level)
    return model.decision_function(X[idx]), y, obs["group"].astype(str).to_numpy()[idx]


def fit_calibration(model: Any, scores: Any, y: Any, target: float) -> Calibration:
    classes = [str(c) for c in model.classes_]
    y_index = np.searchsorted(model.classes_, y)
    temperature = fit_temperature(scores, y_index)
    proba = calibrated_proba(scores, temperature)
    correct = proba.argmax(axis=1) == y_index
    threshold = choose_threshold(proba.max(axis=1), correct, target)
    return Calibration(classes=classes, temperature=temperature, threshold=threshold)


def report(model: Any, calib: Calibration, scores: Any, y: Any, bins: int) -> dict[str, float]:
    raw = calibrated_proba(scores, 1.0)  # temperature 1 = uncalibrated softmax of scores
    raw_correct = np.asarray(model.classes_)[raw.argmax(axis=1)] == y
    labels, confidence = calib.predict(scores)
    called = labels != UNKNOWN
    correct = (
        np.asarray(model.classes_)[np.argmax(calibrated_proba(scores, calib.temperature), 1)] == y
    )
    return {
        "ece_raw": expected_calibration_error(raw.max(axis=1), raw_correct, bins),
        "ece_calibrated": expected_calibration_error(confidence, correct, bins),
        "coverage": float(called.mean()),
        "accuracy_called": float((labels[called] == y[called]).mean()) if called.any() else 0.0,
        "accuracy_all": float(correct.mean()),
        "accuracy_top90pct": accuracy_at_coverage(confidence, correct, 0.9),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", default="configs/baseline.yaml")
    parser.add_argument("--config", default="configs/calibration.yaml")
    parser.add_argument("--features-dir", default="data/processed/features")
    args = parser.parse_args()

    bcfg = BaselineConfig.from_yaml(args.baseline_config)
    ccfg = CalibrationConfig.from_yaml(args.config)
    features = Path(args.features_dir)
    X_train, obs_train = load_split(features / "train.h5ad")
    X_val, obs_val = load_split(features / "val.h5ad")
    X_test, obs_test = load_split(features / "test.h5ad")
    X_ext, obs_ext = load_split(features / "external.h5ad")
    setup_experiment(EXPERIMENT)
    provenance = {"data_md5": data_fingerprint(), "git_commit": git_commit()}

    rows = []
    for level in LEVELS:
        idx, y = labelled_cells(obs_train, level)
        with (
            mlflow.start_run(run_name=f"logreg-{level}-calibrated"),
            tempfile.TemporaryDirectory() as tmp,
        ):
            mlflow.log_params(
                {
                    **asdict(bcfg),
                    **provenance,
                    "level": level,
                    "target_accuracy": ccfg.target_accuracy,
                }
            )
            model = build_pipeline(bcfg).fit(X_train[idx], y)
            calib = fit_calibration(
                model, *scored(model, X_val, obs_val, level)[:2], ccfg.target_accuracy
            )
            mlflow.log_params(
                {
                    "temperature": round(calib.temperature, 4),
                    "unknown_threshold": round(calib.threshold, 4),
                }
            )

            test_scores, test_y, _ = scored(model, X_test, obs_test, level)
            scopes = {"test": (test_scores, test_y)}
            ext_scores, ext_y, ext_groups = scored(model, X_ext, obs_ext, level)
            for g in sorted(set(ext_groups)):
                scopes[f"external_{g}"] = (ext_scores[ext_groups == g], ext_y[ext_groups == g])
            for scope, (s, yy) in scopes.items():
                result = report(model, calib, s, yy, ccfg.ece_bins)
                mlflow.log_metrics({f"{scope}_{k}": v for k, v in result.items()})
                rows.append({"level": level, "scope": scope, **result})

            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(calib.to_dict(), indent=2))
            mlflow.log_artifact(str(path))
            print(
                f"{level}: temperature {calib.temperature:.3f}, Unknown below {calib.threshold:.3f}"
            )

    print("\n", pd.DataFrame(rows).round(3).to_string(index=False))
    holdout_experiment(bcfg, ccfg, provenance, X_train, obs_train, X_val, obs_val, X_test, obs_test)


def holdout_experiment(
    bcfg: BaselineConfig,
    ccfg: CalibrationConfig,
    provenance: dict[str, str],
    X_train: Any,
    obs_train: pd.DataFrame,
    X_val: Any,
    obs_val: pd.DataFrame,
    X_test: Any,
    obs_test: pd.DataFrame,
) -> None:
    """Retrain coarse without the holdout types; are those cells flagged Unknown on test?"""
    holdout = ccfg.holdout_types
    with mlflow.start_run(run_name="logreg-coarse-holdout"), tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params({**asdict(bcfg), **provenance, "holdout_types": ",".join(holdout)})
        idx, y = labelled_cells(obs_train, "coarse")
        keep = ~np.isin(y, holdout)
        model = build_pipeline(bcfg).fit(X_train[idx[keep]], y[keep])

        val_scores, val_y, _ = scored(model, X_val, obs_val, "coarse")
        known_val = ~np.isin(val_y, holdout)
        calib = fit_calibration(
            model, val_scores[known_val], val_y[known_val], ccfg.target_accuracy
        )

        scores, y_test, _ = scored(model, X_test, obs_test, "coarse")
        labels, confidence = calib.predict(scores)
        is_unknown = np.isin(y_test, holdout)
        metrics = {
            "unknown_auroc": unknown_detection_auroc(confidence, is_unknown),
            "known_flagged_rate": float((labels[~is_unknown] == UNKNOWN).mean()),
        }
        print(
            f"\nHeld-out types {holdout}: AUROC {metrics['unknown_auroc']:.3f}, "
            f"known cells wrongly flagged {metrics['known_flagged_rate']:.1%}"
        )
        calls = []
        for cell_type in holdout:
            mask = y_test == cell_type
            flagged = float((labels[mask] == UNKNOWN).mean())
            metrics[f"{cell_type}_flagged_rate"] = flagged
            top = pd.Series(labels[mask]).value_counts(normalize=True).head(3).round(3)
            calls.append(top.rename(cell_type))
            print(
                f"  {cell_type}: {mask.sum():,} test cells, {flagged:.1%} flagged Unknown; "
                f"called as {top.to_dict()}"
            )
        mlflow.log_metrics(metrics)
        path = Path(tmp) / "holdout_calls.csv"
        pd.concat(calls, axis=1).to_csv(path, lineterminator="\n")
        mlflow.log_artifact(str(path))


if __name__ == "__main__":
    main()
