"""Test the distance-to-reference detector for cell types the model has never seen.

1. Held-out test: build the classifier AND the reference without the holdout types
   (configs/calibration.yaml), then ask how well each signal flags those unseen cells:
   calibrated confidence alone, distance alone, and the two combined.
2. Full reference: build the detector from all training types and measure how many
   KNOWN cells it wrongly flags on test and on each external study. A high rate on an
   external study means distance is picking up batch effects, not novel biology.

Usage:
    uv run python scripts/evaluate_novelty.py
"""

import argparse
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd

from immunomap.data.io import load_split
from immunomap.evaluation.calibration_metrics import unknown_detection_auroc
from immunomap.models.baseline import BaselineConfig, build_pipeline, labelled_cells
from immunomap.models.calibration import UNKNOWN, CalibrationConfig, fit_calibration
from immunomap.models.novelty import NoveltyConfig, ReferenceDistance, reference_sample
from immunomap.utils.tracking import data_fingerprint, git_commit, setup_experiment

EXPERIMENT = "immunomap-novelty"
LEVEL = "coarse"


def build_detector(cfg: NoveltyConfig, X_train: Any, idx: Any, y: Any) -> ReferenceDistance:
    ref = idx[reference_sample(y, cfg.ref_per_type, cfg.seed)]
    return ReferenceDistance(cfg).fit(X_train[ref])


def holdout_test(
    bcfg: BaselineConfig,
    ccfg: CalibrationConfig,
    ncfg: NoveltyConfig,
    data: dict[str, tuple[Any, pd.DataFrame]],
) -> dict[str, float]:
    holdout = ccfg.holdout_types
    X_train, obs_train = data["train"]
    idx, y = labelled_cells(obs_train, LEVEL)
    keep = ~np.isin(y, holdout)

    detector = build_detector(ncfg, X_train, idx[keep], y[keep])
    clf = build_pipeline(bcfg).fit(X_train[idx[keep]], y[keep])

    X_val, obs_val = data["val"]
    v_idx, v_y = labelled_cells(obs_val, LEVEL)
    known = ~np.isin(v_y, holdout)
    detector.set_threshold(detector.score(X_val[v_idx[known]]))
    calib = fit_calibration(
        clf, clf.decision_function(X_val[v_idx[known]]), v_y[known], ccfg.target_accuracy
    )

    X_test, obs_test = data["test"]
    t_idx, t_y = labelled_cells(obs_test, LEVEL)
    distance = detector.score(X_test[t_idx])
    labels, confidence = calib.predict(clf.decision_function(X_test[t_idx]))
    by_distance = detector.is_novel(distance)
    by_confidence = labels == UNKNOWN
    combined = by_distance | by_confidence
    unseen = np.isin(t_y, holdout)

    metrics = {
        "auroc_confidence": unknown_detection_auroc(confidence, unseen),
        "auroc_distance": unknown_detection_auroc(-distance, unseen),
        "known_flagged_distance": float(by_distance[~unseen].mean()),
        "known_flagged_combined": float(combined[~unseen].mean()),
        "distance_threshold": detector.threshold,
    }
    for cell_type in holdout:
        mask = t_y == cell_type
        metrics[f"{cell_type}_flagged_distance"] = float(by_distance[mask].mean())
        metrics[f"{cell_type}_flagged_combined"] = float(combined[mask].mean())
    return metrics


def full_reference(
    ncfg: NoveltyConfig, data: dict[str, tuple[Any, pd.DataFrame]], out_dir: Path
) -> dict[str, float]:
    X_train, obs_train = data["train"]
    idx, y = labelled_cells(obs_train, LEVEL)
    detector = build_detector(ncfg, X_train, idx, y)
    X_val, obs_val = data["val"]
    v_idx, _ = labelled_cells(obs_val, LEVEL)
    metrics = {"distance_threshold": detector.set_threshold(detector.score(X_val[v_idx]))}

    X_test, obs_test = data["test"]
    t_idx, _ = labelled_cells(obs_test, LEVEL)
    metrics["test_flagged"] = float(detector.is_novel(detector.score(X_test[t_idx])).mean())
    X_ext, obs_ext = data["external"]
    groups = obs_ext["group"].astype(str).to_numpy()
    flagged = detector.is_novel(detector.score(X_ext))
    for group in sorted(set(groups)):
        metrics[f"external_{group}_flagged"] = float(flagged[groups == group].mean())

    joblib.dump(detector, out_dir / "reference_distance.joblib")  # the product will load this
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-dir", default="data/processed/features")
    args = parser.parse_args()

    bcfg = BaselineConfig.from_yaml("configs/baseline.yaml")
    ccfg = CalibrationConfig.from_yaml("configs/calibration.yaml")
    ncfg = NoveltyConfig.from_yaml("configs/novelty.yaml")
    features = Path(args.features_dir)
    data = {s: load_split(features / f"{s}.h5ad") for s in ["train", "val", "test", "external"]}
    setup_experiment(EXPERIMENT)
    params = {**asdict(ncfg), "data_md5": data_fingerprint(), "git_commit": git_commit()}

    with mlflow.start_run(run_name="novelty-holdout"):
        mlflow.log_params({**params, "holdout_types": ",".join(ccfg.holdout_types)})
        held = holdout_test(bcfg, ccfg, ncfg, data)
        mlflow.log_metrics(held)

    with mlflow.start_run(run_name="novelty-full-reference"), tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params(params)
        full = full_reference(ncfg, data, Path(tmp))
        mlflow.log_metrics(full)
        mlflow.log_artifact(str(Path(tmp) / "reference_distance.joblib"))

    print("\nHeld-out types:", ", ".join(ccfg.holdout_types))
    print(f"  AUROC confidence {held['auroc_confidence']:.3f}")
    print(f"  AUROC distance   {held['auroc_distance']:.3f}")
    for cell_type in ccfg.holdout_types:
        print(
            f"  {cell_type:12s} flagged: distance {held[f'{cell_type}_flagged_distance']:.1%}, "
            f"combined {held[f'{cell_type}_flagged_combined']:.1%}"
        )
    print(
        f"  known cells wrongly flagged: distance {held['known_flagged_distance']:.1%}, "
        f"combined {held['known_flagged_combined']:.1%}"
    )
    print("\nFull reference, known cells wrongly flagged:")
    for key, value in full.items():
        if key.endswith("_flagged"):
            print(f"  {key.removesuffix('_flagged'):24s} {value:.1%}")


if __name__ == "__main__":
    main()
