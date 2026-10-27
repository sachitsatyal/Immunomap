"""Calibrated confidence and the Unknown flag for scANVI (the main model).

Runs (MLflow experiment immunomap-scanvi-calibration), using the models saved by
scripts/train_scanvi.py:
  scanvi-{coarse,fine}-calibrated  temperature + Unknown threshold fitted on val; ECE,
                                   coverage and accuracy on test and each external study
  scanvi-novelty-full              distance-to-reference in scANVI's latent space (coarse
                                   model); % of known cells wrongly flagged; saves the
                                   detector the product will load
  scanvi-holdout                   scVI + scANVI retrained WITHOUT the holdout types
                                   (configs/calibration.yaml); are those cells flagged?

Usage:
    uv run python scripts/calibrate_scanvi.py --smoke 20000   # quick check, tiny retrain
    uv run python scripts/calibrate_scanvi.py                 # full run
"""

import argparse
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import anndata as ad
import joblib
import mlflow
import numpy as np
import pandas as pd
import torch

from immunomap.evaluation.calibration_metrics import calibration_summary, unknown_detection_auroc
from immunomap.models.baseline import LEVELS, labelled_cells
from immunomap.models.calibration import UNKNOWN, Calibration, CalibrationConfig, calibrate_scores
from immunomap.models.novelty import NoveltyConfig, ReferenceDistance, reference_sample
from immunomap.models.scanvi import (
    ScanviConfig,
    load_counts,
    load_scanvi,
    log_scores,
    n_cells,
    predict_proba,
    train_scanvi,
    train_scvi,
)
from immunomap.utils.tracking import data_fingerprint, git_commit, setup_experiment

EXPERIMENT = "immunomap-scanvi-calibration"
EVAL_SPLITS = ["val", "test", "external"]
Prediction = tuple[pd.DataFrame, Any, bool]  # probabilities, latent, mapped


def sample_rows(path: Path, n: int | None, seed: int) -> np.ndarray | None:
    if n is None:
        return None
    rng = np.random.default_rng(seed)
    total = n_cells(path)
    return np.sort(rng.choice(total, min(n, total), replace=False))


def scopes(adata: ad.AnnData, split: str) -> dict[str, np.ndarray]:
    """Cell masks to report on: the whole split for test, each study for external."""
    if split != "external":
        return {split: np.ones(adata.n_obs, dtype=bool)}
    groups = adata.obs["group"].astype(str).to_numpy()
    return {f"external_{g}": groups == g for g in sorted(set(groups))}


def fit_on_val(proba: pd.DataFrame, obs: pd.DataFrame, level: str, target: float) -> Calibration:
    idx, y = labelled_cells(obs, level)
    return calibrate_scores(list(proba.columns), log_scores(proba)[idx], y, target)


def calibration_runs(
    preds: dict[str, dict[str, Prediction]],
    data: dict[str, ad.AnnData],
    ccfg: CalibrationConfig,
    params: dict[str, Any],
) -> dict[str, Calibration]:
    calibrations, rows = {}, []
    for level, by_split in preds.items():
        with (
            mlflow.start_run(run_name=f"scanvi-{level}-calibrated"),
            tempfile.TemporaryDirectory() as tmp,
        ):
            mlflow.log_params({**params, "level": level, "target_accuracy": ccfg.target_accuracy})
            calib = fit_on_val(by_split["val"][0], data["val"].obs, level, ccfg.target_accuracy)
            calibrations[level] = calib
            mlflow.log_params(
                {
                    "temperature": round(calib.temperature, 4),
                    "unknown_threshold": round(calib.threshold, 4),
                }
            )
            for split in ["test", "external"]:
                proba, _, _ = by_split[split]
                scores = log_scores(proba)
                for scope, mask in scopes(data[split], split).items():
                    idx, y = labelled_cells(data[split].obs[mask], level)
                    if len(idx) == 0:
                        continue
                    result = calibration_summary(calib, scores[mask][idx], y, ccfg.ece_bins)
                    mlflow.log_metrics({f"{scope}_{k}": v for k, v in result.items()})
                    rows.append({"level": level, "scope": scope, **result})
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(calib.to_dict(), indent=2))
            mlflow.log_artifact(str(path))
            print(
                f"{level}: temperature {calib.temperature:.3f}, Unknown below {calib.threshold:.3f}"
            )
    print("\n", pd.DataFrame(rows).round(3).to_string(index=False))
    return calibrations


def novelty_full(
    model: Any,
    train: ad.AnnData,
    preds: dict[str, Prediction],
    data: dict[str, ad.AnnData],
    calib: Calibration,
    ncfg: NoveltyConfig,
    params: dict[str, Any],
) -> dict[str, float]:
    """Distance detector in the coarse model's latent space; false flags on known cells."""
    with mlflow.start_run(run_name="scanvi-novelty-full"), tempfile.TemporaryDirectory() as tmp:
        mlflow.log_params({**params, **asdict(ncfg), "space": "scanvi_coarse_latent"})
        idx, y = labelled_cells(train.obs, "coarse")
        ref = idx[reference_sample(y, ncfg.ref_per_type, ncfg.seed)]
        reference = model.get_latent_representation(train, indices=ref)
        detector = ReferenceDistance(ncfg, embed=False).fit(np.asarray(reference))
        metrics = {"distance_threshold": detector.set_threshold(detector.score(preds["val"][1]))}
        for split in ["test", "external"]:
            proba, latent, _ = preds[split]
            far = detector.is_novel(detector.score(latent))
            unsure = calib.predict(log_scores(proba))[0] == UNKNOWN
            for scope, mask in scopes(data[split], split).items():
                metrics[f"{scope}_flagged_distance"] = float(far[mask].mean())
                metrics[f"{scope}_flagged_combined"] = float((far | unsure)[mask].mean())
        mlflow.log_metrics(metrics)
        path = Path(tmp) / "reference_distance_scanvi.joblib"
        joblib.dump(detector, path)  # the product will load this
        mlflow.log_artifact(str(path))
    return metrics


def holdout_run(
    train: ad.AnnData,
    data: dict[str, ad.AnnData],
    cfg: ScanviConfig,
    ccfg: CalibrationConfig,
    ncfg: NoveltyConfig,
    params: dict[str, Any],
) -> dict[str, float]:
    """Retrain scVI + coarse scANVI without the holdout types, then try to flag them."""
    holdout = ccfg.holdout_types
    with mlflow.start_run(run_name="scanvi-holdout"):
        mlflow.log_params({**params, "holdout_types": ",".join(holdout)})
        keep = ~train.obs["coarse"].astype(str).isin(holdout).to_numpy()
        subset = train[keep].copy()  # holdout cells never seen, not even unlabelled
        vae = train_scvi(subset, cfg)
        model = train_scanvi(vae, "coarse", cfg)
        known_batches = set(subset.obs[cfg.batch_key].astype(str))

        val_proba, val_latent, _ = predict_proba(model, data["val"], cfg, known_batches)
        v_idx, v_y = labelled_cells(data["val"].obs, "coarse")
        v_known = ~np.isin(v_y, holdout)
        calib = calibrate_scores(
            list(val_proba.columns),
            log_scores(val_proba)[v_idx[v_known]],
            v_y[v_known],
            ccfg.target_accuracy,
        )
        idx, y = labelled_cells(subset.obs, "coarse")
        ref = idx[reference_sample(y, ncfg.ref_per_type, ncfg.seed)]
        detector = ReferenceDistance(ncfg, embed=False).fit(
            np.asarray(model.get_latent_representation(subset, indices=ref))
        )
        detector.set_threshold(detector.score(val_latent[v_idx[v_known]]))

        proba, latent, _ = predict_proba(model, data["test"], cfg, known_batches)
        t_idx, t_y = labelled_cells(data["test"].obs, "coarse")
        labels, confidence = calib.predict(log_scores(proba)[t_idx])
        distance = detector.score(latent[t_idx])
        by_distance = detector.is_novel(distance)
        combined = by_distance | (labels == UNKNOWN)
        unseen = np.isin(t_y, holdout)
        metrics = {
            "auroc_confidence": unknown_detection_auroc(confidence, unseen),
            "auroc_distance": unknown_detection_auroc(-distance, unseen),
            "known_flagged_distance": float(by_distance[~unseen].mean()),
            "known_flagged_combined": float(combined[~unseen].mean()),
        }
        for cell_type in holdout:
            mask = t_y == cell_type
            metrics[f"{cell_type}_flagged_confidence"] = float((labels[mask] == UNKNOWN).mean())
            metrics[f"{cell_type}_flagged_distance"] = float(by_distance[mask].mean())
            metrics[f"{cell_type}_flagged_combined"] = float(combined[mask].mean())
        mlflow.log_metrics(metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scanvi.yaml")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--models-dir", default="data/models/scanvi")
    parser.add_argument("--smoke", type=int, metavar="N_CELLS", help="quick check on N cells")
    args = parser.parse_args()

    cfg = ScanviConfig.from_yaml(args.config)
    ccfg = CalibrationConfig.from_yaml("configs/calibration.yaml")
    ncfg = NoveltyConfig.from_yaml("configs/novelty.yaml")
    smoke = args.smoke is not None
    if smoke:
        cfg = replace(cfg, scvi_epochs=2, scanvi_epochs=2, query_epochs=2)
        ncfg = replace(ncfg, ref_per_type=500)
    torch.set_float32_matmul_precision("high")
    features = Path(args.features_dir)
    models = Path(args.models_dir)

    train = load_counts(
        features / "train.h5ad", sample_rows(features / "train.h5ad", args.smoke, cfg.seed)
    )
    eval_n = None if args.smoke is None else max(args.smoke // 4, 1000)
    data = {
        s: load_counts(
            features / f"{s}.h5ad", sample_rows(features / f"{s}.h5ad", eval_n, cfg.seed)
        )
        for s in EVAL_SPLITS
    }
    setup_experiment(EXPERIMENT + ("-smoke" if smoke else ""))
    params = {"data_md5": data_fingerprint(), "git_commit": git_commit(), "models": str(models)}

    preds: dict[str, dict[str, Prediction]] = {}
    for level in LEVELS:
        model = load_scanvi(models / f"scanvi_{level}", train, level, cfg)
        known = set(train.obs[cfg.batch_key].astype(str))
        preds[level] = {s: predict_proba(model, data[s], cfg, known) for s in EVAL_SPLITS}

    calibrations = calibration_runs(preds, data, ccfg, params)
    # Reload: loading the fine model re-attached the training data with fine labels
    coarse_model = load_scanvi(models / "scanvi_coarse", train, "coarse", cfg)
    full = novelty_full(
        coarse_model, train, preds["coarse"], data, calibrations["coarse"], ncfg, params
    )
    held = holdout_run(train, data, cfg, ccfg, ncfg, params)

    print("\nKnown cells flagged Unknown (full model, coarse):")
    for key, value in full.items():
        if key.endswith("_flagged_distance"):
            scope = key.removesuffix("_flagged_distance")
            combined = full[f"{scope}_flagged_combined"]
            print(f"  {scope:24s} distance {value:.1%}, combined {combined:.1%}")
    print("\nHeld-out types:", ", ".join(ccfg.holdout_types))
    print(f"  AUROC confidence {held['auroc_confidence']:.3f}")
    print(f"  AUROC distance   {held['auroc_distance']:.3f}")
    for cell_type in ccfg.holdout_types:
        print(
            f"  {cell_type:12s} flagged: confidence "
            f"{held[f'{cell_type}_flagged_confidence']:.1%}, distance "
            f"{held[f'{cell_type}_flagged_distance']:.1%}, combined "
            f"{held[f'{cell_type}_flagged_combined']:.1%}"
        )
    print(
        f"  known cells wrongly flagged: distance {held['known_flagged_distance']:.1%}, "
        f"combined {held['known_flagged_combined']:.1%}"
    )


if __name__ == "__main__":
    main()
