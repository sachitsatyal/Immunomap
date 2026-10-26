"""Train and evaluate the logistic-regression baseline, logging everything to MLflow.

One model per label level (coarse, fine). Each is trained on the train split and
scored on val, test (overall and per study) and the external studies.

Usage:
    uv run python scripts/train_baseline.py
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000
"""

import argparse
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from immunomap.data.io import load_split
from immunomap.evaluation.metrics import classification_metrics, per_class_report
from immunomap.models.baseline import LEVELS, BaselineConfig, build_pipeline, labelled_cells
from immunomap.utils.tracking import data_fingerprint, git_commit, setup_experiment

EXPERIMENT = "immunomap-baseline"

# External studies label at different depths (Lupus has no Treg or MAIT label, Hao does).
# Pooling them would score a correct "Treg" call on a Lupus cell as a false positive for
# Hao's Treg class, so external studies are only ever scored one study at a time.
POOLED_SPLITS = {"val", "test"}


def evaluate(model: Any, X: Any, obs: pd.DataFrame, level: str, split: str) -> dict[str, Any]:
    """Score a split (pooled for val/test) and each study group in it.

    Returns MLflow metrics, per-class reports and summary rows.
    """
    result: dict[str, Any] = {"metrics": {}, "reports": {}, "rows": []}
    idx, y_true = labelled_cells(obs, level)
    if len(idx) == 0:
        return result
    y_pred = model.predict(X[idx])
    groups = obs["group"].astype(str).to_numpy()[idx]

    scopes = [(split, np.ones(len(idx), dtype=bool))] if split in POOLED_SPLITS else []
    scopes += [(f"{split}_{g}", groups == g) for g in sorted(set(groups))]
    for name, mask in scopes:
        scores = classification_metrics(y_true[mask], y_pred[mask])
        result["metrics"].update({f"{name}_{k}": v for k, v in scores.items()})
        result["metrics"][f"{name}_n_cells"] = float(mask.sum())
        headline = name == split or split not in POOLED_SPLITS
        if headline:
            result["reports"][name] = per_class_report(y_true[mask], y_pred[mask])
            result["rows"].append(
                {
                    "scope": name,
                    "cells": int(mask.sum()),
                    "macro_f1": scores["macro_f1"],
                    "accuracy": scores["accuracy"],
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    args = parser.parse_args()

    cfg = BaselineConfig.from_yaml(args.config)
    features = Path(args.features_dir)
    X_train, obs_train = load_split(features / "train.h5ad")
    evaluation = {s: load_split(features / f"{s}.h5ad") for s in ["val", "test", "external"]}
    n_genes = X_train.shape[1]

    setup_experiment(EXPERIMENT)

    summary = []
    for level in args.levels:
        idx, y = labelled_cells(obs_train, level)
        with mlflow.start_run(run_name=f"logreg-{level}"):
            mlflow.log_params(asdict(cfg))
            mlflow.log_params(
                {
                    "model": "logistic_regression_sgd",
                    "level": level,
                    "n_genes": n_genes,
                    "n_train_cells": len(idx),
                    "n_classes": len(set(y)),
                    "data_md5": data_fingerprint(),
                    "git_commit": git_commit(),
                }
            )

            start = time.time()
            model = build_pipeline(cfg).fit(X_train[idx], y)
            mlflow.log_metric("train_seconds", time.time() - start)

            with tempfile.TemporaryDirectory() as tmp:
                for split, (X, obs) in evaluation.items():
                    result = evaluate(model, X, obs, level, split)
                    mlflow.log_metrics(result["metrics"])
                    for name, report in result["reports"].items():
                        path = Path(tmp) / f"per_class_{name}.csv"
                        report.to_csv(path, lineterminator="\n")
                        mlflow.log_artifact(str(path), artifact_path="reports")
                    summary += [{"level": level, **row} for row in result["rows"]]
            try:
                mlflow.sklearn.log_model(model, name="model")
            except TypeError:  # older MLflow
                mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"{level}: trained on {len(idx):,} cells in {time.time() - start:.0f} s")

    print("\n", pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
