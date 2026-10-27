"""Train and evaluate the logistic-regression baseline, logging everything to MLflow.

One model per label level (coarse, fine). Each is trained on the train split and
scored on val, test (overall and per study) and the external studies (per study).

Usage:
    uv run python scripts/train_baseline.py
    uv run mlflow ui --backend-store-uri sqlite:///mlflow.db     # then open http://127.0.0.1:5000
"""

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from immunomap.data.io import load_split
from immunomap.evaluation.scoring import evaluate, log_result
from immunomap.models.baseline import LEVELS, BaselineConfig, build_pipeline, labelled_cells
from immunomap.utils.tracking import data_fingerprint, git_commit, setup_experiment

EXPERIMENT = "immunomap-baseline"


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

            for split, (X, obs) in evaluation.items():
                result = evaluate(model.predict(X), obs, level, split)
                log_result(result)
                summary += [{"level": level, **row} for row in result["rows"]]
            try:
                mlflow.sklearn.log_model(model, name="model")
            except TypeError:  # older MLflow
                mlflow.sklearn.log_model(model, artifact_path="model")
            print(f"{level}: trained on {len(idx):,} cells in {time.time() - start:.0f} s")

    print("\n", pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
