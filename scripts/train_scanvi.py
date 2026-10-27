"""Train scVI, then one scANVI classifier per label level, and score them like the baseline.

Runs (MLflow experiment immunomap-scanvi):
  scvi-pretrain     unsupervised scVI on training cells (batch = source dataset)
  scanvi-coarse     scANVI from scVI, coarse labels; scored on val, test, external
  scanvi-fine       same for fine labels
Splits with a dataset the model was not trained on (the external studies) are scored
after reference mapping, which uses their expression only, never their labels.
Models are saved to data/models/scanvi/ and logged to MLflow.

Usage:
    uv run python scripts/train_scanvi.py --smoke 20000   # 2-minute check: tiny data, 2 epochs
    uv run python scripts/train_scanvi.py                 # full run
"""

import argparse
import time
from dataclasses import asdict, replace
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

from immunomap.evaluation.scoring import evaluate, log_result
from immunomap.models.baseline import LEVELS
from immunomap.models.scanvi import (
    ScanviConfig,
    load_counts,
    n_cells,
    predict_proba,
    top_label,
    train_scanvi,
    train_scvi,
)
from immunomap.utils.tracking import data_fingerprint, git_commit, setup_experiment

EXPERIMENT = "immunomap-scanvi"
EVAL_SPLITS = ["val", "test", "external"]


def sample_rows(path: Path, n: int | None, seed: int) -> np.ndarray | None:
    """Random cell positions for smoke runs; None means all cells."""
    if n is None:
        return None
    total = n_cells(path)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(total, min(n, total), replace=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scanvi.yaml")
    parser.add_argument("--features-dir", default="data/processed/features")
    parser.add_argument("--levels", nargs="+", default=list(LEVELS))
    parser.add_argument("--smoke", type=int, metavar="N_CELLS", help="quick check on N cells")
    args = parser.parse_args()

    cfg = ScanviConfig.from_yaml(args.config)
    smoke = args.smoke is not None
    if smoke:
        cfg = replace(cfg, scvi_epochs=2, scanvi_epochs=2, query_epochs=2)
    torch.set_float32_matmul_precision("high")  # faster matrix maths on recent NVIDIA GPUs
    features = Path(args.features_dir)
    out_dir = Path("data/models") / ("scanvi-smoke" if smoke else "scanvi")
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_counts(
        features / "train.h5ad", sample_rows(features / "train.h5ad", args.smoke, cfg.seed)
    )
    eval_n = None if args.smoke is None else max(args.smoke // 4, 1000)
    evaluation = {
        s: load_counts(
            features / f"{s}.h5ad", sample_rows(features / f"{s}.h5ad", eval_n, cfg.seed)
        )
        for s in EVAL_SPLITS
    }
    print(f"train {train.n_obs:,} cells x {train.n_vars:,} genes;", end=" ")
    print(", ".join(f"{s} {a.n_obs:,}" for s, a in evaluation.items()))

    setup_experiment(EXPERIMENT + ("-smoke" if smoke else ""))
    params = {
        **asdict(cfg),
        "n_genes": train.n_vars,
        "n_train_cells": train.n_obs,
        "data_md5": data_fingerprint(),
        "git_commit": git_commit(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }

    with mlflow.start_run(run_name="scvi-pretrain"):
        mlflow.log_params({**params, "model": "scvi"})
        start = time.time()
        vae = train_scvi(train, cfg)
        mlflow.log_metric("train_seconds", time.time() - start)
        elbo = vae.history.get("elbo_train")
        if elbo is not None:  # training loss curve: lower is better
            mlflow.log_metric("elbo_train_final", float(elbo.iloc[-1, 0]))
        vae.save(str(out_dir / "scvi"), overwrite=True, save_anndata=False)
        print(f"scVI trained in {time.time() - start:.0f} s")
    known_batches = set(train.obs[cfg.batch_key].astype(str))

    summary = []
    for level in args.levels:
        with mlflow.start_run(run_name=f"scanvi-{level}"):
            n_labelled = int(train.obs[level].notna().sum())
            mlflow.log_params({**params, "model": "scanvi", "level": level})
            mlflow.log_params({"n_labelled_train_cells": n_labelled})
            start = time.time()
            model = train_scanvi(vae, level, cfg)
            mlflow.log_metric("train_seconds", time.time() - start)
            print(f"scANVI {level}: trained in {time.time() - start:.0f} s")

            for split, adata in evaluation.items():
                proba, _, mapped = predict_proba(model, adata, cfg, known_batches)
                labels, confidence = top_label(proba)
                result = evaluate(labels, adata.obs, level, split)
                log_result(result)
                mlflow.log_metric(f"{split}_mean_confidence", float(confidence.mean()))
                mlflow.log_param(f"{split}_reference_mapping", mapped)
                summary += [{"level": level, **row, "mapped": mapped} for row in result["rows"]]

            model_dir = out_dir / f"scanvi_{level}"
            model.save(str(model_dir), overwrite=True, save_anndata=False)
            mlflow.log_artifacts(str(model_dir), artifact_path="model")

    print("\n", pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
