"""Score predictions on one split and log them to MLflow, identically for every model.

External studies label at different depths (Lupus has no Treg or MAIT label, Hao does).
Pooling them would score a correct "Treg" call on a Lupus cell as a false positive for
Hao's Treg class, so external studies are only ever scored one study at a time.
"""

import tempfile
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from immunomap.evaluation.metrics import classification_metrics, per_class_report
from immunomap.models.baseline import labelled_cells

POOLED_SPLITS = {"val", "test"}


def evaluate(predicted: Any, obs: pd.DataFrame, level: str, split: str) -> dict[str, Any]:
    """Score one split (pooled for val/test) and each study group in it.

    `predicted` holds one predicted label per cell in `obs` (same order). Only cells
    labelled at this level are scored. Returns MLflow metrics, per-class reports and
    summary rows.
    """
    result: dict[str, Any] = {"metrics": {}, "reports": {}, "rows": []}
    idx, y_true = labelled_cells(obs, level)
    if len(idx) == 0:
        return result
    y_pred = np.asarray(predicted).astype(str)[idx]
    groups = obs["group"].astype(str).to_numpy()[idx]

    scopes = [(split, np.ones(len(idx), dtype=bool))] if split in POOLED_SPLITS else []
    scopes += [(f"{split}_{g}", groups == g) for g in sorted(set(groups))]
    for name, mask in scopes:
        scores = classification_metrics(y_true[mask], y_pred[mask])
        result["metrics"].update({f"{name}_{k}": v for k, v in scores.items()})
        result["metrics"][f"{name}_n_cells"] = float(mask.sum())
        if name == split or split not in POOLED_SPLITS:
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


def log_result(result: dict[str, Any]) -> None:
    """Log metrics and per-class CSVs (under reports/) to the active MLflow run."""
    mlflow.log_metrics(result["metrics"])
    with tempfile.TemporaryDirectory() as tmp:
        for name, report in result["reports"].items():
            path = Path(tmp) / f"per_class_{name}.csv"
            report.to_csv(path, lineterminator="\n")
            mlflow.log_artifact(str(path), artifact_path="reports")
