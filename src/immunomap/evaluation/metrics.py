"""Classification metrics used for every model, so results are comparable."""

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)


def classification_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Headline metrics. Macro-F1 averages over the classes present in y_true,
    so a test set that lacks a cell type is not penalized for it, while every
    cell type it does contain counts equally (rare types are not drowned out)."""
    labels = np.unique(y_true)
    with warnings.catch_warnings():
        # Expected by design: a model may predict a type this evaluation set does not contain
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        balanced = balanced_accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
        ),
    }


def per_class_report(y_true: Any, y_pred: Any) -> pd.DataFrame:
    """Precision, recall, F1 and support for each cell type present in y_true."""
    labels = np.unique(y_true)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    report = pd.DataFrame(
        {"precision": precision, "recall": recall, "f1": f1, "support": support},
        index=pd.Index(labels, name="label"),
    )
    return report.sort_values("f1")
