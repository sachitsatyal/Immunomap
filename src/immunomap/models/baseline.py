"""Baseline classifier: logistic regression on log-normalized expression (CellTypist-style).

Every later model has to beat this. It is fast, well understood, and strong on
immune cells, so it is also the fallback the product can always ship.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LEVELS = ("coarse", "fine")


@dataclass(frozen=True)
class BaselineConfig:
    alpha: float  # L2 regularization strength
    max_iter: int  # passes over the training data
    tol: float
    class_weight: str | None  # "balanced" up-weights rare cell types
    n_jobs: int  # CPU cores (shared machine: keep this modest)
    seed: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls(**yaml.safe_load(Path(path).read_text()))


def build_pipeline(cfg: BaselineConfig) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler(with_mean=False)),  # keeps sparse input sparse
            (
                "classify",
                SGDClassifier(
                    loss="log_loss",
                    alpha=cfg.alpha,
                    max_iter=cfg.max_iter,
                    tol=cfg.tol,
                    class_weight=cfg.class_weight,
                    n_jobs=cfg.n_jobs,
                    random_state=cfg.seed,
                ),
            ),
        ]
    )


def labelled_cells(obs: pd.DataFrame, level: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions of cells that have a label at this level, and those labels.

    Studies label at different depths: a cell may have a coarse label but no fine one.
    Such cells are left out of training and evaluation at that level only.
    """
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level}")
    labels = obs[level]
    has_label = labels.notna().to_numpy()
    return np.flatnonzero(has_label), labels[has_label].astype(str).to_numpy()
