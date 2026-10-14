"""Cell quality control: per-cell metrics, per-batch adaptive thresholds, pass/fail flags.

The same functions run on the reference data (training) and on files users upload,
so both are filtered by exactly the same rules.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import yaml

PASS = "pass"
# Order matters: when a cell fails several rules it gets the first matching reason.
REASONS = [
    "too_few_genes",
    "too_few_counts",
    "high_mt",
    "counts_outlier",
    "genes_outlier",
    "mt_outlier",
]


@dataclass(frozen=True)
class QCConfig:
    batch_key: str  # thresholds are computed separately within each batch
    min_genes: int  # hard floor
    min_counts: int  # hard floor
    max_pct_mt: float  # hard ceiling, percent
    n_mads: float  # outlier width for log counts and log genes
    n_mads_mt: float  # outlier width for percent mitochondrial (upper side only)
    remove_high_outliers: bool  # False: only unusually LOW counts/genes are removed

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls(**yaml.safe_load(Path(path).read_text()))


def compute_metrics(X: Any, gene_names: Sequence[str]) -> pd.DataFrame:
    """Per-cell total counts, detected genes and percent mitochondrial counts."""
    is_mt = np.array([str(g).upper().startswith("MT-") for g in gene_names])
    total = np.asarray(X.sum(axis=1), dtype=float).ravel()
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    mt_counts = np.asarray(X[:, is_mt].sum(axis=1), dtype=float).ravel()
    pct_mt = np.divide(100 * mt_counts, total, out=np.zeros_like(total), where=total > 0)
    return pd.DataFrame({"total_counts": total, "n_genes": n_genes, "pct_mt": pct_mt})


def _mad_bounds(values: np.ndarray, n_mads: float) -> tuple[float, float]:
    """Median +/- n_mads * MAD; no bound at all when the MAD is zero."""
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad == 0:
        return -np.inf, np.inf
    return median - n_mads * mad, median + n_mads * mad


def compute_thresholds(metrics: pd.DataFrame, batches: pd.Series, cfg: QCConfig) -> pd.DataFrame:
    """Adaptive bounds per batch, on log1p counts, log1p genes and percent MT."""
    rows = {}
    for batch, idx in metrics.groupby(batches.to_numpy(), observed=True).indices.items():
        m = metrics.iloc[idx]
        counts_lo, counts_hi = _mad_bounds(np.log1p(m["total_counts"].to_numpy()), cfg.n_mads)
        genes_lo, genes_hi = _mad_bounds(np.log1p(m["n_genes"].to_numpy()), cfg.n_mads)
        _, mt_hi = _mad_bounds(m["pct_mt"].to_numpy(), cfg.n_mads_mt)
        rows[batch] = {
            "log_counts_lo": counts_lo,
            "log_counts_hi": counts_hi,
            "log_genes_lo": genes_lo,
            "log_genes_hi": genes_hi,
            "pct_mt_hi": mt_hi,
            "cells": len(idx),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def flag_cells(
    metrics: pd.DataFrame, batches: pd.Series, thresholds: pd.DataFrame, cfg: QCConfig
) -> pd.Series:
    """Return one reason per cell: 'pass' or the first rule it fails."""
    t = thresholds.loc[batches.to_numpy()]
    log_counts = np.log1p(metrics["total_counts"].to_numpy())
    log_genes = np.log1p(metrics["n_genes"].to_numpy())
    pct_mt = metrics["pct_mt"].to_numpy()
    # High counts/genes are often real biology (plasmablasts, RNA-rich cells), not low quality
    high = cfg.remove_high_outliers

    masks = {
        "too_few_genes": metrics["n_genes"].to_numpy() < cfg.min_genes,
        "too_few_counts": metrics["total_counts"].to_numpy() < cfg.min_counts,
        "high_mt": pct_mt > cfg.max_pct_mt,
        "counts_outlier": (log_counts < t["log_counts_lo"].to_numpy())
        | (high & (log_counts > t["log_counts_hi"].to_numpy())),
        "genes_outlier": (log_genes < t["log_genes_lo"].to_numpy())
        | (high & (log_genes > t["log_genes_hi"].to_numpy())),
        "mt_outlier": pct_mt > t["pct_mt_hi"].to_numpy(),
    }
    reason = np.full(len(metrics), PASS, dtype=object)
    for name in reversed(REASONS):  # earlier rules overwrite later ones
        reason[masks[name]] = name
    return pd.Series(
        pd.Categorical(reason, categories=[PASS, *REASONS]), index=metrics.index, name="qc_reason"
    )
