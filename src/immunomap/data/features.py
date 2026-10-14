"""Turn raw counts into model inputs: the model's gene set, normalized and log-transformed.

The same function runs on reference data and on user uploads, so the model always
sees identically prepared input.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

TARGET_SUM = 1e4  # counts per cell after normalization ("counts per 10k")


def load_gene_list(path: str | Path) -> pd.DataFrame:
    """The model's genes, in model order (columns: feature_id, feature_name, source)."""
    return pd.read_csv(path)


def normalize_counts(
    X: Any, var_names: Sequence[str], gene_ids: Sequence[str], target_sum: float = TARGET_SUM
) -> tuple[Any, Any, int]:
    """Select gene_ids (in that order) and normalize.

    Library size is computed over ALL genes of the input, before selection, so a
    cell's values do not depend on which genes the model happens to use.
    Genes missing from the input become all-zero columns.

    Returns (raw counts of the selected genes, log1p-normalized values, number missing).
    """
    X = sp.csr_matrix(X, dtype=np.float32)
    total = np.asarray(X.sum(axis=1), dtype=np.float64).ravel()

    position = {gene: i for i, gene in enumerate(var_names)}
    source_cols = [position.get(gene, -1) for gene in gene_ids]
    present = np.array([c >= 0 for c in source_cols])
    target_cols = np.flatnonzero(present)
    placement = sp.csr_matrix(
        (np.ones(len(target_cols), dtype=np.float32), (np.arange(len(target_cols)), target_cols)),
        shape=(len(target_cols), len(gene_ids)),
    )
    counts = (X[:, [c for c in source_cols if c >= 0]] @ placement).tocsr()

    scale = np.divide(target_sum, total, out=np.zeros_like(total), where=total > 0)
    normalized = (sp.diags(scale.astype(np.float32)) @ counts).tocsr()
    normalized.data = np.log1p(normalized.data)
    return counts, normalized, int((~present).sum())


def build_model_input(adata: Any, genes: pd.DataFrame) -> Any:
    """AnnData with X = log1p-normalized values, layers['counts'] = raw counts."""
    gene_ids = genes["feature_id"].astype(str).tolist()
    counts, normalized, n_missing = normalize_counts(adata.X, list(adata.var_names), gene_ids)
    var = pd.DataFrame(
        {"feature_name": genes["feature_name"].astype(str).to_numpy()},
        index=pd.Index(gene_ids, name=None),
    )
    out = ad.AnnData(X=normalized, obs=adata.obs.copy(), var=var, layers={"counts": counts})
    out.uns["n_missing_genes"] = n_missing
    return out
