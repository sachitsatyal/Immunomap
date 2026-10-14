import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from immunomap.data.features import build_model_input, normalize_counts

VAR = ["MT-CO1", "GENE_A", "GENE_B", "GENE_C"]


def test_selects_genes_in_requested_order() -> None:
    X = sp.csr_matrix(np.array([[0, 1, 2, 3]], dtype=np.float32))
    counts, _, _ = normalize_counts(X, VAR, ["GENE_C", "GENE_A"])
    assert counts.toarray().tolist() == [[3.0, 1.0]]


def test_library_size_uses_all_genes_not_just_selected() -> None:
    X = sp.csr_matrix(np.array([[50, 25, 25, 0]], dtype=np.float32))  # total = 100
    _, normalized, _ = normalize_counts(X, VAR, ["GENE_A"])
    assert normalized.toarray()[0, 0] == pytest.approx(np.log1p(25 / 100 * 1e4))


def test_missing_genes_become_zero_columns() -> None:
    X = sp.csr_matrix(np.array([[1, 2, 3, 4]], dtype=np.float32))
    counts, normalized, n_missing = normalize_counts(X, VAR, ["GENE_A", "NOT_IN_DATA"])
    assert n_missing == 1
    assert counts.toarray().tolist() == [[2.0, 0.0]]
    assert normalized.toarray()[0, 1] == 0.0


def test_empty_cell_gives_zeros_not_nan() -> None:
    X = sp.csr_matrix((1, 4), dtype=np.float32)
    _, normalized, _ = normalize_counts(X, VAR, ["GENE_A", "GENE_B"])
    assert np.isfinite(normalized.toarray()).all()
    assert normalized.sum() == 0


def test_dense_input_is_accepted() -> None:
    counts, _, _ = normalize_counts(np.array([[0, 5, 0, 0]]), VAR, ["GENE_A"])
    assert counts.toarray().tolist() == [[5.0]]


def test_build_model_input_keeps_obs_and_raw_counts() -> None:
    import anndata as ad

    adata = ad.AnnData(
        X=sp.csr_matrix(np.array([[10, 30, 60, 0], [0, 0, 5, 5]], dtype=np.float32)),
        obs=pd.DataFrame({"split": ["train", "val"]}, index=["c1", "c2"]),
        var=pd.DataFrame(index=VAR),
    )
    genes = pd.DataFrame({"feature_id": ["GENE_B", "GENE_A"], "feature_name": ["B", "A"]})
    out = build_model_input(adata, genes)
    assert list(out.var_names) == ["GENE_B", "GENE_A"]
    assert out.obs["split"].tolist() == ["train", "val"]
    assert out.layers["counts"].toarray().tolist() == [[60.0, 30.0], [5.0, 0.0]]
    assert out.uns["n_missing_genes"] == 0
