"""Checks on the model-input files. Skipped where they have not been built.

Reads only small slices with h5py: opening the multi-GB train file with anndata
would load its whole counts layer into memory.
"""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

try:
    from anndata.io import read_elem
except ImportError:  # older anndata
    from anndata.experimental import read_elem

FEATURES = Path("data/processed/features")
SPLITS = ["train", "val", "test", "external"]
N_CHECK = 200
pytestmark = pytest.mark.skipif(
    not all((FEATURES / f"{s}.h5ad").exists() for s in SPLITS), reason="model inputs not built"
)


def _first_rows(group: h5py.Group, n: int) -> np.ndarray:
    """First n rows of a CSR matrix stored in an .h5ad group, as a dense array."""
    indptr = group["indptr"][: n + 1]
    start, end = int(indptr[0]), int(indptr[-1])
    matrix = sp.csr_matrix(
        (group["data"][start:end], group["indices"][start:end], indptr - start),
        shape=(n, int(group.attrs["shape"][1])),
    )
    return matrix.toarray()


@pytest.fixture(scope="module")
def gene_ids() -> list[str]:
    return pd.read_csv("artifacts/hvg_genes.csv")["feature_id"].astype(str).tolist()


@pytest.mark.parametrize("split", SPLITS)
def test_split_file_is_consistent(split: str, gene_ids: list[str]) -> None:
    with h5py.File(FEATURES / f"{split}.h5ad", "r") as f:
        var = read_elem(f["var"])
        obs = read_elem(f["obs"])
        assert list(var.index) == gene_ids
        assert set(obs["split"].astype(str)) == {split}

        # X must equal log1p(counts / total_counts * 1e4) for every cell checked
        counts = _first_rows(f["layers"]["counts"], N_CHECK)
        values = _first_rows(f["X"], N_CHECK)
        assert np.allclose(counts, np.round(counts)), "counts layer must hold raw integers"
        total = obs["total_counts"].to_numpy()[:N_CHECK, None]
        assert np.allclose(values, np.log1p(counts / total * 1e4), atol=1e-4)


def test_splits_add_up_to_qc_output() -> None:
    summary = pd.read_csv("docs/qc/qc_summary.csv")
    expected = int((summary["cells"] - summary["removed"]).sum())
    total = 0
    for split in SPLITS:
        with h5py.File(FEATURES / f"{split}.h5ad", "r") as f:
            total += int(f["X"].attrs["shape"][0])
    assert total == expected
