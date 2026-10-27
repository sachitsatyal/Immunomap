"""Checks on the scANVI helpers that do not need scvi-tools (so they also run in CI)."""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData

from immunomap.models.scanvi import (
    LABEL_COLUMN,
    UNLABELED,
    ScanviConfig,
    add_label_column,
    clean_batches,
    needs_reference_mapping,
    top_label,
)


def _adata() -> AnnData:
    obs = pd.DataFrame(
        {
            "dataset_id": pd.Categorical(["d1", "d1", "d2"], categories=["d1", "d2", "d9"]),
            "coarse": ["NK", "B", "NK"],
            "fine": ["NK CD56dim", None, None],
        },
        index=["c1", "c2", "c3"],
    )
    return AnnData(X=sp.csr_matrix(np.ones((3, 4), dtype=np.float32)), obs=obs)


def test_cells_without_a_label_at_that_level_become_unlabeled() -> None:
    adata = _adata()
    add_label_column(adata, "fine")
    assert adata.obs[LABEL_COLUMN].tolist() == ["NK CD56dim", UNLABELED, UNLABELED]
    add_label_column(adata, "coarse")
    assert adata.obs[LABEL_COLUMN].tolist() == ["NK", "B", "NK"]


def test_prediction_data_never_carries_labels() -> None:
    adata = _adata()
    add_label_column(adata, None)
    assert set(adata.obs[LABEL_COLUMN]) == {UNLABELED}


def test_unused_batch_categories_are_dropped() -> None:
    adata = _adata()
    clean_batches(adata, "dataset_id")
    assert list(adata.obs["dataset_id"].cat.categories) == ["d1", "d2"]


def test_reference_mapping_only_for_unseen_batches() -> None:
    known = {"d1", "d2"}
    assert not needs_reference_mapping(["d1", "d2", "d1"], known)
    assert needs_reference_mapping(["d1", "hao"], known)


def test_top_label_picks_the_most_likely_type() -> None:
    proba = pd.DataFrame([[0.2, 0.8], [0.9, 0.1]], columns=["B", "NK"])
    labels, confidence = top_label(proba)
    assert labels.tolist() == ["NK", "B"]
    assert np.allclose(confidence, [0.8, 0.9])


def test_config_loads_from_repo_yaml() -> None:
    cfg = ScanviConfig.from_yaml("configs/scanvi.yaml")
    assert cfg.batch_key == "dataset_id"
    assert cfg.gene_likelihood in {"nb", "zinb"}
    assert cfg.batch_size > 0
    assert cfg.encode_covariates  # needed for reference mapping to adapt to new datasets
