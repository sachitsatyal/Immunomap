"""Checks on the selected gene list. Skipped where it has not been generated."""

from pathlib import Path

import anndata as ad
import pandas as pd
import pytest
import yaml

HVG = Path("artifacts/hvg_genes.csv")
SHARD = Path("data/processed/qc/aida__test__00.h5ad")
pytestmark = pytest.mark.skipif(
    not (HVG.exists() and SHARD.exists()), reason="gene list or data not present"
)

MARKERS = [
    g
    for genes in yaml.safe_load(Path("configs/marker_genes.yaml").read_text()).values()
    for g in genes
]


@pytest.fixture(scope="module")
def hvg() -> pd.DataFrame:
    return pd.read_csv(HVG)


def test_gene_count_and_uniqueness(hvg: pd.DataFrame) -> None:
    assert (hvg["source"] == "hvg").sum() == 3000
    assert hvg["feature_id"].is_unique


def test_genes_exist_in_the_data(hvg: pd.DataFrame) -> None:
    adata = ad.read_h5ad(SHARD, backed="r")
    try:
        assert set(hvg["feature_id"]) <= set(adata.var_names)
    finally:
        adata.file.close()


def test_no_mitochondrial_genes(hvg: pd.DataFrame) -> None:
    assert not hvg["feature_name"].str.upper().str.startswith("MT-").any()


def test_all_marker_genes_included(hvg: pd.DataFrame) -> None:
    missing = set(MARKERS) - set(hvg["feature_name"])
    assert not missing, f"marker genes missing: {missing}"
