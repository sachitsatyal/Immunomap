"""Checks on the QC'd training data. Run locally after scripts/run_qc.py.

Skipped automatically where the data is absent (for example in CI).
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from immunomap.data.labels import LabelScheme
from immunomap.data.qc import QCConfig

QC_DIR = Path("data/processed/qc")
SHARDS = sorted(QC_DIR.glob("*.h5ad"))
pytestmark = pytest.mark.skipif(not SHARDS, reason="processed data not present")

SPLITS = {"train", "val", "test", "external"}
REQUIRED = ["group", "dataset_id", "donor_id", "split", "lineage"]
OBS_COLUMNS = [*REQUIRED, "coarse", "fine", "total_counts", "n_genes", "pct_mt"]


def _text(value: object) -> str | None:
    return None if pd.isna(value) else str(value)


@pytest.fixture(scope="module")
def obs() -> pd.DataFrame:
    frames = []
    for path in SHARDS:
        adata = ad.read_h5ad(path, backed="r")
        frames.append(adata.obs[OBS_COLUMNS].copy())
        adata.file.close()
    return pd.concat(frames)


@pytest.fixture(scope="module")
def reference_genes() -> pd.Index:
    adata = ad.read_h5ad(SHARDS[0], backed="r")
    genes = adata.var_names.copy()
    adata.file.close()
    return genes


def test_expected_number_of_cells(obs: pd.DataFrame) -> None:
    summary = pd.read_csv("docs/qc/qc_summary.csv")
    assert len(obs) == int((summary["cells"] - summary["removed"]).sum())


def test_cell_ids_are_unique(obs: pd.DataFrame) -> None:
    assert obs.index.is_unique


def test_required_metadata_present(obs: pd.DataFrame) -> None:
    assert obs[REQUIRED].notna().all().all()
    assert set(obs["split"].astype(str)) <= SPLITS


def test_no_donor_in_two_splits(obs: pd.DataFrame) -> None:
    splits_per_donor = obs.groupby(["group", "donor_id"], observed=True)["split"].nunique()
    assert (splits_per_donor == 1).all()


def test_external_studies_only_in_external_split(obs: pd.DataFrame) -> None:
    external = obs["group"].astype(str).str.startswith("ext_")
    assert (obs.loc[external, "split"].astype(str) == "external").all()
    assert (obs.loc[~external, "split"].astype(str) != "external").all()


def test_labels_follow_label_tree(obs: pd.DataFrame) -> None:
    scheme = LabelScheme.from_yaml("configs/labels.yaml")
    allowed = set()
    for ontology_id in scheme.source_map:
        resolved = scheme.resolve(ontology_id)
        if resolved is not None:
            allowed.add((resolved.lineage, resolved.coarse, resolved.fine))
    labels = obs[["lineage", "coarse", "fine"]].drop_duplicates()
    observed = {tuple(_text(v) for v in row) for row in labels.itertuples(index=False)}
    assert observed <= allowed, f"unexpected label combinations: {observed - allowed}"


def test_cells_meet_qc_rules(obs: pd.DataFrame) -> None:
    cfg = QCConfig.from_yaml("configs/qc.yaml")
    assert obs["n_genes"].min() >= cfg.min_genes
    assert obs["total_counts"].min() >= cfg.min_counts
    assert obs["pct_mt"].max() <= cfg.max_pct_mt


@pytest.mark.parametrize("path", SHARDS, ids=lambda p: p.stem)
def test_shard_has_raw_counts_and_same_genes(path: Path, reference_genes: pd.Index) -> None:
    adata = ad.read_h5ad(path, backed="r")
    try:
        assert adata.var_names.equals(reference_genes)
        values = adata.X[:500].data  # first 500 cells is enough to catch normalized data
        assert (values >= 0).all()
        assert np.allclose(values, np.round(values)), "counts must be raw integers"
    finally:
        adata.file.close()
