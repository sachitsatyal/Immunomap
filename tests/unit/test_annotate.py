"""The annotation pipeline, with scANVI replaced by a stub (so it runs in CI, no GPU)."""

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import yaml
from anndata import AnnData

from immunomap.data.labels import LabelScheme
from immunomap.inference.annotate import (
    QC_FAILED,
    Annotator,
    Calls,
    combine_calls,
    ensure_raw_counts,
    harmonize_genes,
)
from immunomap.models.calibration import UNKNOWN
from immunomap.models.novelty import NoveltyConfig, ReferenceDistance

GENES = pd.DataFrame(
    {
        "feature_id": ["ENSG1", "ENSG2", "ENSG3", "ENSG4"],
        "feature_name": ["CD3E", "MS4A1", "LYZ", "NKG7"],
        "source": ["hvg"] * 4,
    }
)
LABELS = {
    "lineage": ["T cell", "B cell"],
    "coarse": {"CD4 T": "T cell", "B": "B cell"},
    "fine": {"CD4 naive T": "CD4 T", "B naive": "B"},
    "source_map": {},
    "drop": [],
}


def _scheme() -> LabelScheme:
    return LabelScheme(LABELS)


def test_gene_symbols_are_mapped_to_ensembl_ids() -> None:
    adata = AnnData(
        X=sp.csr_matrix(np.ones((2, 3), dtype=np.float32)),
        var=pd.DataFrame(index=["CD3E", "LYZ", "MT-CO1"]),
    )
    renamed, symbols = harmonize_genes(adata, GENES)
    assert list(renamed.var_names) == ["ENSG1", "ENSG3", "MT-CO1"]
    assert list(symbols) == ["CD3E", "LYZ", "MT-CO1"]  # kept for the MT QC rule


def test_ensembl_files_are_left_unchanged() -> None:
    adata = AnnData(
        X=sp.csr_matrix(np.ones((1, 2), dtype=np.float32)),
        var=pd.DataFrame({"feature_name": ["CD3E", "LYZ"]}, index=["ENSG1", "ENSG3"]),
    )
    renamed, symbols = harmonize_genes(adata, GENES)
    assert renamed is adata
    assert list(symbols) == ["CD3E", "LYZ"]


def test_combine_calls_explains_every_unknown() -> None:
    coarse = Calls(np.array(["CD4 T", "CD4 T", UNKNOWN, "B", "B"], dtype=object), np.ones(5))
    fine = Calls(
        np.array(["CD4 naive T", UNKNOWN, "CD4 naive T", "CD4 naive T", "B naive"], dtype=object),
        np.ones(5),
    )
    novel = np.array([False, False, False, False, True])
    qc = np.array(["pass"] * 5, dtype=object)
    table = combine_calls(coarse, fine, novel, qc, _scheme())
    assert table["status"].tolist() == [
        "ok",
        "low_confidence_fine",
        "low_confidence",
        "fine_disagrees_with_coarse",
        "unlike_reference",
    ]
    assert table["fine"].tolist() == ["CD4 naive T", UNKNOWN, UNKNOWN, UNKNOWN, UNKNOWN]
    assert table["coarse"].tolist() == ["CD4 T", "CD4 T", UNKNOWN, "B", UNKNOWN]
    assert table["lineage"].tolist() == ["T cell", "T cell", UNKNOWN, "B cell", UNKNOWN]


def test_failed_qc_cells_are_reported_not_dropped() -> None:
    calls = Calls(np.array(["B", QC_FAILED], dtype=object), np.array([0.9, np.nan]))
    fine = Calls(np.array(["B naive", QC_FAILED], dtype=object), np.array([0.9, np.nan]))
    qc = np.array(["pass", "too_few_genes"], dtype=object)
    table = combine_calls(calls, fine, np.array([False, False]), qc, _scheme())
    assert table["coarse"].tolist() == ["B", QC_FAILED]
    assert table["status"].tolist() == ["ok", "qc_too_few_genes"]
    assert np.isnan(table["coarse_confidence"].iloc[1])


class StubAnnotator(Annotator):
    """scANVI replaced by a rule: CD3E-high cells are CD4 T, MS4A1-high cells are B."""

    def map_and_predict(self, query: Any, level: str, epochs: int) -> tuple[Any, Any]:
        X = np.asarray(sp.csr_matrix(query.X).todense())
        is_t = X[:, 0] > X[:, 1]
        classes = ["CD4 T", "B"] if level == "coarse" else ["CD4 naive T", "B naive"]
        p_t = np.where(is_t, 0.97, 0.03)
        proba = pd.DataFrame({classes[0]: p_t, classes[1]: 1 - p_t})
        latent = np.log1p(X[:, :2])  # the novelty detector reads this 2-d "latent"
        return proba[sorted(proba.columns)], latent


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    GENES.to_csv(tmp_path / "genes.csv", index=False)
    (tmp_path / "labels.yaml").write_text(yaml.safe_dump(LABELS))
    qc = {
        "batch_key": "donor_id",
        "min_genes": 1,
        "min_counts": 5,
        "max_pct_mt": 50,
        "n_mads": 50,
        "n_mads_mt": 50,
        "remove_high_outliers": False,
    }
    (tmp_path / "qc.yaml").write_text(yaml.safe_dump(qc))
    shutil.copy("configs/scanvi.yaml", tmp_path / "scanvi.yaml")
    for level, classes in {"coarse": ["B", "CD4 T"], "fine": ["B naive", "CD4 naive T"]}.items():
        calibration = {"classes": classes, "temperature": 1.0, "threshold": 0.5}
        (tmp_path / f"calibration_{level}.json").write_text(json.dumps(calibration))
    rng = np.random.default_rng(0)
    reference = np.vstack([rng.normal([3, 0], 0.2, (100, 2)), rng.normal([0, 3], 0.2, (100, 2))])
    cfg = NoveltyConfig(
        n_components=2, n_neighbors=5, ref_per_type=100, quantile=0.99, n_jobs=1, seed=0
    )
    detector = ReferenceDistance(cfg, embed=False).fit(reference)
    detector.set_threshold(detector.score(reference))
    joblib.dump(detector, tmp_path / "novelty.joblib")
    (tmp_path / "manifest.json").write_text(json.dumps({"name": "test-bundle"}))
    return tmp_path


def test_annotator_end_to_end(bundle: Path) -> None:
    X = np.array(
        [
            [20, 0, 0, 0, 1],  # T cell
            [0, 20, 0, 0, 1],  # B cell
            [0, 0, 0, 0, 0],  # empty droplet: fails QC
            [400, 400, 0, 0, 1],  # far from every reference cell
        ],
        dtype=np.float32,
    )
    adata = AnnData(
        X=sp.csr_matrix(X),
        obs=pd.DataFrame(index=["c1", "c2", "c3", "c4"]),
        var=pd.DataFrame(index=["ENSG1", "ENSG2", "ENSG3", "ENSG4", "ENSG_OTHER"]),
    )
    table, summary = StubAnnotator.from_bundle(bundle).annotate(adata)
    assert list(table.index) == ["c1", "c2", "c3", "c4"]
    assert table["coarse"].tolist() == ["CD4 T", "B", QC_FAILED, UNKNOWN]
    assert table["fine"].tolist() == ["CD4 naive T", "B naive", QC_FAILED, UNKNOWN]
    assert table["status"].tolist()[3] == "unlike_reference"
    assert summary["passed_qc"] == 3 and summary["model_genes_missing"] == 0
    assert summary["model"] == "test-bundle"


def test_raw_counts_are_found_in_x_layers_or_raw() -> None:
    counts = sp.csr_matrix(np.array([[3, 0], [1, 2]], dtype=np.float32))
    normalized = sp.csr_matrix(np.log1p(counts.toarray() / 3))
    plain = AnnData(X=counts)
    assert ensure_raw_counts(plain) is plain
    layered = AnnData(X=normalized, layers={"counts": counts})
    assert (counts != ensure_raw_counts(layered).X).nnz == 0
    with pytest.raises(ValueError):
        ensure_raw_counts(AnnData(X=normalized))
