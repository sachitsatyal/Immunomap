"""Annotate a user's blood scRNA-seq file: raw counts in, a cell-type table out.

Per cell:
1. QC with the same rules as the reference data (one batch per upload, or per sample
   column if given). Failed cells are reported, not silently dropped.
2. The model's 3,002 genes are selected (by Ensembl ID, or by gene symbol if the file
   has no Ensembl IDs); missing genes count as zero and are reported.
3. scANVI maps the upload onto the reference (scArches; only the new dataset's
   parameters are trained, from its expression alone) and gives class probabilities
   at the coarse and fine level, plus each cell's position in the reference space.
4. Calibrated confidence; Unknown when confidence is below the threshold or the cell is
   far from every reference cell; a fine label must agree with the coarse label.

Everything the annotator needs is in one "bundle" folder (scripts/export_bundle.py).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import anndata as ad
import joblib  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import scipy.sparse as sp

from immunomap.data.features import load_gene_list, normalize_counts
from immunomap.data.labels import LabelScheme
from immunomap.data.qc import PASS, QCConfig, compute_metrics, compute_thresholds, flag_cells
from immunomap.models.calibration import UNKNOWN, Calibration
from immunomap.models.scanvi import ScanviConfig, add_label_column, log_scores

LEVELS = ("coarse", "fine")
UPLOAD_BATCH = "upload"
QC_FAILED = "QC failed"


@dataclass(frozen=True)
class Calls:
    """Calibrated per-cell calls for one level (before coarse/fine reconciliation)."""

    labels: Any  # cell type, or UNKNOWN when confidence is below the threshold
    confidence: Any


def harmonize_genes(adata: ad.AnnData, genes: pd.DataFrame) -> tuple[ad.AnnData, Any]:
    """Return (adata with Ensembl IDs as var_names where possible, gene symbols).

    Files from CELLxGENE use Ensembl IDs; many others use symbols. Symbols are needed
    for the mitochondrial QC rule (MT- prefix).
    """
    names = np.asarray(adata.var_names).astype(str)
    if "feature_name" in adata.var:
        symbols = adata.var["feature_name"].astype(str).to_numpy()
    else:
        symbols = names
    ids = set(genes["feature_id"].astype(str))
    if np.isin(names, list(ids)).sum() >= np.isin(names, genes["feature_name"]).sum():
        return adata, symbols
    to_id = dict(
        zip(genes["feature_name"].astype(str), genes["feature_id"].astype(str), strict=True)
    )
    renamed = adata.copy()
    renamed.var_names = pd.Index([to_id.get(n, n) for n in names]).astype(str)
    renamed.var_names_make_unique()
    return renamed, names


def run_qc(adata: ad.AnnData, symbols: Any, cfg: QCConfig, batch_key: str | None) -> Any:
    """QC reason per cell ('pass' or the first failed rule)."""
    metrics = compute_metrics(sp.csr_matrix(adata.X), list(symbols))
    batches = (
        pd.Series(np.asarray(adata.obs[batch_key]).astype(str))
        if batch_key
        else pd.Series(["all"] * adata.n_obs)
    )
    thresholds = compute_thresholds(metrics, batches, cfg)
    return flag_cells(metrics, batches, thresholds, cfg).to_numpy()


def calibrated_calls(proba: pd.DataFrame, calibration: Calibration) -> Calls:
    if list(proba.columns) != list(calibration.classes):
        raise ValueError("model classes do not match the calibration file")
    labels, confidence = calibration.predict(log_scores(proba))
    return Calls(labels=np.asarray(labels, dtype=object), confidence=np.asarray(confidence))


def combine_calls(
    coarse: Calls,
    fine: Calls,
    novel: Any,
    qc_reason: Any,
    labels: LabelScheme,
) -> pd.DataFrame:
    """Final per-cell table. status explains every Unknown."""
    n = len(qc_reason)
    passed = np.asarray(qc_reason) == PASS
    novel = np.asarray(novel, dtype=bool)
    coarse_label = np.asarray(coarse.labels, dtype=object).copy()
    fine_label = np.asarray(fine.labels, dtype=object).copy()
    status = np.full(n, "ok", dtype=object)

    parent = np.array([labels.fine_parent.get(f, f) for f in fine_label], dtype=object)
    inconsistent = (coarse_label != UNKNOWN) & (fine_label != UNKNOWN) & (parent != coarse_label)
    fine_label[inconsistent] = UNKNOWN
    status[inconsistent] = "fine_disagrees_with_coarse"
    low_fine = (fine.labels == UNKNOWN) & (coarse_label != UNKNOWN)
    status[low_fine] = "low_confidence_fine"
    low_coarse = coarse_label == UNKNOWN
    fine_label[low_coarse] = UNKNOWN
    status[low_coarse] = "low_confidence"
    coarse_label[novel] = UNKNOWN
    fine_label[novel] = UNKNOWN
    status[novel] = "unlike_reference"
    for column in (coarse_label, fine_label):
        column[~passed] = QC_FAILED
    status[~passed] = "qc_" + np.asarray(qc_reason, dtype=object)[~passed].astype(str)

    lineage = np.array(
        [
            labels.coarse_parent.get(c, c) if c not in (UNKNOWN, QC_FAILED) else c
            for c in coarse_label
        ],
        dtype=object,
    )
    return pd.DataFrame(
        {
            "lineage": lineage,
            "coarse": coarse_label,
            "coarse_confidence": np.where(passed, coarse.confidence, np.nan),
            "fine": fine_label,
            "fine_confidence": np.where(passed, fine.confidence, np.nan),
            "status": status,
        }
    )


class Annotator:
    """Loads a bundle once; annotate() can then be called for any number of files."""

    def __init__(self, bundle: str | Path) -> None:
        self.bundle = Path(bundle)
        self.genes = load_gene_list(self.bundle / "genes.csv")
        self.labels = LabelScheme.from_yaml(self.bundle / "labels.yaml")
        self.qc = QCConfig.from_yaml(self.bundle / "qc.yaml")
        self.scanvi = ScanviConfig.from_yaml(self.bundle / "scanvi.yaml")
        self.calibration = {
            level: Calibration(
                **json.loads((self.bundle / f"calibration_{level}.json").read_text())
            )
            for level in LEVELS
        }
        self.novelty = joblib.load(self.bundle / "novelty.joblib")
        self.manifest = json.loads((self.bundle / "manifest.json").read_text())

    @classmethod
    def from_bundle(cls, bundle: str | Path) -> Self:
        return cls(bundle)

    def model_counts(self, adata: ad.AnnData) -> tuple[ad.AnnData, int]:
        """Raw counts of the model genes (missing genes = zero), and how many were missing."""
        gene_ids = self.genes["feature_id"].astype(str).tolist()
        counts, _, n_missing = normalize_counts(adata.X, list(adata.var_names), gene_ids)
        obs = pd.DataFrame(index=adata.obs_names.astype(str))
        obs[self.scanvi.batch_key] = pd.Categorical([UPLOAD_BATCH] * adata.n_obs)
        return ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=gene_ids)), n_missing

    def map_and_predict(self, query: ad.AnnData, level: str, epochs: int) -> tuple[Any, Any]:
        """scArches mapping onto the saved reference model; (probabilities, latent)."""
        import scvi

        model_dir = str(self.bundle / f"scanvi_{level}")
        query = query.copy()
        add_label_column(query, None)  # the upload's labels (if any) are never used
        scvi.model.SCANVI.prepare_query_anndata(query, model_dir)
        model = scvi.model.SCANVI.load_query_data(query, model_dir)
        model.train(
            max_epochs=epochs,
            batch_size=self.scanvi.batch_size,
            plan_kwargs={"weight_decay": 0.0},
            accelerator="auto",
        )
        proba = pd.DataFrame(model.predict(query, soft=True, batch_size=self.scanvi.batch_size))
        latent = np.asarray(
            model.get_latent_representation(query, batch_size=self.scanvi.batch_size)
        )
        return proba[sorted(proba.columns)], latent

    def annotate(
        self, adata: ad.AnnData, batch_key: str | None = None, epochs: int | None = None
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Per-cell table (index = the file's cell IDs) and a run summary."""
        adata, symbols = harmonize_genes(adata, self.genes)
        qc_reason = run_qc(adata, symbols, self.qc, batch_key)
        passed = qc_reason == PASS
        query, n_missing = self.model_counts(adata[passed])
        epochs = epochs or self.scanvi.query_epochs

        n = adata.n_obs
        calls, latent = {}, None
        for level in LEVELS:
            labels = np.full(n, QC_FAILED, dtype=object)
            confidence = np.full(n, np.nan)
            if passed.any():
                proba, z = self.map_and_predict(query, level, epochs)
                level_calls = calibrated_calls(proba, self.calibration[level])
                labels[passed], confidence[passed] = level_calls.labels, level_calls.confidence
                if level == "coarse":
                    latent = z
            calls[level] = Calls(labels, confidence)

        distance = np.full(n, np.nan)
        novel = np.zeros(n, dtype=bool)
        if latent is not None:
            distance[passed] = self.novelty.score(latent)
            novel[passed] = self.novelty.is_novel(distance[passed])
        table = combine_calls(calls["coarse"], calls["fine"], novel, qc_reason, self.labels)
        table.insert(table.columns.get_loc("status"), "distance_to_reference", distance)
        table.index = adata.obs_names.astype(str)
        summary = {
            "cells": int(n),
            "passed_qc": int(passed.sum()),
            "model_genes_missing": int(n_missing),
            "model_genes": len(self.genes),
            "unknown_coarse": int((table["coarse"] == UNKNOWN).sum()),
            "model": self.manifest.get("name", "unknown"),
        }
        return table, summary
