"""scANVI: a deep generative model of raw counts that removes batch effects and labels cells.

Two stages, as recommended by scvi-tools:
1. scVI learns a batch-corrected latent space from raw counts, without labels.
2. scANVI adds a classifier on top of it, one model per label level. Cells that have no
   label at that level are still used (semi-supervised); the baseline has to drop them.

Cells from batches the model has never seen (the external studies now, user uploads in
the product) are handled by reference mapping (scArches): only the parameters for the new
batch are trained, on its expression alone. Its labels are never used.

scvi-tools lives in the `train` dependency group (GPU machine only), so it is imported
inside the functions that need it; the rest of this module is tested in CI.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, cast

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import yaml
from anndata.io import read_elem, sparse_dataset

UNLABELED = "unlabeled"
LABEL_COLUMN = "scanvi_label"


@dataclass(frozen=True)
class ScanviConfig:
    batch_key: str  # obs column treated as the batch to correct for
    n_latent: int  # size of the learned cell representation
    n_layers: int
    n_hidden: int
    gene_likelihood: str  # "nb" (negative binomial) or "zinb"
    # Reference-mapping (scArches) architecture: the encoder sees the batch, so mapping a
    # new dataset can adapt how its cells are embedded; layer norm instead of batch norm.
    encode_covariates: bool
    use_layer_norm: str
    use_batch_norm: str
    dropout_rate: float
    scvi_epochs: int  # stage 1: unsupervised scVI
    scanvi_epochs: int  # stage 2: scANVI classifier, per label level
    query_epochs: int  # reference mapping of unseen batches
    batch_size: int
    seed: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls(**yaml.safe_load(Path(path).read_text()))


def n_cells(path: str | Path) -> int:
    with h5py.File(path, "r") as f:
        return int(f["layers"]["counts"].attrs["shape"][0])


def load_counts(path: str | Path, rows: Any = None) -> ad.AnnData:
    """Raw counts (as X), cell metadata and genes of a model-input file.

    `rows` (optional, positions) reads only those cells, for quick smoke runs.
    """
    with h5py.File(path, "r") as f:
        obs = cast(pd.DataFrame, read_elem(f["obs"]))
        var = read_elem(f["var"])
        counts = sparse_dataset(f["layers"]["counts"])
        if rows is None:
            X = counts.to_memory()
        else:
            rows = np.sort(np.asarray(rows))
            X = counts[rows]
            obs = obs.iloc[rows]
    return ad.AnnData(X=X, obs=obs, var=var)


def clean_batches(adata: ad.AnnData, batch_key: str) -> None:
    """Batch column as a categorical of the values present only.

    Unused categories would be registered as batches the model has 'seen'.
    """
    adata.obs[batch_key] = pd.Categorical(adata.obs[batch_key].astype(str))


def add_label_column(adata: ad.AnnData, level: str | None) -> None:
    """Training labels for one level; cells without a label there become 'unlabeled'.

    level=None marks every cell unlabeled: used for prediction, so labels of evaluation
    data can never leak into the model.
    """
    if level is None:
        values = np.full(adata.n_obs, UNLABELED, dtype=object)
    else:
        labels = adata.obs[level].astype(object)
        values = labels.where(labels.notna(), UNLABELED).astype(str).to_numpy()
    adata.obs[LABEL_COLUMN] = pd.Categorical(values)


def needs_reference_mapping(batches: Any, known_batches: set[str]) -> bool:
    """True if any cell comes from a batch the model was not trained on."""
    return not set(np.asarray(batches).astype(str)) <= known_batches


def train_scvi(adata: ad.AnnData, cfg: ScanviConfig) -> Any:
    import scvi

    scvi.settings.seed = cfg.seed
    clean_batches(adata, cfg.batch_key)
    scvi.model.SCVI.setup_anndata(adata, batch_key=cfg.batch_key)
    model = scvi.model.SCVI(
        adata,
        n_latent=cfg.n_latent,
        n_layers=cfg.n_layers,
        n_hidden=cfg.n_hidden,
        gene_likelihood=cfg.gene_likelihood,
        encode_covariates=cfg.encode_covariates,
        use_layer_norm=cfg.use_layer_norm,
        use_batch_norm=cfg.use_batch_norm,
        dropout_rate=cfg.dropout_rate,
    )
    model.train(max_epochs=cfg.scvi_epochs, batch_size=cfg.batch_size, accelerator="auto")
    return model


def train_scanvi(scvi_model: Any, level: str, cfg: ScanviConfig) -> Any:
    """scANVI for one label level, starting from the trained scVI model."""
    import scvi

    add_label_column(scvi_model.adata, level)
    model = scvi.model.SCANVI.from_scvi_model(
        scvi_model, unlabeled_category=UNLABELED, labels_key=LABEL_COLUMN
    )
    model.train(max_epochs=cfg.scanvi_epochs, batch_size=cfg.batch_size, accelerator="auto")
    return model


def predict_proba(
    model: Any, adata: ad.AnnData, cfg: ScanviConfig, known_batches: set[str]
) -> tuple[pd.DataFrame, bool]:
    """Class probabilities (cells x cell types), and whether reference mapping was used."""
    import scvi

    clean_batches(adata, cfg.batch_key)
    add_label_column(adata, None)
    mapped = needs_reference_mapping(adata.obs[cfg.batch_key], known_batches)
    if mapped:
        scvi.model.SCANVI.prepare_query_anndata(adata, model)
        query = scvi.model.SCANVI.load_query_data(adata, model)
        query.train(
            max_epochs=cfg.query_epochs,
            batch_size=cfg.batch_size,
            plan_kwargs={"weight_decay": 0.0},  # keep new-batch parameters near the reference
            accelerator="auto",
        )
        model = query
    proba = model.predict(adata, soft=True, batch_size=cfg.batch_size)
    return pd.DataFrame(proba), mapped


def top_label(proba: pd.DataFrame) -> tuple[Any, Any]:
    """Most likely cell type per cell and its probability."""
    values = proba.to_numpy()
    return proba.columns.to_numpy().astype(str)[values.argmax(axis=1)], values.max(axis=1)
