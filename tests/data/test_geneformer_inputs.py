"""Checks on the Geneformer token files. Skipped where they have not been generated."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from anndata.io import read_elem

TOKENS = Path("data/processed/geneformer")
FEATURES = Path("data/processed/features")
SPLITS = ["train", "val", "test", "external"]
pytestmark = pytest.mark.skipif(
    not (TOKENS / "external_offsets.npy").exists(), reason="Geneformer tokens not generated"
)


@pytest.mark.parametrize("split", SPLITS)
def test_same_cells_in_same_order_as_model_inputs(split: str) -> None:
    obs = pd.read_csv(TOKENS / f"{split}_obs.csv", usecols=["soma_joinid"])
    with h5py.File(FEATURES / f"{split}.h5ad", "r") as f:
        features = read_elem(f["obs"])
    assert np.array_equal(obs["soma_joinid"].to_numpy(), features["soma_joinid"].to_numpy())


@pytest.mark.parametrize("split", SPLITS)
def test_every_cell_is_wrapped_in_cls_and_eos(split: str) -> None:
    tokens = np.load(TOKENS / f"{split}_tokens.npy", mmap_mode="r")
    offsets = np.load(TOKENS / f"{split}_offsets.npy")
    starts, ends = offsets[:-1], offsets[1:] - 1
    assert offsets[-1] == len(tokens)
    assert (tokens[starts] == 2).all() and (tokens[ends] == 3).all()
    assert (np.diff(offsets) <= 4096).all()
    sample = np.asarray(tokens[: min(len(tokens), 5_000_000)])
    assert sample.min() >= 0 and sample.max() < 20_275
