"""Reading model-input files."""

from pathlib import Path
from typing import Any

import h5py
import pandas as pd
from anndata.io import read_elem


def load_split(path: str | Path) -> tuple[Any, pd.DataFrame]:
    """Normalized matrix and cell metadata only (skips the large raw-counts layer)."""
    with h5py.File(path, "r") as f:
        return read_elem(f["X"]), read_elem(f["obs"])
