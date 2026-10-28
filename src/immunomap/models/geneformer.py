"""Data handling for fine-tuning Geneformer (no torch here, so CI can test it).

The model code itself (LoRA adapters, classification heads, training loop) lives in
scripts/finetune_geneformer.py, which runs on the GPU machine only.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import pandas as pd
import yaml

IGNORE = -100  # label for "no label at this level": skipped by the loss and the metrics


@dataclass(frozen=True)
class FinetuneConfig:
    per_label_cap: int  # training cells per cell type (fine label, else coarse); rare kept whole
    epochs: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    max_tokens: int  # padded tokens per training batch (cells x longest cell in batch)
    eval_max_tokens: int  # same, for prediction (no gradients, so larger)
    megabatch: int  # cells sorted by length together, so batches carry little padding
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: list[str]
    head_dropout: float
    val_monitor_cells: int  # val cells scored after each epoch (full scoring at the end)
    seed: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls(**yaml.safe_load(Path(path).read_text()))


class TokenStore:
    """Tokenized cells of one split (from scripts/tokenize_geneformer.py), read lazily."""

    def __init__(self, directory: str | Path, split: str) -> None:
        directory = Path(directory)
        self.tokens = np.load(directory / f"{split}_tokens.npy", mmap_mode="r")
        self.offsets = np.load(directory / f"{split}_offsets.npy")
        self.obs = pd.read_csv(directory / f"{split}_obs.csv", index_col="cell_id")
        if len(self.obs) != len(self.offsets) - 1:
            raise ValueError(f"{split}: {len(self.obs)} cells but {len(self.offsets) - 1} offsets")

    def __len__(self) -> int:
        return len(self.offsets) - 1

    @property
    def lengths(self) -> Any:
        return np.diff(self.offsets)

    def cell(self, i: int) -> Any:
        return np.asarray(self.tokens[self.offsets[i] : self.offsets[i + 1]], dtype=np.int64)


def sampling_key(obs: pd.DataFrame) -> Any:
    """The most specific label of each cell (fine, else coarse; None if unlabelled)."""
    return np.array(
        [
            f if isinstance(f, str) else c if isinstance(c, str) else None
            for f, c in zip(obs["fine"], obs["coarse"], strict=True)
        ],
        dtype=object,
    )


def class_list(values: Any) -> list[str]:
    return sorted({str(v) for v in values if isinstance(v, str)})


def encode_labels(values: Any, classes: list[str]) -> Any:
    """Class index per cell; IGNORE where the cell has no label (or one outside `classes`)."""
    index = {c: i for i, c in enumerate(classes)}
    return np.array(
        [index.get(v, IGNORE) if isinstance(v, str) else IGNORE for v in values], dtype=np.int64
    )


def make_batches(
    lengths: Any, max_tokens: int, megabatch: int, rng: np.random.Generator | None
) -> list[Any]:
    """Groups of cell positions whose padded size (cells x longest) stays within max_tokens.

    Training (rng given): shuffle, sort each megabatch by length, pack, shuffle the batches.
    Prediction (rng None): sort everything by length; callers put results back by position.
    """
    lengths = np.asarray(lengths)
    if rng is None:
        chunks = [np.argsort(-lengths, kind="stable")]
    else:
        order = rng.permutation(len(lengths))
        chunks = [order[i : i + megabatch] for i in range(0, len(order), megabatch)]
    batches = []
    for chunk in chunks:
        chunk = chunk[np.argsort(-lengths[chunk], kind="stable")]
        start = 0
        while start < len(chunk):
            longest = lengths[chunk[start]]
            size = max(1, min(len(chunk) - start, max_tokens // max(int(longest), 1)))
            batches.append(chunk[start : start + size])
            start += size
    if rng is not None:
        batches = [batches[i] for i in rng.permutation(len(batches))]
    return batches


def collate(cells: list[Any], pad_token: int) -> tuple[Any, Any]:
    """Pad token arrays to the longest: (input_ids, attention_mask), both int64."""
    width = max(len(c) for c in cells)
    ids = np.full((len(cells), width), pad_token, dtype=np.int64)
    mask = np.zeros((len(cells), width), dtype=np.int64)
    for row, cell in enumerate(cells):
        ids[row, : len(cell)] = cell
        mask[row, : len(cell)] = 1
    return ids, mask
