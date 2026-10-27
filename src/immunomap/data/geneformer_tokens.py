"""Geneformer V2 tokenization: each cell becomes its genes ranked by expression.

Reimplements the Geneformer TranscriptomeTokenizer (tokenize_anndata, V2 settings) so
that the product can tokenize uploads without the geneformer package. Per cell:

1. Map Ensembl IDs through Geneformer's mapping dictionary; genes mapping to the same
   ID are summed; genes outside the token dictionary are dropped.
2. Normalize: counts / total counts of the cell (all genes) * 10,000, then divide each
   gene by its non-zero median across Geneformer's pretraining corpus. This turns
   "always high" housekeeping genes down and cell-type-specific genes up.
3. Rank expressed genes from highest to lowest; the ranked gene tokens are the input.
4. Keep the top 4,094 and wrap them in <cls> ... <eos> (4,096 tokens at most).

tests/unit/test_geneformer_tokens.py checks this against a line-by-line transcription
of the original algorithm.
"""

import pickle
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import scipy.sparse as sp

TARGET_SUM = 10_000


@dataclass(frozen=True)
class GeneformerVocab:
    token_dict: dict[str, int]  # Ensembl ID (and <pad>, <mask>, <cls>, <eos>) -> token
    median_dict: dict[str, float]  # Ensembl ID -> non-zero median in the pretraining corpus
    mapping_dict: dict[str, str]  # Ensembl ID -> collapsed Ensembl ID (in the vocabulary)
    model_input_size: int = 4096

    @classmethod
    def from_files(
        cls, token_file: str | Path, median_file: str | Path, mapping_file: str | Path
    ) -> Self:
        def load(path: str | Path) -> Any:
            with open(path, "rb") as f:
                return pickle.load(f)

        tokens = load(token_file)
        # Same filter as the original: keep mappings whose target is in the vocabulary
        mapping = {k: v for k, v in load(mapping_file).items() if v in tokens}
        return cls(token_dict=tokens, median_dict=load(median_file), mapping_dict=mapping)

    @property
    def cls_token(self) -> int:
        return self.token_dict["<cls>"]

    @property
    def eos_token(self) -> int:
        return self.token_dict["<eos>"]

    @property
    def pad_token(self) -> int:
        return self.token_dict["<pad>"]


@dataclass(frozen=True)
class GeneMap:
    """How a dataset's genes become Geneformer genes (computed once per gene list)."""

    collapse: Any  # sparse (n_input_genes x n_kept): sums input genes into Geneformer genes
    tokens: Any  # token of each kept Geneformer gene
    medians: Any  # normalization factor of each kept Geneformer gene

    @property
    def n_genes(self) -> int:
        return len(self.tokens)


def build_gene_map(ensembl_ids: Any, vocab: GeneformerVocab) -> GeneMap:
    """Kept genes in the original tokenizer's order: genes whose collapsed ID is unique, in
    input order, then IDs shared by several input genes (summed), in order of first
    appearance. Order only matters for exact ties in the ranking, but it keeps us identical.
    """
    collapsed = [vocab.mapping_dict.get(str(g).upper()) for g in ensembl_ids]
    present = [c for c in collapsed if c is not None]
    counts = Counter(present)
    unique = [c for c in present if counts[c] == 1]
    shared = [c for c in dict.fromkeys(present) if counts[c] > 1]
    targets = unique + shared
    column = {gene: j for j, gene in enumerate(targets)}
    rows = [i for i, c in enumerate(collapsed) if c is not None]
    cols = [column[c] for c in present]
    collapse = sp.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(len(collapsed), len(targets))
    )
    return GeneMap(
        collapse=collapse,
        tokens=np.array([vocab.token_dict[g] for g in targets], dtype=np.int64),
        medians=np.array([vocab.median_dict[g] for g in targets], dtype=np.float64),
    )


def tokenize_counts(
    X: Any, gene_map: GeneMap, vocab: GeneformerVocab, n_counts: Any = None
) -> list[Any]:
    """Token arrays (int64, <cls> ... <eos>) for each cell of a raw-count matrix.

    n_counts defaults to each cell's total over ALL input genes, as Geneformer expects.
    Cells with no Geneformer gene expressed get just <cls> <eos>.
    """
    X = sp.csr_matrix(X, dtype=np.float64)
    if n_counts is None:
        n_counts = np.asarray(X.sum(axis=1)).ravel()
    n_counts = np.asarray(n_counts, dtype=np.float64)
    norm = sp.csr_matrix(X @ gene_map.collapse)
    norm.eliminate_zeros()
    norm.sort_indices()  # values in gene order before ranking, as in the original
    # Same arithmetic order as the original: counts / n_counts * 10,000 / median
    rows = np.repeat(np.arange(norm.shape[0]), np.diff(norm.indptr))
    norm.data = norm.data / n_counts[rows] * TARGET_SUM / gene_map.medians[norm.indices]

    keep = vocab.model_input_size - 2
    head = np.array([vocab.cls_token], dtype=np.int64)
    tail = np.array([vocab.eos_token], dtype=np.int64)
    cells = []
    for i in range(norm.shape[0]):
        start, end = norm.indptr[i], norm.indptr[i + 1]
        values, genes = norm.data[start:end], norm.indices[start:end]
        ranked = gene_map.tokens[genes][np.argsort(-values)][:keep]
        cells.append(np.concatenate([head, ranked, tail]))
    return cells


def pack(cells: list[Any]) -> tuple[Any, Any]:
    """Store variable-length token arrays compactly: flat int16 tokens + int64 offsets."""
    lengths = np.array([len(c) for c in cells], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    flat = np.concatenate(cells) if cells else np.zeros(0, dtype=np.int64)
    return flat.astype(np.int16), offsets


def unpack(flat: Any, offsets: Any, i: int) -> Any:
    return np.asarray(flat[offsets[i] : offsets[i + 1]], dtype=np.int64)
