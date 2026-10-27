"""Our Geneformer tokenizer must produce exactly what the original produces.

`_reference` transcribes Geneformer's TranscriptomeTokenizer.tokenize_anndata (V2,
h5ad path: sum_ensembl_ids, normalization, rank_genes, create_dataset truncation) as
closely as possible without its loompy/datasets dependencies.
"""

from collections import Counter
from dataclasses import replace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from immunomap.data.geneformer_tokens import (
    GeneformerVocab,
    build_gene_map,
    pack,
    tokenize_counts,
    unpack,
)

GENES = [f"ENSG{i:011d}" for i in range(40)]


def _vocab(input_size: int = 4096) -> GeneformerVocab:
    tokens = {"<pad>": 0, "<mask>": 1, "<cls>": 2, "<eos>": 3}
    tokens |= {g: 4 + i for i, g in enumerate(GENES[:30])}  # last 10 genes unknown
    rng = np.random.default_rng(0)
    medians = {g: float(rng.uniform(0.5, 5.0)) for g in GENES[:30]}
    mapping = {g: g for g in GENES}  # identity, as for most genes in the real file
    mapping["ENSGOLDALIAS1"] = GENES[5]  # retired ID that maps onto a current gene
    mapping["ENSGOLDALIAS2"] = GENES[7]
    mapping = {k: v for k, v in mapping.items() if v in tokens}
    return GeneformerVocab(tokens, medians, mapping, model_input_size=input_size)


def _reference(X: np.ndarray, ids: list[str], vocab: GeneformerVocab) -> list[np.ndarray]:
    mapping, token_dict = vocab.mapping_dict, vocab.token_dict
    n_counts = X.sum(axis=1)[:, None]  # obs["n_counts"]: total over all genes
    in_map = [g for g in ids if g in mapping]
    collapsed = pd.Series(ids).str.upper().map(mapping)
    if len(set(in_map)) == len(set(mapping.get(g) for g in in_map)):
        data, names = X, collapsed.to_numpy()
    else:
        keep = ~collapsed.isna().to_numpy()
        X_kept, names_kept = X[:, keep], collapsed[keep].to_numpy()
        dup = [g for g, c in Counter(names_kept).items() if c > 1]
        summed = np.stack([X_kept[:, names_kept == g].sum(axis=1) for g in dup], axis=1)
        not_dup = ~np.isin(names_kept, dup)
        data = np.concatenate([X_kept[:, not_dup], summed], axis=1)
        names = np.concatenate([names_kept[not_dup], dup])
    genelist = dict.fromkeys(token_dict, True)
    loc = np.where([genelist.get(i, False) for i in names])[0]
    norm_factor = np.array([vocab.median_dict[i] for i in names[loc]])
    tokens = np.array([token_dict[i] for i in names[loc]])
    X_norm = sp.csr_matrix(data[:, loc] / n_counts * 10_000 / norm_factor)
    cells = []
    for i in range(X_norm.shape[0]):
        ranked = tokens[X_norm[i].indices][np.argsort(-X_norm[i].data)]
        ranked = ranked[: vocab.model_input_size - 2]
        cells.append(np.concatenate([[2], ranked, [3]]))
    return cells


def _counts(n_cells: int, n_genes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = rng.poisson(rng.uniform(0.1, 4.0, n_genes), (n_cells, n_genes)).astype(np.float32)
    X[rng.random(X.shape) < 0.5] = 0
    return X


def _check(ids: list[str], vocab: GeneformerVocab, seed: int) -> None:
    X = _counts(60, len(ids), seed)
    ours = tokenize_counts(sp.csr_matrix(X), build_gene_map(ids, vocab), vocab)
    for a, b in zip(ours, _reference(X, ids, vocab), strict=True):
        assert np.array_equal(a, b)


def test_matches_original_when_every_gene_maps_to_itself() -> None:
    _check(GENES, _vocab(), seed=1)


def test_matches_original_when_old_ids_are_summed_into_one_gene() -> None:
    ids = [*GENES[:20], "ENSGOLDALIAS1", *GENES[20:], "ENSGOLDALIAS2"]
    _check(ids, _vocab(), seed=2)


def test_matches_original_when_truncating_long_cells() -> None:
    _check(GENES, _vocab(input_size=8), seed=3)


def test_cells_start_with_cls_and_end_with_eos_and_skip_unknown_genes() -> None:
    vocab = _vocab()
    X = np.zeros((2, 40), dtype=np.float32)
    X[0, 35] = 9  # only an unknown gene expressed
    X[1, [3, 4]] = [1, 5]
    cells = tokenize_counts(sp.csr_matrix(X), build_gene_map(GENES, vocab), vocab)
    assert cells[0].tolist() == [2, 3]
    assert cells[1][0] == 2 and cells[1][-1] == 3 and len(cells[1]) == 4


def test_library_size_uses_all_genes() -> None:
    vocab = _vocab()
    gene_map = build_gene_map(GENES, vocab)
    X = np.zeros((1, 40), dtype=np.float32)
    X[0, [0, 1]] = [3, 1]
    alone = tokenize_counts(sp.csr_matrix(X), gene_map, vocab)
    X[0, 39] = 1000  # an unknown gene changes the library size, not the ranking
    assert np.array_equal(alone[0], tokenize_counts(sp.csr_matrix(X), gene_map, vocab)[0])


def test_pack_and_unpack_round_trip() -> None:
    cells = [np.array([2, 10, 11, 3]), np.array([2, 3]), np.array([2, 20_000, 3])]
    flat, offsets = pack(cells)
    assert flat.dtype == np.int16
    for i, cell in enumerate(cells):
        assert np.array_equal(unpack(flat, offsets, i), cell)


def test_vocab_properties() -> None:
    vocab = replace(_vocab(), model_input_size=4096)
    assert (vocab.pad_token, vocab.cls_token, vocab.eos_token) == (0, 2, 3)
