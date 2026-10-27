"""Tokenize every QC'd cell for Geneformer (all genes, not the 3,002 model genes).

For each split, cells are read from data/processed/qc in the same order as the model
input files, so row i here is row i in data/processed/features/<split>.h5ad. Output per
split in data/processed/geneformer/:
    <split>_tokens.npy   all cells' tokens, concatenated (int16)
    <split>_offsets.npy  cell i = tokens[offsets[i]:offsets[i+1]] (int64)
    <split>_obs.csv      cell id, study, dataset, donor and labels

Usage:
    uv run python scripts/tokenize_geneformer.py
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from immunomap.data.geneformer_tokens import GeneformerVocab, build_gene_map, tokenize_counts

SPLITS = ["train", "val", "test", "external"]
OBS_COLUMNS = ["soma_joinid", "group", "dataset_id", "donor_id", "split", "coarse", "fine"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geneformer.yaml")
    parser.add_argument("--input-dir", default="data/processed/qc")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    local = Path(cfg["local_dir"])
    vocab = GeneformerVocab.from_files(
        local / cfg["token_dictionary"], local / cfg["gene_median"], local / cfg["gene_mapping"]
    )
    vocab = replace(vocab, model_input_size=cfg["model_input_size"])
    out_dir = Path(cfg["tokens_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    gene_map, genes = None, None
    for split in SPLITS:
        start = time.time()
        shards = sorted(Path(args.input_dir).glob(f"*__{split}__*.h5ad"))
        if not shards:
            raise SystemExit(f"no {split} shards in {args.input_dir}")
        flats, lengths, obs = [], [], []
        for path in shards:
            adata = ad.read_h5ad(path)
            if genes is None or not np.array_equal(genes, adata.var_names):
                genes = np.asarray(adata.var_names)
                gene_map = build_gene_map(genes, vocab)
                print(f"{gene_map.n_genes:,} of {len(genes):,} genes are Geneformer genes")
            assert gene_map is not None
            cells = tokenize_counts(adata.X, gene_map, vocab)
            lengths.append(np.array([len(c) for c in cells], dtype=np.int64))
            flats.append(np.concatenate(cells).astype(np.int16))
            obs.append(adata.obs[[c for c in OBS_COLUMNS if c in adata.obs]].copy())
            del adata, cells

        length = np.concatenate(lengths)
        offsets = np.concatenate([[0], np.cumsum(length)])
        np.save(out_dir / f"{split}_tokens.npy", np.concatenate(flats))
        np.save(out_dir / f"{split}_offsets.npy", offsets)
        table = pd.concat(obs)
        table.index.name = "cell_id"
        table.to_csv(out_dir / f"{split}_obs.csv", lineterminator="\n")

        genes_per_cell = length - 2  # without <cls> and <eos>
        median, p99 = np.median(genes_per_cell), np.percentile(genes_per_cell, 99)
        truncated = (genes_per_cell >= vocab.model_input_size - 2).mean()
        print(
            f"{split:9s} {len(length):>9,} cells in {time.time() - start:.0f} s | "
            f"genes per cell: median {int(median):,}, p99 {int(p99):,}, "
            f"max {int(genes_per_cell.max()):,} | truncated {truncated:.2%} | "
            f"empty {int((genes_per_cell == 0).sum())}"
        )


if __name__ == "__main__":
    main()
