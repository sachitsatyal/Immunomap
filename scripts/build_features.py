"""Build one model-input file per split from the QC'd shards.

Each output has the model's genes (artifacts/hvg_genes.csv, in that order),
X = log1p(counts per 10k), layers['counts'] = raw counts, and all cell metadata.

Usage:
    uv run python scripts/build_features.py
"""

import argparse
from pathlib import Path

import anndata as ad

from immunomap.data.features import TARGET_SUM, build_model_input, load_gene_list

SPLITS = ["train", "val", "test", "external"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed/qc")
    parser.add_argument("--genes", default="artifacts/hvg_genes.csv")
    parser.add_argument("--output-dir", default="data/processed/features")
    args = parser.parse_args()

    genes = load_gene_list(args.genes)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        shards = sorted(Path(args.input_dir).glob(f"*__{split}__*.h5ad"))
        if not shards:
            raise SystemExit(f"no {split} shards in {args.input_dir}")
        parts = []
        for path in shards:
            part = build_model_input(ad.read_h5ad(path), genes)
            if part.uns["n_missing_genes"]:
                raise RuntimeError(f"{path.name}: {part.uns['n_missing_genes']} genes missing")
            parts.append(part)
        combined = ad.concat(parts, merge="same")
        combined.var["feature_name"] = parts[0].var["feature_name"].to_numpy()
        combined.uns["normalization"] = f"log1p(counts / total_counts_all_genes * {TARGET_SUM:g})"
        combined.uns["gene_list"] = args.genes
        combined.write_h5ad(out_dir / f"{split}.h5ad", compression="gzip")
        del parts

        groups = combined.obs["group"].value_counts().to_dict()
        print(f"{split:9s} {combined.n_obs:>9,} cells x {combined.n_vars:,} genes  {groups}")
        del combined


if __name__ == "__main__":
    main()
