"""Select highly variable genes (HVGs) from TRAINING cells only.

Val, test and external cells are never looked at, so gene selection cannot leak
information from the evaluation sets. Genes are ranked by how many study groups
call them highly variable (robust to batch effects), then by dispersion.

Canonical marker genes from configs/marker_genes.yaml are always added, so the
model can never lose a key lineage marker (e.g. CD3E, which is expressed so broadly
that it is not "highly variable" even though it defines T cells).

Usage:
    uv run python scripts/select_genes.py
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import yaml


def load_training_subsample(shards: list[Path], n_cells: int, seed: int) -> ad.AnnData:
    sizes = []
    for path in shards:
        backed = ad.read_h5ad(path, backed="r")
        sizes.append(backed.n_obs)
        backed.file.close()
    total = sum(sizes)
    rng = np.random.default_rng(seed)
    parts = []
    for path, size in zip(shards, sizes, strict=True):
        take = min(size, round(n_cells * size / total))
        adata = ad.read_h5ad(path)
        idx = np.sort(rng.choice(size, size=take, replace=False))
        parts.append(adata[idx].copy())
        del adata
        print(f"sampled {take:,} of {size:,} cells from {path.name}")
    return ad.concat(parts, merge="same")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed/qc")
    parser.add_argument("--n-cells", type=int, default=200_000)
    parser.add_argument("--n-genes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--markers", default="configs/marker_genes.yaml")
    parser.add_argument("--out", default="artifacts/hvg_genes.csv")
    args = parser.parse_args()

    shards = sorted(Path(args.input_dir).glob("*__train__*.h5ad"))
    if not shards:
        raise SystemExit(f"no training shards in {args.input_dir}")
    adata = load_training_subsample(shards, args.n_cells, args.seed)

    # Mitochondrial genes track cell quality, not cell type: leave them out
    is_mt = adata.var["feature_name"].astype(str).str.upper().str.startswith("MT-").to_numpy()
    adata = adata[:, ~is_mt].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_genes, batch_key="group", flavor="seurat")

    hvg = adata.var[adata.var["highly_variable"]]
    hvg = hvg.sort_values(
        ["highly_variable_nbatches", "dispersions_norm"], ascending=[False, False]
    )
    table = hvg[["feature_name", "highly_variable_nbatches", "dispersions_norm"]]
    table = table.rename_axis("feature_id").reset_index().assign(source="hvg")

    markers = [
        g for genes in yaml.safe_load(Path(args.markers).read_text()).values() for g in genes
    ]
    names = adata.var["feature_name"].astype(str)
    unknown = [m for m in markers if m not in set(names)]
    if unknown:
        raise SystemExit(f"marker genes not found in the data: {unknown}")
    added = [m for m in markers if m not in set(table["feature_name"])]
    extra = pd.DataFrame(
        {"feature_id": [names[names == m].index[0] for m in added], "feature_name": added}
    ).assign(source="marker")
    table = pd.concat([table, extra], ignore_index=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, lineterminator="\n")

    n_groups = adata.obs["group"].nunique()
    is_hvg = table["source"] == "hvg"
    print(
        f"\n{int(is_hvg.sum()):,} HVGs from {adata.n_obs:,} training cells + {len(added)} markers"
    )
    print(
        "HVGs variable in all",
        n_groups,
        "groups:",
        int((table["highly_variable_nbatches"] == n_groups).sum()),
    )
    print("markers added (not highly variable):", added)
    print(f"total genes: {len(table):,} -> {out}")


if __name__ == "__main__":
    main()
