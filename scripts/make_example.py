"""Build the demo's example file: 2,000 random cells from the Hao 2021 study.

Hao 2021 is one of the two external studies the model never trained on, so the demo
shows honest performance. All genes are kept: QC counts detected genes and the
mitochondrial fraction over the whole transcriptome, so a gene-filtered file would fail
QC for most cells. Author labels are kept as
`coarse` / `fine` so the app can show agreement; they are never used for prediction.
Source: CELLxGENE Census 2025-11-08 (CC BY 4.0).

Usage:
    uv run python scripts/make_example.py
"""

import argparse
from pathlib import Path

import anndata as ad
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qc-dir", default="data/processed/qc")
    parser.add_argument("--n-cells", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="data/examples/hao2021_2000cells.h5ad")
    args = parser.parse_args()

    shard = sorted(Path(args.qc_dir).glob("*hao*.h5ad"))[0]
    adata = ad.read_h5ad(shard)
    rng = np.random.default_rng(args.seed)
    cells = np.sort(rng.choice(adata.n_obs, args.n_cells, replace=False))
    example = adata[cells].copy()
    example.obs = example.obs[["coarse", "fine"]].copy()
    example.var = example.var[["feature_name"]].copy()
    example.uns = {"source": "Hao et al. 2021 via CELLxGENE Census 2025-11-08 (CC BY 4.0)"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    example.write_h5ad(out, compression="gzip")
    print(
        f"{out}: {example.n_obs:,} cells x {example.n_vars:,} genes, "
        f"{out.stat().st_size / 1e6:.1f} MB (from {shard.name})"
    )


if __name__ == "__main__":
    main()
