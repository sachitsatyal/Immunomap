"""Download raw counts for every cell in the manifest, as sharded .h5ad files.

Runs on Linux (Colab or WSL), because cellxgene-census has no Windows build.
Resumable: a shard is skipped once its .json record exists in --out, so a
disconnected Colab session can simply be re-run with the same command.
"""

import argparse
import gc
import hashlib
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

SHARD_SIZE = 40_000  # cells per shard; ~30k cells peaked well within free Colab RAM
MANIFEST_OBS = ["group", "dataset_id", "donor_id", "assay", "split", "lineage", "coarse", "fine"]


def plan_shards(manifest: pd.DataFrame, shard_size: int) -> list[tuple[str, np.ndarray]]:
    """Split each (group, split) into near-equal shards of at most shard_size cells."""
    shards = []
    for (group, split), df in manifest.groupby(["group", "split"], sort=True):
        ids = np.sort(df["soma_joinid"].to_numpy())
        n_chunks = -(-len(ids) // shard_size)  # ceiling division
        for i, chunk in enumerate(np.array_split(ids, n_chunks)):
            shards.append((f"{group}__{split}__{i:02d}", chunk))
    return shards


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pull_shard(census: Any, organism: str, ids: np.ndarray, labels: pd.DataFrame) -> Any:
    import cellxgene_census  # Linux only

    adata = cellxgene_census.get_anndata(
        census,
        organism=organism,
        obs_coords=ids,
        obs_column_names=["soma_joinid"],
        var_column_names=["feature_id", "feature_name"],
    )
    obs = adata.obs.join(labels, on="soma_joinid")
    if len(obs) != len(ids) or obs["split"].isna().any():
        raise RuntimeError("downloaded cells do not match the manifest")
    # Unnamed indexes: an index named like a column (soma_joinid) cannot be written to .h5ad
    obs.index = pd.Index(obs["soma_joinid"].astype(str).to_numpy(), name=None)
    adata.obs = obs
    adata.var_names = pd.Index(adata.var["feature_id"].astype(str).to_numpy(), name=None)
    return adata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-shards", type=int, default=None, help="stop after N new shards")
    parser.add_argument("--shard-size", type=int, default=SHARD_SIZE, help="max cells per shard")
    args = parser.parse_args()

    import cellxgene_census  # Linux only

    cfg = yaml.safe_load(Path(args.data_config).read_text())
    manifest = pd.read_parquet(args.manifest)
    labels = manifest.set_index("soma_joinid")[MANIFEST_OBS]
    shards = plan_shards(manifest, args.shard_size)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(shards)} shards planned, {len(manifest):,} cells")

    new = 0
    with cellxgene_census.open_soma(census_version=cfg["census_version"]) as census:
        for name, ids in shards:
            record_path = out / f"{name}.json"
            if record_path.exists():
                continue
            if args.max_shards is not None and new >= args.max_shards:
                break
            start = time.time()
            adata = pull_shard(census, cfg["organism"], ids, labels)
            adata.uns["census_version"] = cfg["census_version"]
            tmp = Path(tempfile.gettempdir()) / f"{name}.h5ad"
            adata.write_h5ad(tmp, compression="gzip")
            record = {
                "shard": f"{name}.h5ad",
                "cells": int(adata.n_obs),
                "genes": int(adata.n_vars),
                "nonzeros": int(adata.X.nnz),
                "bytes": tmp.stat().st_size,
                "sha256": sha256(tmp),
                "seconds": round(time.time() - start),
            }
            shutil.move(tmp, out / f"{name}.h5ad")
            record_path.write_text(json.dumps(record, indent=2))
            new += 1
            del adata  # free this shard before downloading the next one
            gc.collect()
            print(
                f"{name}: {record['cells']:,} cells, {record['bytes'] / 1e9:.2f} GB, "
                f"{record['seconds']} s"
            )

    records = [json.loads(p.read_text()) for p in sorted(out.glob("*.json"))]
    index = pd.DataFrame(records)
    index.to_csv(out / "shard_index.csv", index=False)
    print(
        f"\n{len(index)}/{len(shards)} shards done: {index['cells'].sum():,} cells, "
        f"{index['bytes'].sum() / 1e9:.2f} GB"
    )


if __name__ == "__main__":
    main()
