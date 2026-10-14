"""Check local expression shards against the checksums recorded at download time.

Usage:
    uv run python scripts/verify_shards.py
Exit code 0 = every shard present and byte-identical; 1 = something is wrong.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="docs/shard_index_2025-11-08.csv")
    parser.add_argument("--data-dir", default="data/raw/expression")
    args = parser.parse_args()

    index = pd.read_csv(args.index)
    data_dir = Path(args.data_dir)
    problems = []
    ok = 0

    for row in index.itertuples():
        path = data_dir / row.shard
        if not path.exists():
            problems.append(f"missing: {row.shard}")
        elif path.stat().st_size != row.bytes:
            problems.append(f"wrong size: {row.shard}")
        elif sha256(path) != row.sha256:
            problems.append(f"checksum mismatch: {row.shard}")
        else:
            ok += 1
            print(f"ok  {row.shard}")

    extra = sorted({p.name for p in data_dir.glob("*.h5ad")} - set(index["shard"]))
    problems += [f"not in index: {name}" for name in extra]

    print(f"\n{ok}/{len(index)} shards OK, {index['cells'].sum():,} cells")
    for problem in problems:
        print("PROBLEM", problem)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
