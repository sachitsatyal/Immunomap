"""Annotate an .h5ad file of raw counts with the exported bundle.

    uv run python scripts/annotate.py input.h5ad output.csv
    uv run python scripts/annotate.py input.h5ad output.csv --max-cells 5000 --evaluate

--evaluate compares against `coarse` / `fine` columns in the file, if present: useful
for checking the bundle end to end on a study the model has never seen.
"""

import argparse
import json
import time

import anndata as ad
import numpy as np

from immunomap.evaluation.metrics import classification_metrics
from immunomap.inference.annotate import Annotator
from immunomap.models.calibration import UNKNOWN


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--bundle", default="data/models/bundle/immunomap-scanvi-v1")
    parser.add_argument("--batch-key", help="obs column with sample IDs, for per-sample QC")
    parser.add_argument("--max-cells", type=int, help="random subsample (quick checks)")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    if args.max_cells and adata.n_obs > args.max_cells:
        rng = np.random.default_rng(args.seed)
        adata = adata[np.sort(rng.choice(adata.n_obs, args.max_cells, replace=False))].copy()

    start = time.time()
    table, summary = Annotator.from_bundle(args.bundle).annotate(adata, args.batch_key)
    summary["seconds"] = round(time.time() - start, 1)
    table.to_csv(args.output, lineterminator="\n")
    print(json.dumps(summary, indent=2))
    print(table["coarse"].value_counts().to_string())

    if args.evaluate:
        for level in ("coarse", "fine"):
            if level not in adata.obs:
                continue
            truth = adata.obs[level].astype(object).to_numpy()
            has = np.array([isinstance(t, str) for t in truth])
            called = has & ~np.isin(table[level].to_numpy(), [UNKNOWN, "QC failed"])
            scores = classification_metrics(
                truth[called].astype(str), table[level].to_numpy()[called].astype(str)
            )
            print(
                f"{level}: {has.sum():,} labelled cells, {called.sum() / max(has.sum(), 1):.1%} "
                f"called; on called cells accuracy {scores['accuracy']:.3f}, "
                f"macro-F1 {scores['macro_f1']:.3f}"
            )


if __name__ == "__main__":
    main()
