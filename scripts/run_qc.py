"""Run cell QC on every expression shard.

Pass 1 computes per-cell metrics for all shards, so each batch's thresholds use
all of its cells (not one shard's worth). Pass 2 writes the passing cells.

Usage:
    uv run python scripts/run_qc.py
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from immunomap.data.qc import PASS, QCConfig, compute_metrics, compute_thresholds, flag_cells

METRICS = ["total_counts", "n_genes", "pct_mt"]


def collect_metrics(shards: list[Path], cfg: QCConfig) -> pd.DataFrame:
    parts = []
    for path in shards:
        adata = ad.read_h5ad(path)
        m = compute_metrics(adata.X, adata.var["feature_name"].tolist())
        m.index = adata.obs_names
        for col in sorted({cfg.batch_key, "group", "split", "coarse"}):
            m[col] = adata.obs[col].astype(str).to_numpy()
        m["shard"] = path.name
        parts.append(m)
        print(f"metrics  {path.name}: {adata.n_obs:,} cells")
    return pd.concat(parts)


def write_reports(metrics: pd.DataFrame, thresholds: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    removed = metrics["qc_reason"] != PASS

    readable = thresholds.assign(
        counts_lo=np.expm1(thresholds["log_counts_lo"]).round(),
        counts_hi=np.expm1(thresholds["log_counts_hi"]).round(),
        genes_lo=np.expm1(thresholds["log_genes_lo"]).round(),
        genes_hi=np.expm1(thresholds["log_genes_hi"]).round(),
    )
    readable.to_csv(report_dir / "qc_thresholds.csv", index_label="batch", lineterminator="\n")

    summary = (
        metrics.assign(removed=removed)
        .groupby(["group", "split"])
        .agg(cells=("removed", "size"), removed=("removed", "sum"))
    )
    summary["pct_removed"] = (100 * summary["removed"] / summary["cells"]).round(2)
    summary.to_csv(report_dir / "qc_summary.csv", lineterminator="\n")

    reasons = pd.crosstab(metrics["group"], metrics["qc_reason"])
    reasons.to_csv(report_dir / "qc_reasons.csv", lineterminator="\n")

    by_label = (
        metrics.assign(removed=removed)
        .groupby("coarse")
        .agg(cells=("removed", "size"), removed=("removed", "sum"))
    )
    by_label["pct_removed"] = (100 * by_label["removed"] / by_label["cells"]).round(2)
    by_label = by_label.sort_values("pct_removed", ascending=False)
    by_label.to_csv(report_dir / "qc_removed_by_label.csv", lineterminator="\n")

    print("\n", summary.to_string())
    print("\n", reasons.to_string())
    print("\n", by_label.to_string())
    print(f"\nTotal: {len(metrics):,} cells, {int(removed.sum()):,} removed "
          f"({100 * removed.mean():.2f}%)")  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/qc.yaml")
    parser.add_argument("--input-dir", default="data/raw/expression")
    parser.add_argument("--output-dir", default="data/processed/qc")
    parser.add_argument("--report-dir", default="docs/qc")
    args = parser.parse_args()

    cfg = QCConfig.from_yaml(args.config)
    shards = sorted(Path(args.input_dir).glob("*.h5ad"))
    if not shards:
        raise SystemExit(f"no .h5ad files in {args.input_dir}")

    metrics = collect_metrics(shards, cfg)
    batches = metrics[cfg.batch_key]
    thresholds = compute_thresholds(metrics, batches, cfg)
    metrics["qc_reason"] = flag_cells(metrics, batches, thresholds, cfg).to_numpy()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for path in shards:
        adata = ad.read_h5ad(path)
        m = metrics[metrics["shard"] == path.name]
        if not (m.index == adata.obs_names).all():
            raise RuntimeError(f"cell order changed between passes in {path.name}")
        for col in [*METRICS, "qc_reason"]:
            adata.obs[col] = m[col].to_numpy()
        kept = adata[adata.obs["qc_reason"] == PASS].copy()
        kept.obs = kept.obs.drop(columns="qc_reason")
        kept.uns["qc_config"] = asdict(cfg)
        kept.write_h5ad(out / path.name, compression="gzip")
        print(f"written  {path.name}: kept {kept.n_obs:,} of {adata.n_obs:,}")

    write_reports(metrics, thresholds, Path(args.report_dir))


if __name__ == "__main__":
    main()
