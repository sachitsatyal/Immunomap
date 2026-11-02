"""Collect everything the annotator needs into one versioned folder (the "bundle").

    scanvi_coarse/, scanvi_fine/   trained scANVI reference models (scripts/train_scanvi.py)
    calibration_{coarse,fine}.json temperature + Unknown threshold (scripts/calibrate_scanvi.py)
    novelty.joblib                 distance-to-reference detector (same run)
    genes.csv, labels.yaml, qc.yaml, scanvi.yaml
    manifest.json                  versions, provenance and headline test metrics

The bundle is what gets deployed: the demo app and any future API load only this folder.

Usage:
    uv run python scripts/export_bundle.py
"""

import argparse
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import scvi

from immunomap.utils.tracking import TRACKING_URI, data_fingerprint, git_commit

CALIBRATION_EXPERIMENT = "immunomap-scanvi-calibration"
TRAINING_EXPERIMENT = "immunomap-scanvi"


def latest_run(experiment: str, run_name: str) -> Any:
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"attributes.run_name = '{run_name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise SystemExit(f"no finished run '{run_name}' in experiment '{experiment}'")
    return runs.iloc[0]


def fetch(run: Any, artifact: str, dest: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = mlflow.artifacts.download_artifacts(
            run_id=run["run_id"], artifact_path=artifact, dst_path=tmp
        )
        shutil.copy(path, dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", default="data/models/scanvi")
    parser.add_argument("--name", default="immunomap-scanvi-v1")
    parser.add_argument("--out-dir", default="data/models/bundle")
    args = parser.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    out = Path(args.out_dir) / args.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for level in ("coarse", "fine"):
        shutil.copytree(Path(args.models_dir) / f"scanvi_{level}", out / f"scanvi_{level}")
        run = latest_run(CALIBRATION_EXPERIMENT, f"scanvi-{level}-calibrated")
        fetch(run, "calibration.json", out / f"calibration_{level}.json")
    novelty_run = latest_run(CALIBRATION_EXPERIMENT, "scanvi-novelty-full")
    fetch(novelty_run, "reference_distance_scanvi.joblib", out / "novelty.joblib")
    shutil.copy("artifacts/hvg_genes.csv", out / "genes.csv")
    for config in ("labels", "qc", "scanvi"):
        shutil.copy(f"configs/{config}.yaml", out / f"{config}.yaml")

    metrics = {}
    for level in ("coarse", "fine"):
        run = latest_run(TRAINING_EXPERIMENT, f"scanvi-{level}")
        metrics[level] = {
            key.removeprefix("metrics.").removesuffix("_macro_f1"): round(float(value), 4)
            for key, value in run.items()
            if str(key).endswith("_macro_f1") and value == value  # skip NaN
        }
    manifest = {
        "name": args.name,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": "scANVI with scArches reference mapping (scvi-tools)",
        "scvi_tools": scvi.__version__,
        "git_commit": git_commit(),
        "data_md5": data_fingerprint(),
        "census_version": "2025-11-08",
        "macro_f1": metrics,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
    print(f"bundle {out}  ({size:.1f} MB)")
    for path in sorted(out.rglob("*")):
        if path.is_file():
            print(f"  {path.relative_to(out)}  {path.stat().st_size / 1e6:.2f} MB")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
