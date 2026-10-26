"""MLflow setup and provenance shared by every training script."""

import os
import subprocess
from pathlib import Path

import mlflow
import yaml

TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")


def setup_experiment(name: str) -> None:
    """Point MLflow at the local database and select (or create) an experiment."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=Path("mlartifacts").resolve().as_uri())
    mlflow.set_experiment(name)


def data_fingerprint() -> str:
    """DVC hash of the model inputs: ties every run to the exact data it used."""
    lock = yaml.safe_load(Path("dvc.lock").read_text())
    return str(lock["stages"]["build_features"]["outs"][0]["md5"])


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() or "unknown"
