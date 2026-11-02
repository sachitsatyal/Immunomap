"""Publish the model bundle and the live demo to Hugging Face.

1. Model repo  <user>/immunomap-scanvi-v1  : the bundle + the example file + a model card.
2. Space       <user>/immunomap            : app.py, a README with Space settings, and a
   requirements.txt that installs this package from GitHub at the current commit, with
   the same library versions as this machine (CPU PyTorch).

Needs a Hugging Face write token in the HF_TOKEN environment variable, and the current
commit pushed to GitHub (the Space installs the code from there).

Usage:
    $env:HF_TOKEN = "hf_..."      # PowerShell; never commit the token
    uv run python scripts/publish_hf.py --user <hf-username>
"""

import argparse
import json
import subprocess
import tempfile
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import HfApi

GITHUB = "https://github.com/sachitsatyal/Immunomap"
PINNED = ["scvi-tools", "anndata", "scanpy", "scikit-learn", "scipy", "numpy", "pandas"]
PINNED += ["h5py", "pyyaml", "joblib", "huggingface_hub"]


def run(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()


def requirements(commit: str) -> str:
    torch = version("torch").split("+")[0]  # local build is +cu128; the Space uses CPU
    lines = [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        f"torch=={torch}",
        *[f"{name}=={version(name)}" for name in PINNED],
        f"immunomap @ git+{GITHUB}@{commit}",
    ]
    return "\n".join(lines) + "\n"


def space_readme(gradio_version: str, model_repo: str) -> str:
    return f"""---
title: ImmunoMap
emoji: 🩸
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Immune cell-type annotation for blood single-cell RNA-seq
---

Live demo of [ImmunoMap]({GITHUB}): upload a blood (PBMC) scRNA-seq `.h5ad` file with
raw counts and get lineage / coarse / fine cell types with calibrated confidence and an
Unknown flag. Model: scANVI with reference mapping ([model files](https://huggingface.co/{model_repo})).
Code, training pipeline and evaluation: {GITHUB}
"""


def model_card(manifest: dict[str, object], space: str) -> str:
    return f"""---
library_name: scvi-tools
tags: [single-cell, scRNA-seq, cell-type-annotation, scanvi, immunology]
---

# ImmunoMap scANVI bundle ({manifest["name"]})

Everything needed to annotate human blood (PBMC) scRNA-seq with ImmunoMap: two scANVI
reference models (coarse, fine), temperature calibration + Unknown thresholds, a
distance-to-reference novelty detector, the 3,002-gene list, label tree and QC rules.

- Code and evaluation: {GITHUB}
- Live demo: https://huggingface.co/spaces/{space}
- Training data: CELLxGENE Census {manifest["census_version"]} (three reference studies);
  external test studies Hao 2021 and a lupus cohort's healthy controls were never trained on.

Macro-F1 (test = held-out donors; external = unseen studies):

```json
{json.dumps(manifest["macro_f1"], indent=2)}
```

Limitations: human blood only; new cell types close to a known type may be labelled as
that type. Research use only.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="your Hugging Face username")
    parser.add_argument("--bundle", default="data/models/bundle/immunomap-scanvi-v1")
    parser.add_argument("--example", default="data/examples/hao2021_2000cells.h5ad")
    parser.add_argument("--space-name", default="immunomap")
    parser.add_argument("--max-cells", default="10000", help="upload cap on the free CPU")
    args = parser.parse_args()

    commit = run("git", "rev-parse", "HEAD")
    if run("git", "branch", "-r", "--contains", commit) == "":
        raise SystemExit(f"commit {commit[:8]} is not on GitHub yet: push / merge it first")
    bundle = Path(args.bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    model_repo = f"{args.user}/{manifest['name']}"
    space = f"{args.user}/{args.space_name}"
    api = HfApi()  # reads the token from HF_TOKEN

    api.create_repo(model_repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(model_card(manifest, space), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=model_repo)
    api.upload_folder(folder_path=str(bundle), repo_id=model_repo, commit_message="Model bundle")
    api.upload_file(
        path_or_fileobj=args.example,
        path_in_repo=f"examples/{Path(args.example).name}",
        repo_id=model_repo,
    )
    print(f"model files: https://huggingface.co/{model_repo}")

    api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.add_space_variable(space, "IMMUNOMAP_HF_REPO", model_repo)
    api.add_space_variable(space, "IMMUNOMAP_MAX_CELLS", args.max_cells)
    with tempfile.TemporaryDirectory() as tmp:
        files = {
            "README.md": space_readme(version("gradio"), model_repo),
            "requirements.txt": requirements(commit),
            "app.py": Path("app.py").read_text(encoding="utf-8"),
        }
        for name, text in files.items():
            (Path(tmp) / name).write_text(text, encoding="utf-8")
        api.upload_folder(
            folder_path=tmp, repo_id=space, repo_type="space", commit_message=f"Deploy {commit[:8]}"
        )
    print(f"demo: https://huggingface.co/spaces/{space}  (first build takes ~10 minutes)")


if __name__ == "__main__":
    main()
