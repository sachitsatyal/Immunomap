"""Publish ImmunoMap to Hugging Face: the model bundle and the live demo (Space).

The Space's files live in deploy/space/ (committed), so a deployment is reproducible
and GitHub Actions can redeploy without this machine:
    deploy/space/README.md          Space settings (Gradio version, Python 3.12)
    deploy/space/requirements.txt   pinned libraries (CPU PyTorch) + this package from
                                    GitHub at "{commit}", filled in at deploy time
    app.py                          the demo itself (repo root)

Modes:
    --write-space-files   regenerate deploy/space/ from this machine's library versions
                          (run locally after upgrading libraries, then commit)
    (default)             upload the model bundle + example, then the Space (local only:
                          the bundle is not in git)
    --space-only          upload only the Space files (what GitHub Actions runs)
    --wait                wait until the Space is running; fail if its build fails

Needs a Hugging Face write token in the HF_TOKEN environment variable.

Usage:
    uv run python scripts/publish_hf.py --user sawcheet --write-space-files
    uv run python scripts/publish_hf.py --user sawcheet
    python scripts/publish_hf.py --user sawcheet --space-only --commit <sha> --wait
"""

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

GITHUB = "https://github.com/sachitsatyal/Immunomap"
SPACE_DIR = Path("deploy/space")
BUNDLE_NAME = "immunomap-scanvi-v1"
PINNED = ["scvi-tools", "anndata", "scanpy", "scikit-learn", "scipy", "numpy", "pandas"]
PINNED += ["h5py", "pyyaml", "joblib", "huggingface_hub"]


def write_space_files(user: str) -> None:
    """Pin the Space to the library versions installed on this machine."""
    from importlib.metadata import version

    torch = version("torch").split("+")[0]  # local build is +cu128; the Space uses CPU
    requirements = [
        "--extra-index-url https://download.pytorch.org/whl/cpu",
        f"torch=={torch}",
        *[f"{name}=={version(name)}" for name in PINNED],
        f"immunomap @ git+{GITHUB}@{{commit}}",
    ]
    readme = f"""---
title: ImmunoMap
emoji: 🩸
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: {version("gradio")}
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Immune cell-type annotation for blood single-cell RNA-seq
---

Live demo of [ImmunoMap]({GITHUB}): upload a blood (PBMC) scRNA-seq `.h5ad` file with
raw counts and get lineage / coarse / fine cell types with calibrated confidence and an
Unknown flag. Model: scANVI with reference mapping
([model files](https://huggingface.co/{user}/{BUNDLE_NAME})).
Deployed automatically from GitHub by GitHub Actions after CI passes.
"""
    SPACE_DIR.mkdir(parents=True, exist_ok=True)
    (SPACE_DIR / "requirements.txt").write_text("\n".join(requirements) + "\n", encoding="utf-8")
    (SPACE_DIR / "README.md").write_text(readme, encoding="utf-8")
    print(f"wrote {SPACE_DIR}/requirements.txt and README.md: commit them")


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


def publish_model(api: HfApi, user: str, bundle: Path, example: Path, space: str) -> str:
    manifest = json.loads((bundle / "manifest.json").read_text())
    repo = f"{user}/{manifest['name']}"
    api.create_repo(repo, repo_type="model", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        card = Path(tmp) / "README.md"
        card.write_text(model_card(manifest, space), encoding="utf-8")
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo)
    api.upload_folder(folder_path=str(bundle), repo_id=repo, commit_message="Model bundle")
    api.upload_file(
        path_or_fileobj=str(example), path_in_repo=f"examples/{example.name}", repo_id=repo
    )
    print(f"model files: https://huggingface.co/{repo}")
    return repo


def publish_space(api: HfApi, space: str, commit: str) -> None:
    requirements = (SPACE_DIR / "requirements.txt").read_text(encoding="utf-8")
    if "{commit}" not in requirements:
        raise SystemExit(f"{SPACE_DIR}/requirements.txt has no {{commit}} placeholder")
    with tempfile.TemporaryDirectory() as tmp:
        files = {
            "README.md": (SPACE_DIR / "README.md").read_text(encoding="utf-8"),
            "requirements.txt": requirements.replace("{commit}", commit),
            "app.py": Path("app.py").read_text(encoding="utf-8"),
        }
        for name, text in files.items():
            (Path(tmp) / name).write_text(text, encoding="utf-8")
        api.upload_folder(
            folder_path=tmp,
            repo_id=space,
            repo_type="space",
            commit_message=f"Deploy {commit[:8]}",
        )
    print(f"demo: https://huggingface.co/spaces/{space}  (deployed commit {commit[:8]})")


def wait_until_running(api: HfApi, space: str, minutes: int) -> None:
    """Poll the Space; succeed when it runs, fail on a build or runtime error."""
    # The old version keeps RUNNING until the rebuild starts: wait up to 5 minutes for it
    rebuild_deadline = time.time() + 300
    while time.time() < rebuild_deadline and api.get_space_runtime(space).stage == "RUNNING":
        time.sleep(15)
    deadline = time.time() + minutes * 60
    while time.time() < deadline:
        stage = api.get_space_runtime(space).stage
        print(f"space stage: {stage}", flush=True)
        if stage == "RUNNING":
            return
        if stage in {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "NO_APP_FILE"}:
            raise SystemExit(f"Space failed: {stage}. See the Space's Logs tab.")
        time.sleep(30)
    raise SystemExit(f"Space not running after {minutes} minutes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", required=True, help="Hugging Face username")
    parser.add_argument("--space-name", default="immunomap")
    parser.add_argument("--bundle", default=f"data/models/bundle/{BUNDLE_NAME}")
    parser.add_argument("--example", default="data/examples/hao2021_2000cells.h5ad")
    parser.add_argument("--max-cells", default="10000", help="upload cap on the free CPU")
    parser.add_argument("--commit", help="commit to deploy (default: current HEAD)")
    parser.add_argument("--write-space-files", action="store_true")
    parser.add_argument("--space-only", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--wait-minutes", type=int, default=30)
    args = parser.parse_args()

    if args.write_space_files:
        write_space_files(args.user)
        return

    space = f"{args.user}/{args.space_name}"
    commit = (
        args.commit
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    api = HfApi()  # reads the token from HF_TOKEN

    if not args.space_only:
        on_github = subprocess.run(
            ["git", "branch", "-r", "--contains", commit], capture_output=True, text=True
        ).stdout.strip()
        if not on_github:
            raise SystemExit(f"commit {commit[:8]} is not on GitHub yet: push / merge it first")
        model_repo = publish_model(api, args.user, Path(args.bundle), Path(args.example), space)
        api.create_repo(space, repo_type="space", space_sdk="gradio", exist_ok=True)
        api.add_space_variable(space, "IMMUNOMAP_HF_REPO", model_repo)
        api.add_space_variable(space, "IMMUNOMAP_MAX_CELLS", args.max_cells)

    publish_space(api, space, commit)
    if args.wait:
        wait_until_running(api, space, args.wait_minutes)
        print("Space is running")


if __name__ == "__main__":
    main()
