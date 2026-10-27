"""Download Geneformer V2 weights and its gene dictionaries, then check the model runs.

Only the chosen variant's folder and the tokenizer's dictionary files are downloaded
(the full repository holds every model version). The commit hash is printed so it can be
pinned in configs/geneformer.yaml; every later run then uses exactly the same weights.

Usage:
    uv run python scripts/download_geneformer.py
"""

import argparse
import time
from pathlib import Path

import torch
import yaml
from huggingface_hub import HfApi, snapshot_download
from transformers import BertForMaskedLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geneformer.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    revision = cfg["revision"] or HfApi().model_info(cfg["repo_id"]).sha
    local = Path(
        snapshot_download(
            cfg["repo_id"],
            revision=revision,
            allow_patterns=[
                f"{cfg['variant']}/*",
                "geneformer/*.pkl",  # token dictionary, gene medians, gene-ID mapping
                "geneformer/tokenizer.py",  # reference implementation of the tokenizer
            ],
            local_dir=cfg["local_dir"],
        )
    )
    print(f"repo {cfg['repo_id']} at commit {revision}")
    for path in sorted(local.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            print(f"  {path.relative_to(local)}  {path.stat().st_size / 1e6:,.1f} MB")

    model = BertForMaskedLM.from_pretrained(local / cfg["variant"], torch_dtype=torch.bfloat16)
    model = model.cuda().eval()
    c = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"\n{cfg['variant']}: {n_params / 1e6:.0f}M parameters, {c.num_hidden_layers} layers, "
        f"hidden {c.hidden_size}, vocab {c.vocab_size:,}, max input {c.max_position_embeddings:,}"
    )
    print("attention implementation:", getattr(c, "_attn_implementation", "unknown"))

    # Speed check: one batch of 16 cells at the full 4,096-gene input length
    tokens = torch.randint(0, c.vocab_size, (16, c.max_position_embeddings), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(input_ids=tokens)  # warm-up
        torch.cuda.synchronize()
        start = time.time()
        for _ in range(5):
            model(input_ids=tokens)
        torch.cuda.synchronize()
    per_batch = (time.time() - start) / 5
    print(
        f"forward pass: {16 / per_batch:,.0f} cells/s at full length, "
        f"peak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
    )
    if not cfg["revision"]:
        print(f"\nNext: set  revision: {revision}  in {args.config}")


if __name__ == "__main__":
    main()
