"""Fine-tune Geneformer V2 with LoRA to classify cells (coarse and fine at once).

- Base model: pretrained Geneformer, frozen; small LoRA adapters are trained inside its
  attention and feed-forward layers (a few % of the parameters).
- Two classification heads on the <cls> token: coarse and fine. One forward pass trains
  both; a cell without a fine label only contributes to the coarse loss.
- Training cells: up to `per_label_cap` per cell type, so rare types are not drowned out.
- Scoring: identical to the other models (immunomap.evaluation.scoring). No reference
  mapping: Geneformer sees external studies as they are.

MLflow experiment immunomap-geneformer; adapters, heads and predicted probabilities are
saved to data/models/geneformer/.

Usage:
    uv run python scripts/finetune_geneformer.py --smoke   # ~3 minutes: tiny data, speed check
    uv run python scripts/finetune_geneformer.py
"""

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import BertModel, get_linear_schedule_with_warmup

from immunomap.evaluation.metrics import classification_metrics
from immunomap.evaluation.scoring import evaluate, log_result
from immunomap.models.baseline import LEVELS
from immunomap.models.geneformer import (
    IGNORE,
    FinetuneConfig,
    TokenStore,
    class_list,
    collate,
    encode_labels,
    make_batches,
    sampling_key,
)
from immunomap.models.novelty import reference_sample
from immunomap.utils.tracking import git_commit, setup_experiment

EXPERIMENT = "immunomap-geneformer"
EVAL_SPLITS = ["val", "test", "external"]
PAD = 0


def build_model(model_dir: Path, cfg: FinetuneConfig, classes: dict[str, list[str]]) -> Any:
    bert = BertModel.from_pretrained(
        model_dir,
        add_pooling_layer=False,
        attn_implementation="sdpa",
        dtype=torch.bfloat16,  # frozen weights in bf16: half the memory traffic
    )
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_targets,
        bias="none",
    )
    base = get_peft_model(bert, lora)
    hidden = bert.config.hidden_size
    heads = nn.ModuleDict(
        {
            level: nn.Sequential(nn.Dropout(cfg.head_dropout), nn.Linear(hidden, len(names)))
            for level, names in classes.items()
        }
    )
    return base.cuda(), heads.cuda()


def forward(base: Any, heads: Any, ids: Any, mask: Any) -> dict[str, Any]:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cls = base(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return {level: head(cls).float() for level, head in heads.items()}


def to_gpu(ids: Any, mask: Any) -> tuple[Any, Any]:
    return torch.from_numpy(ids).cuda(non_blocking=True), torch.from_numpy(mask).cuda()


@torch.no_grad()
def predict(base: Any, heads: Any, store: TokenStore, rows: Any, max_tokens: int) -> dict[str, Any]:
    """Class probabilities per level for the given cell positions (in that order)."""
    base.eval()
    heads.eval()
    out: dict[str, Any] = {}
    for batch in make_batches(store.lengths[rows], max_tokens, 0, rng=None):
        ids, mask = to_gpu(*collate([store.cell(rows[i]) for i in batch], PAD))
        for level, logits in forward(base, heads, ids, mask).items():
            if level not in out:
                out[level] = np.zeros((len(rows), logits.shape[1]), dtype=np.float32)
            out[level][batch] = torch.softmax(logits, dim=1).cpu().numpy()
    base.train()
    heads.train()
    return out


def train(
    base: Any,
    heads: Any,
    store: TokenStore,
    rows: Any,
    labels: dict[str, Any],
    cfg: FinetuneConfig,
    monitor: tuple[TokenStore, Any, dict[str, list[str]]],
) -> None:
    rng = np.random.default_rng(cfg.seed)
    params = [p for p in base.parameters() if p.requires_grad] + list(heads.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    epoch_batches = [
        make_batches(store.lengths[rows], cfg.max_tokens, cfg.megabatch, rng)
        for _ in range(cfg.epochs)
    ]
    total = sum(len(b) for b in epoch_batches)
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, total)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)
    base.train()
    heads.train()
    step, start, cells_seen = 0, time.time(), 0
    for epoch, batches in enumerate(epoch_batches, start=1):
        for batch in batches:
            ids, mask = to_gpu(*collate([store.cell(rows[i]) for i in batch], PAD))
            logits = forward(base, heads, ids, mask)
            loss = sum(
                loss_fn(logits[level], torch.from_numpy(labels[level][batch]).cuda())
                for level in logits
                if (labels[level][batch] != IGNORE).any()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1
            cells_seen += len(batch)
            if step % 100 == 0 or step == total:
                rate = cells_seen / (time.time() - start)
                left = (total - step) * (time.time() - start) / step / 60
                mlflow.log_metrics({"train_loss": loss.item(), "cells_per_s": rate}, step=step)
                print(
                    f"epoch {epoch} step {step:,}/{total:,} loss {loss.item():.3f} "
                    f"{rate:,.0f} cells/s, ~{left:.0f} min left",
                    flush=True,
                )
        store_val, val_rows, classes = monitor
        proba = predict(base, heads, store_val, val_rows, cfg.eval_max_tokens)
        for level in LEVELS:
            y = store_val.obs[level].to_numpy()[val_rows]
            has = np.array([isinstance(v, str) for v in y])
            pred = np.asarray(classes[level])[proba[level].argmax(axis=1)]
            f1 = classification_metrics(y[has].astype(str), pred[has])["macro_f1"]
            mlflow.log_metric(f"monitor_val_{level}_macro_f1", f1, step=epoch)
            print(f"epoch {epoch}: val {level} macro-F1 {f1:.3f} ({has.sum():,} cells)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/geneformer_finetune.yaml")
    parser.add_argument("--model-config", default="configs/geneformer.yaml")
    parser.add_argument("--smoke", action="store_true", help="tiny run to check speed")
    args = parser.parse_args()

    cfg = FinetuneConfig.from_yaml(args.config)
    gcfg = yaml.safe_load(Path(args.model_config).read_text())
    eval_cells = None
    if args.smoke:
        cfg = replace(cfg, per_label_cap=100, epochs=1, warmup_steps=10, val_monitor_cells=1000)
        eval_cells = 2000
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokens_dir = Path(gcfg["tokens_dir"])
    out_dir = Path("data/models") / ("geneformer-smoke" if args.smoke else "geneformer")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_store = TokenStore(tokens_dir, "train")
    stores = {s: TokenStore(tokens_dir, s) for s in EVAL_SPLITS}
    key = sampling_key(train_store.obs)
    labelled = np.flatnonzero([isinstance(k, str) for k in key])
    rows = labelled[reference_sample(key[labelled], cfg.per_label_cap, cfg.seed)]
    classes = {level: class_list(train_store.obs[level]) for level in LEVELS}
    labels = {
        level: encode_labels(train_store.obs[level].to_numpy()[rows], classes[level])
        for level in LEVELS
    }
    rng = np.random.default_rng(cfg.seed)
    val_rows = np.sort(rng.choice(len(stores["val"]), cfg.val_monitor_cells, replace=False))
    eval_rows = {
        s: np.arange(len(st))
        if eval_cells is None
        else np.sort(rng.choice(len(st), min(eval_cells, len(st)), replace=False))
        for s, st in stores.items()
    }
    print(
        f"training on {len(rows):,} cells ({len(classes['coarse'])} coarse, "
        f"{len(classes['fine'])} fine types); median length "
        f"{int(np.median(train_store.lengths[rows])):,} tokens"
    )

    setup_experiment(EXPERIMENT + ("-smoke" if args.smoke else ""))
    model_dir = Path(gcfg["local_dir"]) / gcfg["variant"]
    with mlflow.start_run(run_name="geneformer-lora"):
        mlflow.log_params(
            {
                **asdict(cfg),
                "variant": gcfg["variant"],
                "revision": gcfg["revision"],
                "n_train_cells": len(rows),
                "git_commit": git_commit(),
                "gpu": torch.cuda.get_device_name(0),
            }
        )
        base, heads = build_model(model_dir, cfg, classes)
        trainable = sum(p.numel() for p in base.parameters() if p.requires_grad)
        trainable += sum(p.numel() for p in heads.parameters())
        total = sum(p.numel() for p in base.parameters())
        print(f"trainable parameters {trainable / 1e6:.2f}M of {total / 1e6:.0f}M")
        mlflow.log_params({"trainable_params": trainable})

        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        train(base, heads, train_store, rows, labels, cfg, (stores["val"], val_rows, classes))
        mlflow.log_metrics(
            {
                "train_minutes": (time.time() - start) / 60,
                "peak_gpu_gb": torch.cuda.max_memory_allocated() / 1e9,
            }
        )

        base.save_pretrained(out_dir / "lora")
        torch.save(heads.state_dict(), out_dir / "heads.pt")
        (out_dir / "classes.json").write_text(json.dumps(classes, indent=2))

        summary = []
        for split, store in stores.items():
            start = time.time()
            proba = predict(base, heads, store, eval_rows[split], cfg.eval_max_tokens)
            obs = store.obs.iloc[eval_rows[split]]
            np.savez_compressed(
                out_dir / f"proba_{split}.npz",
                rows=eval_rows[split],
                **{level: p.astype(np.float16) for level, p in proba.items()},
            )
            for level in LEVELS:
                pred = np.asarray(classes[level])[proba[level].argmax(axis=1)]
                result = evaluate(pred, obs, level, split)
                log_result(result)
                summary += [{"level": level, **row} for row in result["rows"]]
            rate = len(eval_rows[split]) / (time.time() - start)
            print(f"{split}: predicted {len(eval_rows[split]):,} cells at {rate:,.0f} cells/s")
        mlflow.log_artifacts(str(out_dir), artifact_path="model")

    print(f"\npeak GPU memory {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    print("\n", pd.DataFrame(summary).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
