from pathlib import Path

import numpy as np
import pandas as pd

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


def test_sampling_key_prefers_fine_then_coarse() -> None:
    obs = pd.DataFrame({"coarse": ["NK", "B", None], "fine": ["NK CD56dim", None, None]})
    assert sampling_key(obs).tolist() == ["NK CD56dim", "B", None]


def test_labels_missing_or_unknown_are_ignored() -> None:
    classes = class_list(["NK", "B", None, float("nan"), "NK"])
    assert classes == ["B", "NK"]
    assert encode_labels(["NK", None, "B", "pDC"], classes).tolist() == [1, IGNORE, 0, IGNORE]


def test_training_batches_cover_every_cell_once_within_token_budget() -> None:
    rng = np.random.default_rng(0)
    lengths = rng.integers(50, 4096, 1000)
    batches = make_batches(lengths, max_tokens=8192, megabatch=128, rng=rng)
    cells = np.concatenate(batches)
    assert sorted(cells.tolist()) == list(range(1000))
    for batch in batches:
        assert len(batch) == 1 or len(batch) * lengths[batch].max() <= 8192


def test_prediction_batches_are_deterministic_and_length_sorted() -> None:
    lengths = np.array([5, 100, 7, 100, 3])
    first = make_batches(lengths, max_tokens=200, megabatch=0, rng=None)
    second = make_batches(lengths, max_tokens=200, megabatch=0, rng=None)
    assert [b.tolist() for b in first] == [b.tolist() for b in second]
    assert first[0].tolist() == [1, 3]  # the two longest cells together


def test_collate_pads_and_masks() -> None:
    ids, mask = collate([np.array([2, 7, 3]), np.array([2, 3])], pad_token=0)
    assert ids.tolist() == [[2, 7, 3], [2, 3, 0]]
    assert mask.tolist() == [[1, 1, 1], [1, 1, 0]]


def test_token_store_reads_cells(tmp_path: Path) -> None:
    np.save(tmp_path / "val_tokens.npy", np.array([2, 9, 3, 2, 3], dtype=np.int16))
    np.save(tmp_path / "val_offsets.npy", np.array([0, 3, 5]))
    pd.DataFrame({"cell_id": ["a", "b"], "coarse": ["NK", "B"]}).to_csv(
        tmp_path / "val_obs.csv", index=False
    )
    store = TokenStore(tmp_path, "val")
    assert len(store) == 2
    assert store.lengths.tolist() == [3, 2]
    assert store.cell(0).tolist() == [2, 9, 3]


def test_config_loads_from_repo_yaml() -> None:
    cfg = FinetuneConfig.from_yaml("configs/geneformer_finetune.yaml")
    assert cfg.lora_r > 0 and cfg.max_tokens >= 4096
