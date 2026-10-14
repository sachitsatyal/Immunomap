from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from immunomap.data.qc import (
    QCConfig,
    compute_metrics,
    compute_thresholds,
    flag_cells,
)

CFG = QCConfig(
    batch_key="batch",
    min_genes=3,
    min_counts=10,
    max_pct_mt=20,
    n_mads=3,
    n_mads_mt=3,
    remove_high_outliers=True,
)
GENES = ["MT-CO1", "CD3E", "MS4A1", "LYZ", "NKG7", "GNLY"]


def test_metrics_on_sparse_matrix() -> None:
    X = sp.csr_matrix(np.array([[2, 4, 4, 0, 0, 0], [0, 1, 1, 1, 1, 0]], dtype=np.float32))
    m = compute_metrics(X, GENES)
    assert m["total_counts"].tolist() == [10, 4]
    assert m["n_genes"].tolist() == [3, 4]
    assert m["pct_mt"].tolist() == pytest.approx([20.0, 0.0])


def test_metrics_on_dense_matrix_and_lowercase_mt() -> None:
    X = np.array([[5, 5, 0, 0, 0, 0]], dtype=np.float32)
    m = compute_metrics(X, ["mt-co1", "CD3E", "MS4A1", "LYZ", "NKG7", "GNLY"])
    assert m["pct_mt"].iloc[0] == pytest.approx(50.0)


def test_empty_cell_has_zero_pct_mt_not_nan() -> None:
    m = compute_metrics(sp.csr_matrix((1, 6), dtype=np.float32), GENES)
    assert m["pct_mt"].iloc[0] == 0.0


def _metrics(counts: list[float], genes: list[int], mt: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"total_counts": counts, "n_genes": genes, "pct_mt": mt})


def test_constant_batch_has_no_outlier_bounds() -> None:
    m = _metrics([100.0] * 5, [50] * 5, [2.0] * 5)
    t = compute_thresholds(m, pd.Series(["a"] * 5), CFG)
    assert np.isinf(t.loc["a", "log_counts_lo"]) and np.isinf(t.loc["a", "pct_mt_hi"])


def test_hard_rules_and_reason_precedence() -> None:
    m = _metrics(
        counts=[1000.0, 1000.0, 5.0, 1000.0],
        genes=[500, 2, 500, 2],
        mt=[1.0, 1.0, 1.0, 50.0],
    )
    batches = pd.Series(["a"] * 4)
    reasons = flag_cells(m, batches, compute_thresholds(m, batches, CFG), CFG)
    assert reasons.tolist()[:3] == ["pass", "too_few_genes", "too_few_counts"]
    assert reasons.iloc[3] == "too_few_genes"  # also high MT, but the first rule wins


def test_outlier_is_judged_within_its_own_batch() -> None:
    rng = np.random.default_rng(0)
    typical = rng.normal(2000, 200, 200).tolist()
    # 20,000 counts is extreme in batch "a" but normal in batch "b"
    m = _metrics(
        counts=[*typical, 20000.0, *(c * 10 for c in typical), 20000.0],
        genes=[800] * 402,
        mt=[3.0] * 402,
    )
    batches = pd.Series(["a"] * 201 + ["b"] * 201)
    reasons = flag_cells(m, batches, compute_thresholds(m, batches, CFG), CFG)
    assert reasons.iloc[200] == "counts_outlier"
    assert reasons.iloc[401] == "pass"


def test_high_outliers_kept_when_disabled_but_low_still_removed() -> None:
    rng = np.random.default_rng(0)
    typical = rng.normal(2000, 200, 200).tolist()
    m = _metrics(counts=[*typical, 50000.0, 300.0], genes=[800] * 202, mt=[3.0] * 202)
    batches = pd.Series(["a"] * 202)
    cfg = replace(CFG, min_counts=10, remove_high_outliers=False)
    reasons = flag_cells(m, batches, compute_thresholds(m, batches, cfg), cfg)
    assert reasons.iloc[200] == "pass"  # e.g. a plasmablast: very high counts, real biology
    assert reasons.iloc[201] == "counts_outlier"  # unusually low counts are still removed


def test_config_loads_from_repo_yaml() -> None:
    cfg = QCConfig.from_yaml("configs/qc.yaml")
    assert cfg.batch_key == "donor_id" and cfg.min_genes > 0
    assert cfg.remove_high_outliers is False
