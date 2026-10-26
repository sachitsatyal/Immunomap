import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from immunomap.evaluation.metrics import classification_metrics, per_class_report
from immunomap.models.baseline import BaselineConfig, build_pipeline, labelled_cells

CFG = BaselineConfig(alpha=1e-4, max_iter=50, tol=1e-4, class_weight="balanced", n_jobs=1, seed=0)


def test_labelled_cells_skips_cells_without_a_label_at_that_level() -> None:
    obs = pd.DataFrame({"coarse": ["NK", "B", "NK"], "fine": ["NK CD56dim", None, None]})
    idx, labels = labelled_cells(obs, "fine")
    assert idx.tolist() == [0]
    assert labels.tolist() == ["NK CD56dim"]
    idx, _ = labelled_cells(obs, "coarse")
    assert idx.tolist() == [0, 1, 2]


def test_unknown_level_is_rejected() -> None:
    with pytest.raises(ValueError):
        labelled_cells(pd.DataFrame({"coarse": ["NK"]}), "lineage_typo")


def test_pipeline_learns_separable_sparse_data() -> None:
    rng = np.random.default_rng(0)
    # Two "cell types", each expressing its own marker gene strongly
    a = np.hstack([rng.poisson(5, (100, 1)), rng.poisson(0.1, (100, 1)), rng.poisson(1, (100, 3))])
    b = np.hstack([rng.poisson(0.1, (100, 1)), rng.poisson(5, (100, 1)), rng.poisson(1, (100, 3))])
    X = sp.csr_matrix(np.log1p(np.vstack([a, b]).astype(np.float32)))
    y = np.array(["T"] * 100 + ["B"] * 100)
    model = build_pipeline(CFG).fit(X, y)
    assert (model.predict(X) == y).mean() > 0.95
    assert model.predict_proba(X).shape == (200, 2)


def test_config_loads_from_repo_yaml() -> None:
    cfg = BaselineConfig.from_yaml("configs/baseline.yaml")
    assert isinstance(cfg.alpha, float) and cfg.alpha > 0
    assert cfg.class_weight == "balanced"


def test_perfect_predictions_score_one() -> None:
    y = np.array(["A", "B", "B", "C"])
    assert classification_metrics(y, y) == pytest.approx(
        {"accuracy": 1.0, "balanced_accuracy": 1.0, "macro_f1": 1.0, "weighted_f1": 1.0}
    )


def test_macro_f1_ignores_classes_absent_from_truth() -> None:
    y_true = np.array(["A", "A", "B", "B"])
    y_pred = np.array(["A", "A", "B", "C"])  # "C" is a wrong prediction, not a scored class
    # A: F1 = 1.0; B: precision 1, recall 0.5 -> F1 = 2/3; macro over {A, B}
    assert classification_metrics(y_true, y_pred)["macro_f1"] == pytest.approx((1 + 2 / 3) / 2)


def test_per_class_report_lists_true_classes_worst_first() -> None:
    report = per_class_report(np.array(["A", "A", "B", "B"]), np.array(["A", "A", "B", "C"]))
    assert report.index.tolist() == ["B", "A"]
    assert report["support"].tolist() == [2, 2]
