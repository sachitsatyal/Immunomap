import numpy as np
import pandas as pd

from immunomap.evaluation.scoring import evaluate


def _obs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "b"],
            "coarse": ["NK", "B", "NK", "B", None],
        }
    )


def test_test_split_is_scored_pooled_and_per_study() -> None:
    result = evaluate(np.array(["NK", "B", "NK", "NK", "B"]), _obs(), "coarse", "test")
    assert result["metrics"]["test_accuracy"] == 0.75  # unlabelled 5th cell is not scored
    assert result["metrics"]["test_a_accuracy"] == 1.0
    assert result["metrics"]["test_b_accuracy"] == 0.5
    assert [r["scope"] for r in result["rows"]] == ["test"]


def test_external_split_is_never_pooled() -> None:
    result = evaluate(np.array(["NK", "B", "NK", "NK", "B"]), _obs(), "coarse", "external")
    assert "external_accuracy" not in result["metrics"]
    assert [r["scope"] for r in result["rows"]] == ["external_a", "external_b"]
    assert set(result["reports"]) == {"external_a", "external_b"}


def test_level_without_labels_returns_empty_result() -> None:
    obs = _obs().assign(fine=None)
    assert evaluate(np.array(["x"] * 5), obs, "fine", "val")["metrics"] == {}
