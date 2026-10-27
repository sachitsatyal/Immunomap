import numpy as np
import scipy.sparse as sp

from immunomap.evaluation.calibration_metrics import unknown_detection_auroc
from immunomap.models.novelty import NoveltyConfig, ReferenceDistance, reference_sample

CFG = NoveltyConfig(
    n_components=5, n_neighbors=5, ref_per_type=100, quantile=0.99, n_jobs=1, seed=0
)


def _cluster(center: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.clip(center + rng.normal(0, 0.3, (n, center.size)), 0, None)


def test_reference_sample_caps_common_types_and_keeps_rare_ones() -> None:
    labels = np.array(["T"] * 500 + ["pDC"] * 20)
    picked = labels[reference_sample(labels, per_type=100, seed=0)]
    assert (picked == "T").sum() == 100
    assert (picked == "pDC").sum() == 20


def test_unseen_cluster_is_far_from_reference() -> None:
    rng = np.random.default_rng(0)
    centers = rng.uniform(0, 4, (3, 20))
    reference = np.vstack([_cluster(centers[0], 200, rng), _cluster(centers[1], 200, rng)])
    detector = ReferenceDistance(CFG).fit(sp.csr_matrix(reference))

    known = np.vstack([_cluster(centers[0], 50, rng), _cluster(centers[1], 50, rng)])
    unseen = _cluster(centers[2], 50, rng)
    scores = detector.score(sp.csr_matrix(np.vstack([known, unseen])))
    is_unseen = np.r_[np.zeros(100), np.ones(50)]
    assert unknown_detection_auroc(-scores, is_unseen) > 0.95


def test_threshold_flags_about_the_chosen_fraction_of_known_cells() -> None:
    rng = np.random.default_rng(1)
    detector = ReferenceDistance(CFG)
    known_scores = rng.normal(1.0, 0.1, 10_000)
    detector.set_threshold(known_scores)
    assert abs(detector.is_novel(known_scores).mean() - 0.01) < 0.002


def test_config_loads_from_repo_yaml() -> None:
    cfg = NoveltyConfig.from_yaml("configs/novelty.yaml")
    assert cfg.n_neighbors > 0 and 0.5 < cfg.quantile < 1.0


def test_precomputed_latent_space_is_used_as_is() -> None:
    rng = np.random.default_rng(2)
    reference = rng.normal(0, 1, (300, 10))
    detector = ReferenceDistance(CFG, embed=False).fit(reference)
    near = detector.score(rng.normal(0, 1, (50, 10)))
    far = detector.score(rng.normal(8, 1, (50, 10)))
    assert near.max() < far.min()
