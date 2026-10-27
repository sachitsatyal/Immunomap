"""Distance-to-reference novelty detection: the second half of the "Unknown" flag.

A classifier always picks the closest known type, often confidently, even for a cell
type it has never seen. This detector asks a different question: is the cell close to
ANY reference cell? Cells embedded with PCA (via SVD) whose average distance to their
k nearest reference cells is larger than almost all known validation cells are novel.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class NoveltyConfig:
    n_components: int  # PCA dimensions the distances are measured in
    n_neighbors: int  # k: novelty score = mean distance to the k nearest reference cells
    ref_per_type: int  # reference cells per cell type (rare types are kept whole)
    quantile: (
        float  # threshold = this quantile of known validation scores (0.99 -> ~1% false flags)
    )
    n_jobs: int
    seed: int

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls(**yaml.safe_load(Path(path).read_text()))


def reference_sample(labels: Any, per_type: int, seed: int) -> Any:
    """Positions of up to `per_type` cells of each label, so every type is well represented."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    picks = []
    for label in np.unique(labels):
        pos = np.flatnonzero(labels == label)
        picks.append(rng.choice(pos, per_type, replace=False) if len(pos) > per_type else pos)
    return np.sort(np.concatenate(picks))


class ReferenceDistance:
    """embed=True: cells are expression (embedded with PCA here).
    embed=False: cells are already embedded, e.g. a scANVI latent space."""

    def __init__(self, cfg: NoveltyConfig, embed: bool = True) -> None:
        self.cfg = cfg
        self.threshold = float("inf")
        self.embed = (
            make_pipeline(
                StandardScaler(with_mean=False),
                TruncatedSVD(n_components=cfg.n_components, random_state=cfg.seed),
            )
            if embed
            else None
        )

    def _space(self, X: Any) -> Any:
        return X if self.embed is None else self.embed.transform(X)

    def fit(self, X_reference: Any) -> Self:
        if self.embed is not None:
            self.embed.fit(X_reference)
        self.neighbors = NearestNeighbors(
            n_neighbors=self.cfg.n_neighbors, algorithm="brute", n_jobs=self.cfg.n_jobs
        ).fit(self._space(X_reference))
        return self

    def score(self, X: Any) -> Any:
        """Mean distance to the k nearest reference cells (higher = more novel)."""
        distances, _ = self.neighbors.kneighbors(self._space(X))
        return distances.mean(axis=1)

    def set_threshold(self, known_scores: Any) -> float:
        self.threshold = float(np.quantile(known_scores, self.cfg.quantile))
        return self.threshold

    def is_novel(self, scores: Any) -> Any:
        return np.asarray(scores) > self.threshold
