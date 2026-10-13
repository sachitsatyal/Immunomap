"""Cell-type label tree: map study labels to lineage / coarse / fine."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import yaml


@dataclass(frozen=True)
class ResolvedLabel:
    lineage: str
    coarse: str | None
    fine: str | None


class LabelScheme:
    def __init__(self, config: dict[str, Any]) -> None:
        self.lineages: list[str] = list(config["lineage"])
        self.coarse_parent: dict[str, str] = dict(config["coarse"])
        self.fine_parent: dict[str, str] = dict(config["fine"])
        self.source_map: dict[str, str] = dict(config["source_map"])
        self.drop: set[str] = set(config["drop"])
        self._validate()

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        return cls(yaml.safe_load(Path(path).read_text()))

    def _validate(self) -> None:
        for coarse, parent in self.coarse_parent.items():
            if parent not in self.lineages:
                raise ValueError(f"coarse '{coarse}' has unknown lineage '{parent}'")
        for fine, parent in self.fine_parent.items():
            if parent not in self.coarse_parent:
                raise ValueError(f"fine '{fine}' has unknown coarse parent '{parent}'")
        names = [*self.lineages, *self.coarse_parent, *self.fine_parent]
        if len(names) != len(set(names)):
            raise ValueError("a label name appears at more than one level")
        known = set(names)
        for source, target in self.source_map.items():
            if target not in known:
                raise ValueError(f"{source} maps to unknown label '{target}'")
        overlap = set(self.source_map) & self.drop
        if overlap:
            raise ValueError(f"labels both mapped and dropped: {sorted(overlap)}")

    @property
    def fine_labels(self) -> list[str]:
        has_children = set(self.fine_parent.values())
        leaves = [c for c in self.coarse_parent if c not in has_children]
        return [*self.fine_parent, *leaves]

    def resolve(self, ontology_id: str) -> ResolvedLabel | None:
        """Return the labels a source cell type supports, or None if it is dropped."""
        if ontology_id in self.drop:
            return None
        if ontology_id not in self.source_map:
            raise KeyError(f"unmapped cell type: {ontology_id}")
        node = self.source_map[ontology_id]
        if node in self.fine_parent:
            coarse = self.fine_parent[node]
            return ResolvedLabel(self.coarse_parent[coarse], coarse, node)
        if node in self.coarse_parent:
            has_children = node in set(self.fine_parent.values())
            return ResolvedLabel(self.coarse_parent[node], node, None if has_children else node)
        return ResolvedLabel(node, None, None)
