import csv
from pathlib import Path

import pytest

from immunomap.data.labels import LabelScheme, ResolvedLabel

SCHEME = LabelScheme.from_yaml("configs/labels.yaml")
INVENTORY = Path("docs/label_inventory_2025-11-08.csv")


def test_every_census_label_is_mapped_or_dropped() -> None:
    with INVENTORY.open() as f:
        ids = {row["cell_type_ontology_term_id"] for row in csv.DictReader(f)}
    for ontology_id in ids:
        SCHEME.resolve(ontology_id)  # raises KeyError if unmapped


def test_fine_label_resolves_full_path() -> None:
    assert SCHEME.resolve("CL:0000895") == ResolvedLabel("T cell", "CD4 T", "CD4 naive T")


def test_leaf_coarse_label_is_its_own_fine_label() -> None:
    assert SCHEME.resolve("CL:0000784") == ResolvedLabel("Myeloid", "pDC", "pDC")


def test_coarse_only_source_has_no_fine_label() -> None:
    assert SCHEME.resolve("CL:0000624") == ResolvedLabel("T cell", "CD4 T", None)


def test_lineage_only_source() -> None:
    assert SCHEME.resolve("CL:0000084") == ResolvedLabel("T cell", None, None)


def test_dropped_label_returns_none() -> None:
    assert SCHEME.resolve("CL:0000542") is None


def test_unknown_label_raises() -> None:
    with pytest.raises(KeyError):
        SCHEME.resolve("CL:9999999")


def test_bad_config_is_rejected() -> None:
    bad = {
        "lineage": ["T cell"],
        "coarse": {"CD4 T": "Nope"},
        "fine": {},
        "source_map": {},
        "drop": [],
    }
    with pytest.raises(ValueError):
        LabelScheme(bad)


def test_fine_label_count() -> None:
    assert len(SCHEME.fine_labels) == 25
