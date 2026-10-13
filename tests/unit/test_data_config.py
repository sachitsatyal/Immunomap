from pathlib import Path

import yaml

CONFIG = yaml.safe_load(Path("configs/data.yaml").read_text())


def test_no_dataset_in_both_reference_and_external() -> None:
    reference = {d["id"] for d in CONFIG["reference_datasets"]}
    external = {d["id"] for d in CONFIG["external_test_datasets"]}
    assert reference.isdisjoint(external)


def test_every_reference_group_has_a_budget() -> None:
    groups = {d["group"] for d in CONFIG["reference_datasets"]}
    assert groups == set(CONFIG["cell_budget"])


def test_census_version_is_pinned() -> None:
    assert CONFIG["census_version"] not in ("stable", "latest")
