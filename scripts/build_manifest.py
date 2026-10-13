"""Build the cell manifest: which Census cells ImmunoMap uses, their split and labels.

Metadata only, no expression data. Runs on Linux (Colab or WSL), because
cellxgene-census has no Windows build. Output is deterministic for a given
config: same Census version + seed gives the same manifest.
"""

import argparse
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    from immunomap.data.labels import LabelScheme
except ImportError:  # Colab: labels.py uploaded next to this script
    from labels import LabelScheme  # type: ignore[no-redef]

OBS_COLUMNS = ["soma_joinid", "dataset_id", "donor_id", "assay", "cell_type_ontology_term_id"]
MANIFEST_COLUMNS = [
    "soma_joinid", "group", "dataset_id", "donor_id", "assay",
    "split", "lineage", "coarse", "fine",
]  # fmt: skip


def read_obs(cfg: dict[str, Any], dataset_ids: list[str]) -> pd.DataFrame:
    import cellxgene_census  # Linux only

    value_filter = f"{cfg['base_filter']} and dataset_id in {dataset_ids}"
    with cellxgene_census.open_soma(census_version=cfg["census_version"]) as census:
        obs = (
            census["census_data"][cfg["organism"]]
            .obs.read(value_filter=value_filter, column_names=OBS_COLUMNS)
            .concat()
            .to_pandas()
        )
    obs = obs[obs["assay"].isin(cfg["assays"])]
    return obs.astype({c: str for c in OBS_COLUMNS[1:]})


def add_labels(obs: pd.DataFrame, scheme: LabelScheme) -> pd.DataFrame:
    lookup: dict[str, tuple[str | None, str | None, str | None]] = {}
    for cell_type in obs["cell_type_ontology_term_id"].unique():
        resolved = scheme.resolve(cell_type)  # raises on an unmapped label
        lookup[cell_type] = (
            (None, None, None)
            if resolved is None
            else (resolved.lineage, resolved.coarse, resolved.fine)
        )
    labels = pd.DataFrame(
        obs["cell_type_ontology_term_id"].map(lookup).tolist(),
        index=obs.index,
        columns=["lineage", "coarse", "fine"],
    )
    obs = obs.join(labels)
    return obs[obs["lineage"].notna()]


def assign_splits(
    donors: Iterable[str], group: str, seed: int, fractions: dict[str, float]
) -> dict[str, str]:
    """Deterministically assign whole donors to test / val / train."""
    order = sorted(donors, key=lambda d: hashlib.sha256(f"{seed}:{group}:{d}".encode()).hexdigest())
    n_test = max(1, round(fractions["test"] * len(order)))
    n_val = max(1, round(fractions["val"] * len(order)))
    return {
        donor: "test" if i < n_test else "val" if i < n_test + n_val else "train"
        for i, donor in enumerate(order)
    }


def sample_per_donor(df: pd.DataFrame, n_cells: int, seed: int) -> pd.DataFrame:
    per_donor = max(0, n_cells) // df["donor_id"].nunique()
    parts = [
        g.sample(n=min(len(g), per_donor), random_state=seed) for _, g in df.groupby("donor_id")
    ]
    return pd.concat(parts)


def sample_reference(df: pd.DataFrame, budget: int, rare_threshold: int, seed: int) -> pd.DataFrame:
    """Keep every cell of rare labels; sample the rest equally per donor."""
    stratum = df["fine"].fillna(df["coarse"]).fillna(df["lineage"])
    counts = stratum.value_counts()
    is_rare = stratum.isin(counts[counts < rare_threshold].index)
    kept = df[is_rare]
    sampled = sample_per_donor(df[~is_rare], budget - len(kept), seed)
    return pd.concat([kept, sampled])


def build_manifest(obs: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    seed = cfg["seed"]
    groups = {d["id"]: d["group"] for d in cfg["reference_datasets"]}
    groups |= {d["id"]: d["group"] for d in cfg["external_test_datasets"]}
    obs = obs.assign(group=obs["dataset_id"].map(groups))

    parts = []
    for group, budget in cfg["cell_budget"].items():
        df = obs[obs["group"] == group]
        splits = assign_splits(df["donor_id"].unique(), group, seed, cfg["split_fractions"])
        df = df.assign(split=df["donor_id"].map(splits))
        parts.append(sample_reference(df, budget, cfg["rare_label_threshold"], seed))
    for ds in cfg["external_test_datasets"]:
        df = obs[obs["dataset_id"] == ds["id"]].assign(split="external")
        if "max_cells" in ds:
            df = sample_per_donor(df, ds["max_cells"], seed)
        parts.append(df)

    manifest = pd.concat(parts)[MANIFEST_COLUMNS]
    manifest = manifest.sort_values("soma_joinid").reset_index(drop=True)

    if not manifest["soma_joinid"].is_unique:
        raise RuntimeError("a cell was selected twice")
    if (manifest.groupby(["group", "donor_id"])["split"].nunique() > 1).any():
        raise RuntimeError("a donor appears in more than one split")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--label-config", default="configs/labels.yaml")
    parser.add_argument("--out", default="manifests")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.data_config).read_text())
    scheme = LabelScheme.from_yaml(args.label_config)
    ids = [d["id"] for d in [*cfg["reference_datasets"], *cfg["external_test_datasets"]]]

    obs = add_labels(read_obs(cfg, ids), scheme)
    manifest = build_manifest(obs, cfg)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(out / f"manifest_{cfg['census_version']}.parquet", index=False)

    summary = manifest.groupby(["group", "split"]).agg(
        cells=("soma_joinid", "size"), donors=("donor_id", "nunique")
    )
    fine_counts = (
        manifest["fine"]
        .fillna("(no fine label)")
        .groupby(manifest["split"])
        .value_counts()
        .unstack("split", fill_value=0)
    )
    summary.to_csv(out / "manifest_summary.csv")
    fine_counts.to_csv(out / "manifest_fine_label_counts.csv")
    print(summary.to_string(), "\n")
    print(fine_counts.to_string())
    print(f"\nTotal cells: {len(manifest):,}")


if __name__ == "__main__":
    main()
