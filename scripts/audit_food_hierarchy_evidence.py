#!/usr/bin/env python3
"""Audit authority-backed food hierarchy evidence without inferring relations.

The output distinguishes direct FoodOn/FoodEx2/LanguaL source facets from
lexical or semantic candidates. Only direct FoodOn ``is_a`` relations are
written as hierarchy edges, and a parent may remain external to this snapshot.

Colab:
    python scripts/audit_food_hierarchy_evidence.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foodcomp.ontology import normalize_ontology_id, parse_foodon  # noqa: E402
from foodcomp.util import write_csv, write_json  # noqa: E402


DEFAULT_AUDIT = ROOT / "data" / "processed" / "global_frozen_prediction_panel_food_audit_v3"
DEFAULT_STAGING = ROOT / "data" / "processed" / "scientific_food_composition_v3" / "staging"
DEFAULT_ONTOLOGY = ROOT / "data" / "raw" / "ontology" / "foodon.owl"
DEFAULT_OUTPUT = ROOT / "outputs" / "food_composition_atlas" / "audits"
HIERARCHY_FIELDS = ("foodon_id", "foodex2_code", "langual_code", "taxonomy_id")


def _read_staging_foods(staging_dir: Path) -> pd.DataFrame:
    frames = []
    for source_dir in sorted(path for path in staging_dir.iterdir() if path.is_dir()):
        path = source_dir / "food_observation.csv.gz"
        if not path.exists():
            continue
        frame = pd.read_csv(path, keep_default_na=False, low_memory=False)
        required = {"food_observation_id", "source_key", *HIERARCHY_FIELDS}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing hierarchy fields: {sorted(missing)}")
        frames.append(frame.loc[:, ["food_observation_id", "source_key", *HIERARCHY_FIELDS]])
    if not frames:
        raise FileNotFoundError(f"No source-native food staging tables were found in {staging_dir}")
    return pd.concat(frames, ignore_index=True)


def build_hierarchy_audit(
    audit_dir: Path,
    staging_dir: Path,
    ontology_path: Path,
    output_dir: Path,
) -> tuple[Path, Path]:
    if not ontology_path.exists():
        raise FileNotFoundError(f"FoodOn ontology is required: {ontology_path}")
    group_path = audit_dir / "food_observation_to_exact_name_group.csv.gz"
    if not group_path.exists():
        raise FileNotFoundError(f"Global food audit is required: {group_path}")
    group_map = pd.read_csv(
        group_path,
        usecols=["food_observation_id", "exact_name_group_id", "source_key", "original_name"],
        keep_default_na=False,
    )
    staged = _read_staging_foods(staging_dir)
    food = group_map.merge(
        staged,
        how="left",
        on=["food_observation_id", "source_key"],
        validate="one_to_one",
    )
    for field in HIERARCHY_FIELDS:
        food[field] = food[field].fillna("").astype(str).str.strip()

    coverage = (
        food.groupby("source_key", as_index=False)
        .agg(
            source_food_observations=("food_observation_id", "size"),
            foodon_id_present=("foodon_id", lambda values: int(values.ne("").sum())),
            foodex2_code_present=("foodex2_code", lambda values: int(values.ne("").sum())),
            langual_code_present=("langual_code", lambda values: int(values.ne("").sum())),
            taxonomy_id_present=("taxonomy_id", lambda values: int(values.ne("").sum())),
        )
        .sort_values("source_food_observations", ascending=False, kind="stable")
        .reset_index(drop=True)
    )

    foodon = parse_foodon(ontology_path)
    food["normalized_foodon_id"] = food.foodon_id.map(lambda value: normalize_ontology_id(value, "FOODON"))
    mapped = food.loc[food.normalized_foodon_id.ne("")].copy()
    group_foodon = (
        mapped.groupby("exact_name_group_id", as_index=False)
        .agg(
            child_foodon_id=("normalized_foodon_id", lambda values: ";".join(sorted(set(values)))),
            child_source_keys=("source_key", lambda values: ";".join(sorted(set(values)))),
            child_original_names=("original_name", lambda values: ";".join(sorted(set(values)))),
        )
    )
    single = group_foodon.loc[~group_foodon.child_foodon_id.str.contains(";", regex=False)].copy()
    parent_groups = {
        foodon_id: sorted(group.exact_name_group_id.tolist())
        for foodon_id, group in single.groupby("child_foodon_id", sort=False)
    }
    edges = []
    for row in single.itertuples(index=False):
        for parent_foodon_id in foodon.get(row.child_foodon_id, {}).get("parents", []):
            matched_groups = parent_groups.get(parent_foodon_id, [])
            edges.append(
                {
                    "child_exact_name_group_id": row.exact_name_group_id,
                    "child_foodon_id": row.child_foodon_id,
                    "child_foodon_name": foodon.get(row.child_foodon_id, {}).get("name", ""),
                    "parent_exact_name_group_ids": ";".join(matched_groups),
                    "parent_foodon_id": parent_foodon_id,
                    "parent_foodon_name": foodon.get(parent_foodon_id, {}).get("name", ""),
                    "relation": "is_a",
                    "relation_status": "authority_backed",
                    "relation_evidence": "Source-supplied FoodOn ID and FoodOn owl:subClassOf",
                    "numeric_inheritance_allowed": False,
                    "parent_is_in_current_snapshot": bool(matched_groups),
                }
            )
    edge_frame = pd.DataFrame(edges)
    output_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = output_dir / "food_hierarchy_evidence_coverage.csv"
    edges_path = output_dir / "authority_backed_foodon_relation_candidates.csv"
    manifest_path = output_dir / "food_hierarchy_evidence_manifest.json"
    write_csv(coverage, coverage_path)
    write_csv(edge_frame, edges_path)
    write_json(
        {
            "policy": "Only direct source-supplied FoodOn identifiers and FoodOn is_a relations are emitted. FoodEx2, LanguaL, taxonomy, names and embeddings may support later review but do not create edges automatically.",
            "foodon_linked_observations": int(len(mapped)),
            "exact_name_groups_with_one_foodon_id": int(len(single)),
            "authority_backed_edges": int(len(edge_frame)),
            "edges_with_parent_in_snapshot": int(edge_frame.parent_is_in_current_snapshot.sum()) if not edge_frame.empty else 0,
            "numeric_inheritance_allowed": False,
        },
        manifest_path,
    )
    return coverage_path, edges_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--staging-dir", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--ontology", type=Path, default=DEFAULT_ONTOLOGY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    coverage, edges = build_hierarchy_audit(args.audit_dir, args.staging_dir, args.ontology, args.output_dir)
    print(f"Wrote hierarchy coverage: {coverage}")
    print(f"Wrote authority-backed candidates: {edges}")


if __name__ == "__main__":
    main()
