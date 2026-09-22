"""Preserve benchmark food identities while adding new training evidence."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from .constants import RANDOM_SEED
from .util import stable_id


def attach_new_family_blocks(concepts, mapping, candidates, previous_concepts):
    """New ambiguous names cannot transitively redefine a frozen benchmark."""
    from .harmonize import UnionFind

    result = concepts.copy()
    previous = previous_concepts.set_index("food_concept_id")
    old_ids = set(previous.index) & set(result["food_concept_id"])
    new_ids = set(result["food_concept_id"]) - old_ids
    uf = UnionFind(new_ids)
    old_blocks = previous["family_cluster_id"].to_dict()
    adjacent = {identifier: set() for identifier in new_ids}

    def connect(members):
        new = sorted(set(members) & new_ids)
        old = {old_blocks[x] for x in members if x in old_ids}
        for node in new:
            adjacent[node].update(old)
        for node in new[1:]:
            uf.union(new[0], node)

    for _, group in result.groupby("family_label"):
        connect(group["food_concept_id"].tolist())
    for _, group in mapping[mapping["normalized_lineage"].fillna("").ne("")].groupby("normalized_lineage"):
        connect(group["food_concept_id"].tolist())
    if not candidates.empty:
        for row in candidates.itertuples(index=False):
            connect([row.left_concept_id, row.right_concept_id])
    block_adjacency = {}
    for identifier, blocks in adjacent.items():
        block_adjacency.setdefault(uf.find(identifier), set()).update(blocks)
    assignments, conflicts = {}, set()
    for identifier in new_ids:
        neighbors = block_adjacency[uf.find(identifier)]
        assignments[identifier] = next(iter(neighbors)) if len(neighbors) == 1 else stable_id("new-family", uf.find(identifier))
        if len(neighbors) > 1:
            conflicts.add(identifier)
    assignments.update({identifier: old_blocks[identifier] for identifier in old_ids})
    result["family_cluster_id"] = result["food_concept_id"].map(assignments)
    # An unresolved bridge does not become safe merely by surviving to the next version.
    if "family_expansion_conflict" in previous:
        held = previous["family_expansion_conflict"].astype(str).str.casefold().eq("true")
        conflicts.update(set(previous.index[held]) & old_ids)
    result["family_expansion_conflict"] = result["food_concept_id"].isin(conflicts)
    result["family_block_policy"] = "frozen_v1_blocks_new_multi_block_candidates_quarantined_pending_expert"
    return result


def assign_frozen_partitions(concepts, mapping, observations, eligible_ids, previous):
    locked = previous[previous["partition"].eq("validation")].set_index("food_concept_id")
    locked_ids = set(locked.index)
    if not locked_ids.issubset(set(concepts["food_concept_id"])):
        raise ValueError("Frozen validation identities are missing; never substitute new foods")
    result = concepts[concepts["food_concept_id"].isin(set(eligible_ids) | locked_ids)].copy()
    family_locked_ids = set(locked.index[locked["validation_panel"].eq("family_holdout")])
    source_locked_ids = locked_ids - family_locked_ids
    family_blocks = set(result.loc[result["food_concept_id"].isin(family_locked_ids), "family_cluster_id"])
    source_blocks = set(result.loc[result["food_concept_id"].isin(source_locked_ids), "family_cluster_id"])
    source_rows = mapping.merge(observations[["food_observation_id", "source_key"]], on="food_observation_id", validate="one_to_one")
    held_lineages = set(source_rows.loc[source_rows["food_concept_id"].isin(locked_ids), "normalized_lineage"].dropna()) - {""}
    lineage_copies = set(source_rows.loc[source_rows["normalized_lineage"].isin(held_lineages), "food_concept_id"])
    additional_foundation = set(source_rows.loc[source_rows["source_key"].eq("usda_foundation"), "food_concept_id"]) - locked_ids
    value_lineage_copies = set()
    if "potential_reference_lineages" in observations:
        linked = mapping.merge(observations[["food_observation_id", "source_key", "source_food_id", "potential_reference_lineages"]], on="food_observation_id", validate="one_to_one")
        held_usda = linked[linked["food_concept_id"].isin(locked_ids) & linked["source_key"].isin(["usda_sr_legacy", "usda_foundation"])]
        held_fdc = {"USDA:FDC:" + str(x).split(".")[0] for x in held_usda["source_food_id"]}
        borrowed = linked["potential_reference_lineages"].fillna("").map(lambda x: bool(set(str(x).split(";")) & held_fdc))
        value_lineage_copies = set(linked.loc[borrowed, "food_concept_id"])
    new_ids = ~result["food_concept_id"].isin(locked_ids)
    unsafe = new_ids & (
        result["family_cluster_id"].isin(family_blocks)
        | result["food_concept_id"].isin(lineage_copies | additional_foundation | value_lineage_copies)
        | result["ml_exclusion_reason"].fillna("").ne("")
    )
    if "family_expansion_conflict" in result:
        unsafe |= new_ids & result["family_expansion_conflict"].eq(True)
    exclusions = result.loc[unsafe, ["food_concept_id", "canonical_name", "family_cluster_id"]].copy()
    exclusions["reason"] = "frozen_family_or_lineage_block_or_food_exclusion"
    if "family_expansion_conflict" in result:
        conflicts = set(result.loc[result["family_expansion_conflict"].eq(True), "food_concept_id"])
        exclusions.loc[exclusions["food_concept_id"].isin(conflicts), "reason"] = "new_food_links_multiple_frozen_families_requires_expert_review"
    exclusions.loc[exclusions["food_concept_id"].isin(value_lineage_copies), "reason"] = "AFCD_sampling_reference_mentions_locked_USDA_food"
    result = result[~unsafe].copy()
    result["partition"] = np.where(result["food_concept_id"].isin(locked_ids), "validation", "train")
    result["validation_panel"] = result["food_concept_id"].map(locked["validation_panel"]).fillna("")
    result["family_holdout"] = result["validation_panel"].eq("family_holdout")
    result["source_holdout"] = result["validation_panel"].eq("source_holdout")
    result["source_holdout_reason"] = np.where(result["source_holdout"], "frozen_v1_source_holdout_identity", "")
    result["source_holdout_family_overlap"] = result["partition"].eq("train") & result["family_cluster_id"].isin(source_blocks)
    result["benchmark_eligible"] = True
    result["cv_fold"] = result["family_cluster_id"].map(
        lambda family: int(hashlib.sha256(f"{RANDOM_SEED}:{family}".encode()).hexdigest()[:8], 16) % 5
    )
    result.loc[result["partition"].eq("validation"), "cv_fold"] = -1
    result["split_policy"] = "v2_frozen_validation_identities_new_related_foods_quarantined"
    result["frozen_validation_identity"] = result["food_concept_id"].isin(locked_ids)
    if set(result.loc[result["partition"].eq("validation"), "food_concept_id"]) != locked_ids:
        raise AssertionError("Validation identity set changed")
    return result.sort_values("food_concept_id").reset_index(drop=True), exclusions
