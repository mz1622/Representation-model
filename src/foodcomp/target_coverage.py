"""Plan nutrition target coverage without weakening label or split evidence."""

from __future__ import annotations

from decimal import Decimal
import hashlib
import re

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack

from .harmonize import UnionFind
from .source_policy import validation_reference_mask


CORE_ROLES = {
    "macronutrient", "micronutrient_mineral", "micronutrient_vitamin",
    "essential_nutrient_choline",
}
FORM_ROLES = {
    "nutrient_constituent_fatty_acid", "nutrient_constituent_amino_acid",
    "nutrient_constituent_carbohydrate",
}


def _true(series):
    return series.astype(str).str.casefold().eq("true")


def provisional_scope(role):
    if role in CORE_ROLES:
        return "core_nutrition_expression"
    if role in FORM_ROLES:
        return "nutrient_chemical_form"
    return "other_composition_scope_review"


def component_identity_collisions(observations, mapping):
    """Detect a decimal USDA identifier folded into a different source analyte."""
    linked = observations.merge(mapping, on="component_observation_id", validate="one_to_one")
    linked["original_nutrient_number"] = linked.source_definition.fillna("").str.extract(
        r"(?:number|code)\s+([0-9]+(?:\.[0-9]+)?)", flags=re.I, expand=False
    )
    numbered = linked[linked.source_key.isin(["usda_sr_legacy", "usda_foundation", "fndds", "cnf"])]
    rows = []
    for identifier, group in numbered.groupby("component_concept_id"):
        numbers = {format(Decimal(number).normalize(), "f") for number in group.original_nutrient_number.dropna()}
        if len(numbers) > 1 and any("." in number for number in numbers):
            for row in group.itertuples(index=False):
                rows.append({
                    "component_concept_id": identifier,
                    "component_observation_id": row.component_observation_id,
                    "source_key": row.source_key, "original_name": row.original_name,
                    "original_nutrient_number": row.original_nutrient_number,
                    "reason": "distinct_decimal_nutrient_numbers_share_canonical_identity",
                })
    return pd.DataFrame(rows, columns=[
        "component_concept_id", "component_observation_id", "source_key", "original_name",
        "original_nutrient_number", "reason",
    ])


def profile_evidence(profiles, measurements):
    """Require every selected record to pass its versioned validation policy.

    The legacy output name ``strict_label_eligible`` is retained for report
    compatibility; with a source-trust policy it means reference eligibility,
    not individual analytical certification.
    """
    accepted = profiles[profiles.aggregation_status.eq("accepted") & profiles.canonical_value_g_per_100g.notna()].copy()
    if accepted.duplicated(["food_concept_id", "component_concept_id"]).any():
        raise ValueError("Canonical food/component profiles are not unique")
    if measurements.measurement_id.duplicated().any():
        raise ValueError("Measurement IDs are not unique")
    if not _true(measurements.main_value_eligible).all():
        raise ValueError("Ineligible measurements appear in the admitted evidence table")
    strict = set(measurements.loc[validation_reference_mask(measurements), "measurement_id"])
    all_ids = set(measurements.measurement_id)
    used_ids = accepted.measurement_ids.fillna("").str.split(";")
    if not used_ids.map(lambda ids: bool(ids) and "" not in ids and set(ids) <= all_ids).all():
        raise ValueError("A canonical profile has missing measurement provenance")
    accepted["strict_label_eligible"] = used_ids.map(lambda ids: set(ids) <= strict)
    if not accepted.loc[accepted.partition.eq("validation"), "strict_label_eligible"].all():
        raise ValueError("Existing validation includes a profile rejected by its versioned source policy")
    return accepted


def make_transfer_blocks(foods, mapping, observations, *, selected_observation_ids=None):
    """Move entire train families, plus any linked copied/borrowed lineages."""
    train_ids = set(foods.loc[foods.partition.eq("train"), "food_concept_id"])
    val_ids = set(foods.loc[foods.partition.eq("validation"), "food_concept_id"])
    uf = UnionFind(train_ids)
    blocked = set()

    def connect(identifiers, block_if_validation=False):
        members = set(identifiers)
        training = sorted(members & train_ids)
        for identifier in training[1:]:
            uf.union(training[0], identifier)
        if block_if_validation and members & val_ids:
            blocked.update(training)

    for _, group in foods.groupby("family_cluster_id"):
        # Source-holdout relatives in train remain allowed by the v3 policy.
        connect(group.food_concept_id)
    joined = mapping.merge(
        observations[["food_observation_id", "source_key", "source_food_id", "potential_reference_lineages"]],
        on="food_observation_id", validate="one_to_one",
    )
    for _, group in joined[joined.normalized_lineage.fillna("").ne("")].groupby("normalized_lineage"):
        connect(group.food_concept_id, block_if_validation=True)
    fdc_to_concepts = {}
    usda = joined[joined.source_key.isin(["usda_sr_legacy", "usda_foundation"])]
    for row in usda.itertuples(index=False):
        fdc = "USDA:FDC:" + str(row.source_food_id).split(".")[0]
        fdc_to_concepts.setdefault(fdc, set()).add(row.food_concept_id)
    reference_rows = joined[joined.potential_reference_lineages.fillna("").ne("")]
    # An archived observation that supplies no released value is not a copied
    # validation label. Preserve its metadata, but do not invent a label link.
    if selected_observation_ids is not None:
        reference_rows = reference_rows[reference_rows.food_observation_id.isin(selected_observation_ids)]
    for row in reference_rows.itertuples(index=False):
        for reference in str(row.potential_reference_lineages).split(";"):
            connect({row.food_concept_id} | fdc_to_concepts.get(reference, set()), block_if_validation=True)
    if blocked:
        raise ValueError(f"Existing training contains {len(blocked)} linked validation copies; repair before reallocating")
    assignments = {identifier: uf.find(identifier) for identifier in sorted(train_ids)}
    return assignments


def coverage_counts(registry, profiles, foods, block_assignments):
    """Audit all represented expressions, including previously excluded nutrients."""
    represented = set(profiles.component_concept_id)
    ledger = registry[registry.component_concept_id.isin(represented) | registry.training_role.ne("excluded")].copy()
    ledger["nutrition_scope"] = ledger.nutritional_role.map(provisional_scope)
    ledger["classification_is_provisional"] = True
    ledger = ledger.set_index("component_concept_id")
    train = profiles[profiles.partition.eq("train")].copy()
    validation = profiles[profiles.partition.eq("validation")]
    train["transfer_block"] = train.food_concept_id.map(block_assignments)
    if train.transfer_block.isna().any():
        raise ValueError("A training profile has no transfer block")
    for name, frame in (("current_train_count", train), ("current_validation_count", validation),
                        ("strict_train_count", train[train.strict_label_eligible])):
        counts = frame.groupby("component_concept_id").food_concept_id.nunique()
        ledger[name] = counts.reindex(ledger.index).fillna(0).astype(int)
    ledger["total_count"] = ledger.current_train_count + ledger.current_validation_count
    families = train.groupby("component_concept_id").transfer_block.nunique()
    ledger["train_independent_blocks"] = families.reindex(ledger.index).fillna(0).astype(int)
    ledger["validation_deficit"] = (30 - ledger.current_validation_count).clip(lower=0)
    ledger["train_count_after_minimum_transfer"] = ledger.current_train_count - ledger.validation_deficit
    ledger["train_count_sufficient_before_transfer"] = ledger.current_train_count.ge(100)
    ledger["validation_count_sufficient_before_transfer"] = ledger.current_validation_count.ge(30)
    ledger["strict_labels_sufficient_for_transfer"] = ledger.strict_train_count.ge(ledger.validation_deficit)
    ledger["minimum_counts_feasible_without_grouping"] = (
        ledger.train_count_after_minimum_transfer.ge(100) & ledger.strict_labels_sufficient_for_transfer
    )
    return ledger.reset_index(), train


def eligibility_issues(ledger, collision_ids):
    """Record overlapping blockers, not just the first failed threshold."""
    result = ledger.copy()
    flags = []
    for row in result.itertuples(index=False):
        issues = []
        if row.component_concept_id in collision_ids:
            issues.append("canonical_identity_collision")
        if row.identity_status not in {"authority_verified", "stable_source_identity_pending_cross_database_review"}:
            issues.append("identity_not_accepted_by_existing_policy")
        if row.expression_variant in {"label_expression", "biological_equivalent", "calculated_from_constituents", "calculated_by_difference"}:
            issues.append("non_primary_or_calculated_expression_review")
        if not np.isfinite(row.capability_adjusted_coverage) or row.capability_adjusted_coverage < 0.02:
            issues.append("coverage_below_existing_2_percent_gate")
        if not np.isfinite(row.train_robust_scale) or row.train_robust_scale <= 1e-8:
            if np.isfinite(row.train_raw_max) and row.train_raw_max > row.train_raw_min:
                issues.append("robust_scale_collapsed_but_values_not_constant")
            else:
                issues.append("no_training_variation_or_no_training_values")
        flags.append(";".join(issues))
    result["non_support_review_reasons"] = flags
    result["eligible_for_support_allocation"] = (
        result.nutrition_scope.ne("other_composition_scope_review")
        & result.non_support_review_reasons.eq("")
        & result.minimum_counts_feasible_without_grouping
    )
    return result


def allocate_validation(ledger, train_profiles, foods, assignments, time_limit=90.0):
    """Lexicographic MILP: core coverage, form coverage, then food transfer cost.

    The optimizer sees observation presence, label eligibility and groups only.
    No validation target magnitudes or model predictions determine selection.
    """
    targets = ledger[ledger.eligible_for_support_allocation & ledger.validation_deficit.gt(0)].copy()
    target_ids = targets.component_concept_id.tolist()
    if not target_ids:
        return set(), {"status": "no_eligible_deficits", "stages": []}
    old_targets = ledger[ledger.training_role.eq("maskable_target") & ledger.current_train_count.ge(100)]
    guarded = sorted(set(old_targets.component_concept_id) | set(target_ids))
    strict_targets = train_profiles[train_profiles.component_concept_id.isin(target_ids) & train_profiles.strict_label_eligible]
    blocks = sorted(strict_targets.transfer_block.unique())
    sizes = pd.Series(assignments).value_counts().reindex(blocks).to_numpy(float)
    b_index = {key: i for i, key in enumerate(blocks)}
    axis_index = {key: i for i, key in enumerate(guarded)}
    t_index = {key: i for i, key in enumerate(target_ids)}
    relevant = train_profiles[train_profiles.transfer_block.isin(blocks) & train_profiles.component_concept_id.isin(guarded)]
    counts = relevant.groupby(["component_concept_id", "transfer_block"]).food_concept_id.nunique()
    loss = coo_matrix((counts.to_numpy(float), (
        [axis_index[a] for a, _ in counts.index], [b_index[b] for _, b in counts.index],
    )), shape=(len(guarded), len(blocks))).tocsr()
    strict_counts = strict_targets.groupby(["component_concept_id", "transfer_block"]).food_concept_id.nunique()
    gain = coo_matrix((strict_counts.to_numpy(float), (
        [t_index[a] for a, _ in strict_counts.index], [b_index[b] for _, b in strict_counts.index],
    )), shape=(len(targets), len(blocks))).tocsr()
    n, k = len(blocks), len(targets)
    indexed = ledger.set_index("component_concept_id")
    old_ids = set(old_targets.component_concept_id)
    # Existing targets must not lose the 100 training observations they had.
    old_rows = [axis_index[x] for x in guarded if x in old_ids]
    matrix = [hstack([loss[old_rows], csr_matrix((len(old_rows), k))])]
    low = [-np.inf] * len(old_rows)
    upper = [float(indexed.loc[guarded[i], "current_train_count"] - 100) for i in old_rows]
    selected_loss = loss[[axis_index[x] for x in target_ids]]
    matrix.append(hstack([selected_loss, csr_matrix(np.eye(k) * 100)]))
    low.extend([-np.inf] * k)
    upper.extend(targets.current_train_count.astype(float))
    matrix.append(hstack([gain, csr_matrix(-np.diag(targets.validation_deficit.to_numpy(float)))]))
    low.extend([0.0] * k)
    upper.extend([np.inf] * k)
    A = vstack(matrix).tocsr()
    lower, upper = np.asarray(low), np.asarray(upper)
    stages = []
    solution = None
    core = targets.nutrition_scope.eq("core_nutrition_expression").to_numpy()
    for name, mask in (("maximize_core_expression_coverage", core), ("maximize_chemical_form_coverage", ~core)):
        objective = np.r_[np.zeros(n), -mask.astype(float)]
        if not mask.any():
            continue
        solution = milp(objective, integrality=np.ones(n + k), bounds=Bounds(0, 1),
                        constraints=LinearConstraint(A, lower, upper),
                        options={"time_limit": time_limit, "mip_rel_gap": 0.0})
        if not solution.success:
            raise RuntimeError(f"Coverage optimization did not prove completion: {name}: {solution.message}")
        optimum = int(round(-solution.fun))
        stages.append({"name": name, "optimum": optimum})
        row = csr_matrix(np.r_[np.zeros(n), mask.astype(float)][None, :])
        A = vstack([A, row]).tocsr()
        lower, upper = np.r_[lower, optimum], np.r_[upper, optimum]
    # The fractional deterministic tie cost sums to less than one food.
    tie = np.array([int(hashlib.sha256(b.encode()).hexdigest()[:8], 16) / 2**32 for b in blocks]) / (n + 1)
    objective = np.r_[sizes + tie, np.zeros(k)]
    solution = milp(objective, integrality=np.ones(n + k), bounds=Bounds(0, 1),
                    constraints=LinearConstraint(A, lower, upper),
                    options={"time_limit": time_limit, "mip_rel_gap": 0.0})
    if not solution.success:
        raise RuntimeError(f"Food-transfer optimization did not prove completion: {solution.message}")
    selected = {blocks[i] for i in np.flatnonzero(solution.x[:n] > 0.5)}
    moved = {food for food, block in assignments.items() if block in selected}
    stages.append({"name": "minimize_transferred_foods", "optimum": len(moved)})
    return moved, {"status": "optimal", "stages": stages, "selected_blocks": sorted(selected),
                   "candidate_target_ids": target_ids}


def apply_candidate_partition(foods, profiles, ledger, moved_ids, assignments):
    """Refit support and scale after the move; flag every unfulfilled axis."""
    partition = foods.copy()
    partition["previous_partition"] = partition.partition
    partition["transfer_block_id"] = partition.food_concept_id.map(assignments).fillna("")
    moved = partition.food_concept_id.isin(moved_ids)
    if not partition.loc[moved, "partition"].eq("train").all():
        raise ValueError("Only original training foods may be moved")
    selected_blocks = {assignments[identifier] for identifier in moved_ids}
    if moved_ids != {identifier for identifier, block in assignments.items() if block in selected_blocks}:
        raise ValueError("Transfer split a family or provenance block")
    partition.loc[moved, "partition"] = "validation"
    partition.loc[moved, "validation_panel"] = "nutrition_coverage_holdout"
    partition.loc[moved, "cv_fold"] = -1
    partition.loc[moved, "family_holdout"] = True
    partition.loc[moved, "source_holdout"] = False
    partition.loc[moved, "frozen_validation_identity"] = False
    partition["split_policy"] = "coverage_candidate_preserve_v3_validation_add_whole_train_blocks"
    for column in ("benchmark_eligible", "text_task_eligible", "reconstruction_task_eligible"):
        partition[column] = False
    partition["candidate_only_not_for_training"] = True
    frame = profiles.copy()
    frame["candidate_partition"] = np.where(frame.food_concept_id.isin(moved_ids), "validation", frame.partition)
    allowed = frame.candidate_partition.eq("train") | frame.strict_label_eligible
    final = frame[allowed].copy()
    result = ledger.set_index("component_concept_id").copy()
    for part in ("train", "validation"):
        values = final[final.candidate_partition.eq(part)]
        result["proposed_" + part + "_count"] = values.groupby("component_concept_id").food_concept_id.nunique().reindex(result.index).fillna(0).astype(int)
    training = final[final.candidate_partition.eq("train")]
    scales = {}
    for identifier, group in training.groupby("component_concept_id"):
        logs = np.log1p(group.canonical_value_g_per_100g.to_numpy(float))
        center = np.median(logs)
        scales[identifier] = max((np.quantile(logs, .75) - np.quantile(logs, .25)) / 1.349,
                                 np.median(np.abs(logs - center)) * 1.4826)
    result["proposed_train_robust_scale"] = pd.Series(scales).reindex(result.index)
    result["proposed_counts_sufficient"] = result.proposed_train_count.ge(100) & result.proposed_validation_count.ge(30)
    statuses = []
    for row in result.itertuples():
        if row.nutrition_scope == "other_composition_scope_review":
            status = "nutrition_scope_requires_review"
        elif row.non_support_review_reasons:
            status = "identity_expression_or_distribution_review"
        elif row.current_train_count < 100:
            status = "insufficient_training_support_review"
        elif row.train_count_after_minimum_transfer < 100:
            status = "insufficient_total_support_for_100_plus_30_review"
        elif not row.strict_labels_sufficient_for_transfer:
            status = "validation_evidence_insufficient_not_fixed_by_split"
        elif not row.proposed_counts_sufficient:
            status = "family_block_or_joint_allocation_infeasible_review"
        elif not np.isfinite(row.proposed_train_robust_scale) or row.proposed_train_robust_scale <= 1e-8:
            status = "post_transfer_training_distribution_review"
        elif row.validation_deficit > 0:
            status = "coverage_restored_by_training_transfer"
        else:
            status = "counts_already_sufficient"
        statuses.append(status)
    result["coverage_decision"] = statuses
    result["support_ready_for_prediction"] = result.coverage_decision.isin([
        "counts_already_sufficient", "coverage_restored_by_training_transfer",
    ])
    result["review_required_before_exclusion"] = ~result.support_ready_for_prediction
    result["final_target_approval"] = "pending_identity_scope_and_label_review"
    result["both_current_partitions_below_minimum"] = result.current_train_count.lt(100) & result.current_validation_count.lt(30)
    result["training_evidence_not_benchmark_evidence"] = result.current_train_count > result.strict_train_count
    withheld = frame[~allowed].copy()
    return partition, result.reset_index(), withheld
