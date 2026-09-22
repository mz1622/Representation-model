"""Machine-readable release-gate checks for the scientific candidate dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DATASET_VERSION
from .source_policy import TRUSTED_REFERENCE_POLICY, TRUSTED_REFERENCE_SOURCES, validation_reference_mask
from .util import normalize_text, read_component_csv, write_csv, write_json


def _check(check_id: str, severity: str, passed: bool, observed: Any, requirement: str) -> dict[str, Any]:
    return {
        "check_id": check_id, "severity": severity, "passed": bool(passed),
        "observed": observed, "requirement": requirement,
    }


def _composition_checks(profiles: pd.DataFrame, components: pd.DataFrame, foods: pd.DataFrame) -> pd.DataFrame:
    accepted = profiles[profiles["aggregation_status"].eq("accepted")].merge(
        components[["component_concept_id", "infoods_tag", "canonical_name", "component_family"]],
        on="component_concept_id", how="left",
    )
    standard = accepted[accepted["infoods_tag"].notna()].copy()
    # Multiple analytical definitions sharing a tag cannot silently collapse
    # through pivot_table(first) for a composition check.
    standard = standard[~standard.duplicated(["food_concept_id", "infoods_tag"], keep=False)]
    rows: list[dict[str, Any]] = []
    wide = standard.pivot_table(index="food_concept_id", columns="infoods_tag", values="canonical_value_g_per_100g", aggfunc="first")
    proximate_tags = [tag for tag in ("WATER", "PROCNT", "FAT", "ASH", "CHOCDF") if tag in wide.columns]
    if len(proximate_tags) == 5:
        complete = wide[proximate_tags].dropna()
        totals = complete.sum(axis=1)
        for food_id, total in totals.items():
            rows.append({
                "check_family": "proximate_mass_balance", "food_concept_id": food_id,
                "observed_total_g_per_100g": total, "lower_reference": 80.0,
                "upper_reference": 120.0, "status": "pass" if 80 <= total <= 120 else "review",
                "note": "Screening only; carbohydrate and fibre definitions must be compatible before interpretation.",
            })
    if "FAT" in wide.columns:
        fatty = standard[standard["infoods_tag"].str.fullmatch(r"F[0-9]+D0", na=False)]
        sums = fatty.groupby("food_concept_id")["canonical_value_g_per_100g"].sum()
        for food_id in sums.index.intersection(wide.index):
            fat = wide.at[food_id, "FAT"]
            if pd.notna(fat):
                ratio = sums[food_id] / fat if fat > 0 else np.inf if sums[food_id] > 0 else 0.0
                rows.append({
                    "check_family": "disjoint_saturated_fatty_acid_subtotal_vs_fat", "food_concept_id": food_id,
                    "observed_total_g_per_100g": sums[food_id], "reference_value_g_per_100g": fat,
                    "ratio": ratio, "status": "pass" if ratio <= 1.2 else "review",
                    "note": "Review trigger only: disjoint chain-length saturated acids form a partial mass sum. Aggregate n-3/n-6 and isomer totals are not added to their constituents.",
                })
    if "PROCNT" in wide.columns:
        amino_tags = {"ALA", "ARG", "ASP", "CYS", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"}
        amino = standard[standard["infoods_tag"].isin(amino_tags)]
        sums = amino.groupby("food_concept_id")["canonical_value_g_per_100g"].sum()
        for food_id in sums.index.intersection(wide.index):
            protein = wide.at[food_id, "PROCNT"]
            if pd.notna(protein):
                ratio = sums[food_id] / protein if protein > 0 else np.inf if sums[food_id] > 0 else 0.0
                rows.append({
                    "check_family": "amino_acid_sum_vs_protein", "food_concept_id": food_id,
                    "observed_total_g_per_100g": sums[food_id], "reference_value_g_per_100g": protein,
                    "ratio": ratio, "status": "pass" if ratio <= 1.2 else "review",
                    "note": "Screening only; amino-acid nitrogen/protein conversion and panel completeness require expert interpretation.",
                })
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.merge(foods[["food_concept_id", "canonical_name"]], on="food_concept_id", how="left")
    return result


def run(root: Path, release_dir: Path, audit_dir: Path) -> dict[str, Any]:
    foods = pd.read_csv(release_dir / "food_concept.csv.gz", low_memory=False)
    mapping = pd.read_csv(release_dir / "food_observation_to_concept.csv.gz", low_memory=False)
    observations = pd.read_csv(release_dir / "food_observation.csv.gz", low_memory=False)
    components = read_component_csv(release_dir / "component_concept.csv.gz")
    measurements = pd.read_csv(release_dir / "measurement_main_eligible.csv.gz", low_memory=False)
    profiles = pd.read_csv(release_dir / "canonical_profile.csv.gz", low_memory=False)
    partitions = pd.read_csv(release_dir / "ml_partition.csv", low_memory=False)
    conflicts_path = audit_dir / "unresolved_measurement_conflicts.csv"
    conflicts = pd.read_csv(conflicts_path) if conflicts_path.exists() and conflicts_path.stat().st_size > 1 else pd.DataFrame()

    checks = []
    checks.append(_check("partitions_only_train_validation", "critical", set(partitions["partition"].dropna()) <= {"train", "validation"}, sorted(partitions["partition"].dropna().unique()), "Only train and validation partitions exist; no test partition."))
    checks.append(_check("food_partition_unique", "critical", not partitions["food_concept_id"].duplicated().any(), int(partitions["food_concept_id"].duplicated().sum()), "Each food concept occurs in exactly one ML partition."))
    if "family_expansion_conflict" in partitions:
        held_in_train = partitions["partition"].eq("train") & partitions["family_expansion_conflict"].astype(str).str.casefold().eq("true")
        checks.append(_check("unresolved_family_bridges_not_in_train", "critical", not held_in_train.any(), int(held_in_train.sum()), "Unresolved frozen-family bridge candidates cannot enter train on a version rebuild."))
    checks.append(_check("profile_key_unique", "critical", not profiles.duplicated(["food_concept_id", "component_concept_id"]).any(), int(profiles.duplicated(["food_concept_id", "component_concept_id"]).sum()), "Canonical profile has one row per food-component key."))
    accepted = profiles[profiles["aggregation_status"].eq("accepted") & profiles["canonical_value_g_per_100g"].notna()]
    invalid_values = accepted[~accepted["canonical_value_g_per_100g"].between(0, 100, inclusive="both")]
    checks.append(_check("physical_mass_fraction_bounds", "critical", invalid_values.empty, len(invalid_values), "All accepted main-modality values are within 0 to 100 g/100 g."))
    checks.append(_check("main_measurement_units_traceable", "critical", measurements["conversion_status"].eq("converted_exact_mass").all(), int((~measurements["conversion_status"].eq("converted_exact_mass")).sum()), "Every main-eligible measurement has an exact mass conversion trace."))
    independent = measurements["independent_evidence"].astype(str).str.casefold().eq("true")
    curated = measurements["data_layer"].eq("curated_reference_training") & measurements["quality_tier"].eq("C")
    admissible = independent | curated
    if "source_admission_policy" in measurements:
        documented_internal_citation = (
            measurements.source_policy_decision.eq("admitted_source_published_internal_citation")
            & measurements.source_policy_reason.eq("user_2026_09_09_internal_citations_preserved_without_origin_resolution")
        )
        trusted_reference = (
            measurements.source_key.isin(TRUSTED_REFERENCE_SOURCES)
            & measurements.source_admission_policy.eq(TRUSTED_REFERENCE_POLICY)
            & (measurements.source_policy_decision.eq("admitted_trusted_reference") | documented_internal_citation)
            & measurements.source_trusted.astype(str).str.casefold().eq("true")
        )
        admissible |= trusted_reference
        newly_strict = (
            measurements.strict_validation_eligible.astype(str).str.casefold().eq("true")
            & ~measurements.source_gate_previous_validation_eligible.astype(str).str.casefold().eq("true")
        )
        checks.append(_check("source_trust_not_assay_certification", "critical", not newly_strict.any(), int(newly_strict.sum()), "Source trust never turns a previously non-strict record into an analytically certified label."))
        legacy_validation = measurements.strict_validation_eligible.astype(str).str.casefold().eq("true")
        unexplained_reference = validation_reference_mask(measurements) & ~(legacy_validation | trusted_reference)
        checks.append(_check("reference_validation_policy_documented", "critical", not unexplained_reference.any(), int(unexplained_reference.sum()), "Every reference-validation record has a legacy gate or explicit trusted-source admission."))
        val_ids = set(profiles.loc[profiles.partition.eq("validation"), "measurement_ids"].dropna().str.split(";").explode())
        valid_ids = set(measurements.loc[validation_reference_mask(measurements), "measurement_id"])
        rejected_ids = val_ids - valid_ids
        checks.append(_check("validation_selected_records_admitted", "critical", not rejected_ids, len(rejected_ids), "All selected validation provenance records satisfy the versioned reference-label policy."))
    checks.append(_check("measurement_origin_admitted", "critical", admissible.all(), int((~admissible).sum()), "Measurements are source-admitted reference values or explicitly flagged curated training-only values, not certified as universally analytical."))
    if "strict_validation_eligible" in measurements:
        strict = measurements["strict_validation_eligible"].astype(str).str.casefold().eq("true")
        checks.append(_check("curated_never_strict_validation", "critical", not (curated & strict).any(), int((curated & strict).sum()), "Curated food-level references do not become strict validation labels."))
    checks.append(_check("no_censored_as_numeric", "critical", not (measurements["is_censored"].astype(str).str.casefold().eq("true") & measurements["normalized_value_g_per_100g"].notna()).any(), int((measurements["is_censored"].astype(str).str.casefold().eq("true") & measurements["normalized_value_g_per_100g"].notna()).sum()), "Censored/trace values are not ordinary numeric labels."))
    checks.append(_check("explicit_zero_distinct", "critical", measurements.loc[measurements["is_explicit_zero"].astype(str).str.casefold().eq("true"), "normalized_value_g_per_100g"].fillna(-1).eq(0).all(), int(measurements["is_explicit_zero"].astype(str).str.casefold().eq("true").sum()), "Explicit zeros remain numeric zero and are not missing/censored."))

    linked_foods = mapping.merge(observations, on="food_observation_id", how="left")
    facet_columns = ["scientific_name", "part", "maturity", "processing", "cooking", "preservation", "physical_state", "packing_medium", "cultivar"]
    facet_conflicts = 0
    for column in facet_columns:
        values = linked_foods.dropna(subset=[column]).copy()
        values[column] = values[column].map(normalize_text)
        values = values[values[column].ne("")]
        facet_conflicts += int(values.groupby("food_concept_id")[column].nunique().gt(1).sum())
    checks.append(_check("food_concept_facets_compatible", "critical", facet_conflicts == 0, facet_conflicts, "No food concept contains conflicting explicit identity facets."))

    family_holdouts = set(partitions.loc[partitions["family_holdout"].astype(str).str.casefold().eq("true"), "family_cluster_id"])
    train_families = set(partitions.loc[partitions["partition"].eq("train"), "family_cluster_id"])
    overlap = family_holdouts & train_families
    checks.append(_check("family_holdout_no_train_family_overlap", "critical", not overlap, len(overlap), "No family-holdout cluster appears in train."))

    obs_partition = mapping.merge(
        observations[["food_observation_id", "source_key"]],
        on="food_observation_id",
        how="left",
    ).merge(partitions[["food_concept_id", "partition"]], on="food_concept_id", how="inner")
    foundation_train = obs_partition[obs_partition["source_key"].eq("usda_foundation") & obs_partition["partition"].eq("train")]
    checks.append(_check("foundation_not_in_train", "critical", foundation_train.empty, len(foundation_train), "Every USDA Foundation concept is locked to source validation."))
    formal_lineage_rows = obs_partition[obs_partition["normalized_lineage"].fillna("").ne("")]
    split_lineage = formal_lineage_rows.groupby("normalized_lineage")["partition"].nunique()
    lineage_leaks = split_lineage[split_lineage.gt(1)]
    checks.append(_check("formal_lineage_no_split_leakage", "critical", lineage_leaks.empty, len(lineage_leaks), "A formal source lineage cannot span train and validation."))
    family_partitions = partitions.groupby("family_cluster_id")["partition"].nunique()
    mixed_families = set(family_partitions[family_partitions.gt(1)].index)
    mixed_validation = partitions[
        partitions["family_cluster_id"].isin(mixed_families)
        & partitions["partition"].eq("validation")
    ]
    invalid_mixed_families = set(
        mixed_validation.loc[
            ~mixed_validation["validation_panel"].eq("source_holdout"),
            "family_cluster_id",
        ]
    )
    unmarked_source_relatives = partitions[
        partitions["family_cluster_id"].isin(mixed_families)
        & partitions["partition"].eq("train")
        & ~partitions["source_holdout_family_overlap"].astype(str).str.casefold().eq("true")
    ]
    source_family_policy_violations = invalid_mixed_families | set(
        unmarked_source_relatives["family_cluster_id"]
    )
    checks.append(_check(
        "family_overlap_only_for_source_holdout",
        "critical",
        not source_family_policy_violations,
        len(source_family_policy_violations),
        "A family may span train and validation only for the prespecified USDA Foundation source-transfer panel; unseen-family holdouts remain fully blocked.",
    ))
    source_holdout_profiles = profiles[
        profiles["validation_panel"].fillna("").isin(["source_holdout", "source_holdout+family_holdout"])
    ]
    nonfoundation_holdout = source_holdout_profiles[~source_holdout_profiles["source_keys"].fillna("").eq("usda_foundation")]
    checks.append(_check("source_holdout_labels_foundation_only", "critical", nonfoundation_holdout.empty, len(nonfoundation_holdout), "USDA Foundation source-holdout labels are aggregated from Foundation measurements only."))

    targets = components[components["training_role"].eq("maskable_target")]
    unsupported_context = components[components["training_role"].eq("context_only") & components["train_count"].eq(0)]
    checks.append(_check("context_has_training_observations", "critical", unsupported_context.empty, len(unsupported_context), "Context axes must have training observations; validation-only evidence stays archival."))
    support_bad = targets[(targets["train_count"] < 100) | (targets["validation_count"] < 30)]
    checks.append(_check("maskable_support", "critical", support_bad.empty, len(support_bad), "Each maskable target has >=100 train and >=30 validation concepts."))
    identity_bad = targets[~targets["identity_status"].isin(["authority_verified", "stable_source_identity_pending_cross_database_review"])]
    checks.append(_check("maskable_identity", "critical", identity_bad.empty, len(identity_bad), "Each maskable target has an authority ID or stable source-local identity with a clear definition."))
    expression_bad = targets[targets["expression_variant"].isin(["label_expression", "biological_equivalent"])]
    checks.append(_check("maskable_mass_expression_only", "critical", expression_bad.empty, len(expression_bad), "IU/RAE/DFE/alpha-TE/label expressions are outside the main target."))
    coverage_bad = targets[targets["capability_adjusted_coverage"] < 0.02]
    checks.append(_check("maskable_coverage", "critical", coverage_bad.empty, len(coverage_bad), "Each maskable target meets the capability-adjusted 2% coverage threshold."))
    definition_bad = targets[
        targets["authority_namespace"].fillna("").eq("")
        | targets["authority_id"].fillna("").eq("")
        | targets["definition"].fillna("").eq("")
    ]
    checks.append(_check("maskable_identity_definition_complete", "critical", definition_bad.empty, len(definition_bad), "Each maskable target has a stable authority/source identifier and a recorded definition."))
    scale_bad = targets[targets["train_robust_scale"].isna() | targets["train_robust_scale"].le(1e-8)]
    checks.append(_check("maskable_train_only_scale", "critical", scale_bad.empty, len(scale_bad), "Each maskable target has a non-degenerate robust scale fitted from train only."))

    reconstruction_bad = partitions[partitions["reconstruction_task_eligible"].astype(str).str.casefold().eq("true") & partitions["mask_family_count"].lt(2)]
    checks.append(_check("reconstruction_two_families", "critical", reconstruction_bad.empty, len(reconstruction_bad), "Reconstruction foods have at least two non-equivalent mask families."))

    composition = _composition_checks(profiles, components, foods)
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(checks), audit_dir / "automated_checks.csv")
    write_csv(composition, audit_dir / "composition_plausibility_checks.csv.gz")
    if not composition.empty:
        write_csv(composition[composition["status"].eq("review")], audit_dir / "composition_expert_review_queue.csv")

    critical_failures = [row for row in checks if row["severity"] == "critical" and not row["passed"]]
    review_queue_count = int((~components["identity_status"].eq("authority_verified")).sum()) + len(conflicts)
    version = json.loads((release_dir / "dataset_summary.json").read_text())["dataset_version"]
    result = {
        "dataset_version": version,
        "critical_checks": sum(row["severity"] == "critical" for row in checks),
        "critical_failures": len(critical_failures),
        "failed_check_ids": [row["check_id"] for row in critical_failures],
        "composition_review_flags": int((composition.get("status", pd.Series(dtype=str)) == "review").sum()),
        "unresolved_measurement_conflicts": len(conflicts),
        "expert_identity_review_items": review_queue_count,
        "automated_release_gate": "pass" if not critical_failures else "fail",
        "scientific_release_status": "candidate_pending_domain_expert_and_dual_reviewer_review",
    }
    write_json(result, audit_dir / "audit_summary.json")
    lines = [
        f"# Automated Dataset Audit: {version}", "",
        f"Automated release gate: **{result['automated_release_gate'].upper()}**.", "",
        "This is a candidate scientific build, not a signed final release. Automated checks cannot replace the required food-composition expert review, dual-reviewer literature screening, or licence confirmation.", "",
        "## Critical checks", "",
    ]
    for row in checks:
        lines.append(f"- {'PASS' if row['passed'] else 'FAIL'} `{row['check_id']}`: {row['requirement']} Observed: {row['observed']}.")
    lines.extend(["", "## Open release gates", "", f"- Unresolved value conflicts: {len(conflicts):,}.", f"- Identity/expert review queue size: {review_queue_count:,}.", f"- Composition plausibility flags: {result['composition_review_flags']:,}."])
    (audit_dir / "AUTOMATED_AUDIT_REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--release-dir", type=Path, default=Path("data/processed/scientific_food_composition_v1/release"))
    parser.add_argument("--audit-dir", type=Path, default=Path("data/audits/scientific_food_composition_v1/dataset"))
    args = parser.parse_args()
    root = args.root.resolve()
    result = run(root, root / args.release_dir, root / args.audit_dir)
    if result["critical_failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
