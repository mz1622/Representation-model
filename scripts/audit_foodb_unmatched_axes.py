#!/usr/bin/env python3
"""Audit FooDB observation-level axes without an accepted external mapping."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_FOODB = ROOT / "foodb_2020_04_07_csv"
OBSERVATION_DIR = ROOT / "data" / "processed" / "fooddb_observations_v1"
AUDIT_DIR = ROOT / "data" / "audits"


def join_values(values: pd.Series) -> str:
    return " | ".join(sorted({str(value).strip() for value in values.dropna() if str(value).strip()}))


def canonical_unit(value: object) -> str | None:
    """Normalize unit spelling only; this does not perform a unit conversion."""
    if pd.isna(value) or not str(value).strip():
        return None
    text = str(value).strip().lower().replace("µ", "u").replace("μ", "u")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*/\s*", "/", text)
    text = re.sub(r"/100\s*g\b", "/100g", text)
    return text


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small report table without requiring the optional tabulate package."""
    columns = list(frame.columns)
    rows = frame.fillna("").astype(str).values.tolist()
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row.replace("|", "\\|") for row in values) + " |" for values in rows]
    return "\n".join([header, divider, *body])


def main() -> None:
    if not (OBSERVATION_DIR / "Content.csv").is_file():
        raise FileNotFoundError("Run scripts/build_fooddb_observations.py before this audit.")

    content = pd.read_csv(
        OBSERVATION_DIR / "Content.csv",
        usecols=["source_id", "source_type", "food_id", "citation", "orig_unit"],
        low_memory=False,
    )
    content["source_type"] = content["source_type"].str.lower()
    content["canonical_unit"] = content["orig_unit"].map(canonical_unit)
    coverage = content.groupby(["source_type", "source_id"]).agg(
        observed_foods=("food_id", "nunique"),
        numeric_rows=("food_id", "size"),
        source_citation_count=("citation", "nunique"),
        source_citations=("citation", join_values),
        canonical_unit_count=("canonical_unit", "nunique"),
        canonical_units=("canonical_unit", join_values),
    ).reset_index()

    nutrients = pd.read_csv(RAW_FOODB / "Nutrient.csv", usecols=["id", "public_id", "name", "type", "description"], low_memory=False)
    nutrients = nutrients.rename(columns={"id": "source_id"})
    nutrients["source_type"] = "nutrient"
    nutrients["kingdom"] = ""
    nutrients["superklass"] = ""
    nutrients["klass"] = ""
    nutrients["subklass"] = ""

    compounds = pd.read_csv(
        RAW_FOODB / "Compound.csv",
        usecols=["id", "public_id", "name", "description", "kingdom", "superklass", "klass", "subklass"],
        low_memory=False,
    ).rename(columns={"id": "source_id"})
    compounds["source_type"] = "compound"
    compounds["type"] = ""

    axes = pd.concat([nutrients, compounds], ignore_index=True, sort=False)
    axes = axes.merge(coverage, on=["source_type", "source_id"], how="inner")
    reviewed = pd.read_csv(AUDIT_DIR / "foodb_component_crosswalk_review.csv")
    reviewed = reviewed[reviewed["review_status"].eq("accepted")].copy()
    reviewed["source_type"] = reviewed["foodb_axis_kind"].str.lower()
    reviewed = reviewed.rename(columns={"foodb_axis_name": "name"})
    axes = axes.merge(
        reviewed[["source_type", "name", "canonical_usda_code", "canonical_name"]],
        on=["source_type", "name"], how="left", validate="one_to_one",
    )
    candidates = pd.read_csv(AUDIT_DIR / "foodb_compound_name_candidates.csv")
    candidates = candidates[candidates["review_status"].eq("name_candidate_only")]
    candidate_ids = set(candidates["source_id"])

    axes["mapping_status"] = "accepted_external_mapping"
    unreviewed = axes["canonical_usda_code"].isna()
    axes.loc[unreviewed & axes["source_type"].eq("nutrient"), "mapping_status"] = "nutrient_without_reviewed_mapping"
    axes.loc[unreviewed & axes["source_type"].eq("compound") & axes["source_id"].isin(candidate_ids), "mapping_status"] = "compound_name_candidate_only"
    axes.loc[unreviewed & axes["source_type"].eq("compound") & ~axes["source_id"].isin(candidate_ids), "mapping_status"] = "compound_no_external_candidate"
    unmatched = axes[axes["canonical_usda_code"].isna()].copy()
    unmatched["eligible_for_foodb_only_auxiliary"] = (
        unmatched["source_type"].eq("compound")
        & unmatched["observed_foods"].ge(50)
        & unmatched["source_citation_count"].ge(2)
        & unmatched["canonical_unit_count"].eq(1)
    )
    unmatched = unmatched.sort_values(["mapping_status", "observed_foods", "name"], ascending=[True, False, True])
    columns = [
        "source_type", "source_id", "public_id", "name", "mapping_status", "observed_foods", "numeric_rows",
        "source_citation_count", "source_citations", "canonical_unit_count", "canonical_units", "eligible_for_foodb_only_auxiliary",
        "kingdom", "superklass", "klass", "subklass", "type", "description",
    ]
    unmatched[columns].to_csv(AUDIT_DIR / "foodb_unmatched_observation_axes.csv", index=False)

    summary = unmatched.groupby(["source_type", "mapping_status"]).agg(
        axes=("source_id", "size"),
        axes_at_least_10_foods=("observed_foods", lambda values: int(values.ge(10).sum())),
        axes_at_least_50_foods=("observed_foods", lambda values: int(values.ge(50).sum())),
        axes_at_least_100_foods=("observed_foods", lambda values: int(values.ge(100).sum())),
        axes_with_two_or_more_citations=("source_citation_count", lambda values: int(values.ge(2).sum())),
        axes_with_one_canonical_unit=("canonical_unit_count", lambda values: int(values.eq(1).sum())),
        foodb_only_auxiliary_candidates=("eligible_for_foodb_only_auxiliary", "sum"),
    ).reset_index()
    summary.to_csv(AUDIT_DIR / "foodb_unmatched_observation_axis_summary.csv", index=False)
    classes = unmatched[unmatched["source_type"].eq("compound")].groupby("superklass").agg(
        axes=("source_id", "size"),
        axes_at_least_50_foods=("observed_foods", lambda values: int(values.ge(50).sum())),
        auxiliary_candidates=("eligible_for_foodb_only_auxiliary", "sum"),
    ).reset_index().sort_values("axes", ascending=False)
    classes.to_csv(AUDIT_DIR / "foodb_unmatched_compound_classes.csv", index=False)

    top = unmatched.sort_values("observed_foods", ascending=False).head(30)
    report = [
        "# FooDB Axes Without an Accepted External Match",
        "",
        "## Scope",
        "",
        "FooDB coverage is computed from the source-level observation dataset, keyed by `citation + orig_food_id + orig_food_part`. An unmatched axis has no accepted FooDB-to-USDA/CNF mapping; a strict name candidate is not treated as an equivalence.",
        "",
        "## Summary",
        "",
        markdown_table(summary),
        "",
        "## Training Decision",
        "",
        "- Do not include unmatched axes in the cross-database shared reconstruction vocabulary: there is no validated target-definition correspondence or cross-source evaluation target.",
        "- Do not treat absent FooDB records as zero. Sparse compound coverage reflects both real absence and database/literature coverage.",
        "- A FooDB-only auxiliary objective is defensible only for the rows marked `eligible_for_foodb_only_auxiliary`: at least 50 observation foods, at least two citations, and one canonically spelled raw unit. This is spelling normalization only, not numerical unit conversion. It is an auxiliary representation objective, not a benchmark target until a held-out provenance-aware evaluation is defined.",
        "- Axes with only a strict name candidate need chemical identifier, unit, basis, and aggregation review before promotion to a shared axis.",
        "",
        "## Most Covered Unmatched Axes",
        "",
        markdown_table(top[["source_type", "name", "mapping_status", "observed_foods", "source_citation_count", "canonical_units", "eligible_for_foodb_only_auxiliary"]]),
        "",
        "## Flavor and Other Relations",
        "",
        "FooDB `Flavor.csv` and `CompoundsFlavor.csv` provide compound-to-flavor relations, rather than per-observation quantitative food values. They should be used later as ontology metadata or a relation-prediction task, not inserted into the current continuous masked-value matrix.",
        "",
        "## Files",
        "",
        "- `foodb_unmatched_observation_axes.csv`: every unmatched observed FooDB axis and its coverage/provenance diagnostics.",
        "- `foodb_unmatched_observation_axis_summary.csv`: counts by matching status and axis kind.",
        "- `foodb_unmatched_compound_classes.csv`: chemical-class distribution for unmatched compounds.",
    ]
    (AUDIT_DIR / "FOODB_UNMATCHED_AXIS_REVIEW.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Unmatched source-level FooDB axes: {len(unmatched):,}")
    print(f"FooDB-only auxiliary candidates: {int(unmatched.eligible_for_foodb_only_auxiliary.sum()):,}")


if __name__ == "__main__":
    main()
