#!/usr/bin/env python3
"""Build a conservative, observed-value-only axis overlap audit.

The audit treats FooDB, USDA Foundation, USDA SR Legacy, FNDDS, and CNF as
five distinct sources.  USDA/CNF axes are joined solely through the official
USDA nutrient number.  FooDB is added only through the reviewed mapping file;
strict normalized-name matches are written separately as candidates and never
promoted to training axes automatically.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data" / "audits"
REVIEW_PATH = AUDIT_DIR / "foodb_component_crosswalk_review.csv"
FOODB_RAW_DIR = ROOT / "foodb_2020_04_07_csv"
FOODB_OBSERVATION_DIR = ROOT / "data" / "processed" / "fooddb_observations_v1"

FDC_SOURCES = {
    "usda_foundation": {
        "root": ROOT / "data/raw/usda/foundation_2026_04_30",
        "foundation_only": True,
        "fndds": False,
    },
    "usda_sr_legacy": {
        "root": ROOT / "data/raw/usda/sr_legacy_2018/FoodData_Central_sr_legacy_food_csv_2018-04",
        "foundation_only": False,
        "fndds": False,
    },
    "usda_fndds": {
        "root": ROOT / "data/raw/usda/fndds_2021_2023/FoodData_Central_survey_food_csv_2024-10-31",
        "foundation_only": False,
        "fndds": True,
    },
}
REFERENCE_SOURCES = [*FDC_SOURCES, "cnf"]
ALL_SOURCES = ["foodb", *REFERENCE_SOURCES]


def strict_name(value: object) -> str:
    """Mechanical candidate key only; this never establishes equivalence."""
    text = str(value).lower()
    text = text.replace("α", "alpha").replace("β", "beta").replace("γ", "gamma").replace("δ", "delta")
    text = text.replace("±", "")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def observed_fdc_axes(source: str, config: dict[str, object]) -> pd.DataFrame:
    root = config["root"]
    nutrients = pd.read_csv(root / "nutrient.csv", low_memory=False)
    values = pd.read_csv(root / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"], low_memory=False)
    if config["foundation_only"]:
        foundation_foods = pd.read_csv(root / "foundation_food.csv", usecols=["fdc_id"])
        values = values[values["fdc_id"].isin(foundation_foods["fdc_id"])]

    values["amount"] = pd.to_numeric(values["amount"], errors="coerce")
    values = values[values["amount"].notna()].copy()
    if config["fndds"]:
        nutrients["canonical_usda_code"] = pd.to_numeric(nutrients["nutrient_nbr"], errors="coerce")
        values["canonical_usda_code"] = pd.to_numeric(values["nutrient_id"], errors="coerce")
    else:
        nutrients["canonical_usda_code"] = pd.to_numeric(nutrients["nutrient_nbr"], errors="coerce")
        values = values.merge(
            nutrients[["id", "canonical_usda_code"]], left_on="nutrient_id", right_on="id", how="left", validate="many_to_one"
        )
    values = values.dropna(subset=["canonical_usda_code"])
    values["canonical_usda_code"] = values["canonical_usda_code"].astype(int)
    coverage = values.groupby("canonical_usda_code").agg(observed_foods=("fdc_id", "nunique")).reset_index()
    labels = nutrients.dropna(subset=["canonical_usda_code"]).copy()
    labels["canonical_usda_code"] = labels["canonical_usda_code"].astype(int)
    labels = labels.drop_duplicates("canonical_usda_code")[["canonical_usda_code", "name", "unit_name"]]
    return coverage.merge(labels, on="canonical_usda_code", how="left").assign(source=source)


def observed_cnf_axes() -> pd.DataFrame:
    root = ROOT / "data/raw/cnf_2026/extracted"
    names = pd.read_csv(root / "Nutrient_Name.csv", low_memory=False)
    values = pd.read_csv(root / "Nutrient_Amount.csv", usecols=["Food_Code", "Nutrient_Code", "Nutrient_Amount"], low_memory=False)
    values["Nutrient_Amount"] = pd.to_numeric(values["Nutrient_Amount"], errors="coerce")
    values = values[values["Nutrient_Amount"].notna()]
    coverage = values.groupby("Nutrient_Code").agg(observed_foods=("Food_Code", "nunique")).reset_index()
    coverage = coverage.rename(columns={"Nutrient_Code": "canonical_usda_code"})
    labels = names.rename(columns={"Nutrient_Code": "canonical_usda_code", "Nutrient_Name_EN": "name", "Nutrient_Unit": "unit_name"})
    return coverage.merge(labels[["canonical_usda_code", "name", "unit_name"]], on="canonical_usda_code", how="left").assign(source="cnf")


def foodb_numeric_coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use generated source-level observations, not FooDB umbrella Food.id."""
    if not (FOODB_OBSERVATION_DIR / "Food.csv").is_file() or not (FOODB_OBSERVATION_DIR / "Content.csv").is_file():
        raise FileNotFoundError(
            "FooDB observation data is missing. Run scripts/build_fooddb_observations.py before this audit."
        )
    content = pd.read_csv(
        FOODB_OBSERVATION_DIR / "Content.csv",
        usecols=["source_id", "source_type", "food_id", "standard_content"],
        low_memory=False,
    )
    content["standard_content"] = pd.to_numeric(content["standard_content"], errors="coerce")
    content = content[content["standard_content"].notna() & content["source_type"].isin(["Nutrient", "Compound"])]
    coverage = content.groupby(["source_type", "source_id"]).agg(observed_foods=("food_id", "nunique")).reset_index()
    nutrient = pd.read_csv(FOODB_RAW_DIR / "Nutrient.csv", usecols=["id", "name"], low_memory=False).assign(source_type="Nutrient")
    compound = pd.read_csv(FOODB_RAW_DIR / "Compound.csv", usecols=["id", "public_id", "name", "cas_number", "moldb_inchikey"], low_memory=False).assign(source_type="Compound")
    axes = pd.concat([nutrient, compound], ignore_index=True, sort=False).rename(columns={"id": "source_id"})
    axes = axes.merge(coverage, on=["source_type", "source_id"], how="inner")
    return axes[axes["source_type"] == "Nutrient"].copy(), axes[axes["source_type"] == "Compound"].copy()


def build_name_candidates(compounds: pd.DataFrame, reference: pd.DataFrame, reviewed: pd.DataFrame) -> pd.DataFrame:
    candidate_reference = reference.copy()
    candidate_reference["strict_name"] = candidate_reference["name"].map(strict_name)
    candidate_reference = candidate_reference.groupby("strict_name").agg(
        external_sources=("source", lambda values: " | ".join(sorted(set(values)))),
        external_source_count=("source", "nunique"),
        external_names=("name", lambda values: " | ".join(sorted(set(values)))),
        external_units=("unit_name", lambda values: " | ".join(sorted(set(map(str, values))))),
    ).reset_index()
    candidates = compounds.copy()
    candidates["strict_name"] = candidates["name"].map(strict_name)
    candidates = candidates.merge(candidate_reference, on="strict_name", how="inner")
    accepted = set(reviewed.loc[reviewed["foodb_axis_kind"] == "compound", "foodb_axis_name"])
    candidates["review_status"] = candidates["name"].map(lambda name: "accepted_mapping" if name in accepted else "name_candidate_only")
    return candidates[["source_id", "public_id", "name", "cas_number", "moldb_inchikey", "observed_foods", "external_sources", "external_source_count", "external_names", "external_units", "review_status"]].sort_values(["external_source_count", "name"], ascending=[False, True])


def main() -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    reviewed = pd.read_csv(REVIEW_PATH)
    if reviewed["review_status"].ne("accepted").any():
        raise ValueError(f"Only accepted rows are permitted in {REVIEW_PATH}")

    reference = pd.concat(
        [observed_fdc_axes(source, config) for source, config in FDC_SOURCES.items()] + [observed_cnf_axes()], ignore_index=True
    )
    reference["canonical_usda_code"] = reference["canonical_usda_code"].astype(int)
    foodb_nutrients, foodb_compounds = foodb_numeric_coverage()
    name_candidates = build_name_candidates(foodb_compounds, reference, reviewed)

    reference_by_code = reference.groupby("canonical_usda_code")
    rows = []
    for code, group in reference_by_code:
        membership = set(group["source"])
        food_counts = dict(zip(group["source"], group["observed_foods"]))
        label = group.sort_values("source").iloc[0]
        rows.append({
            "canonical_usda_code": code,
            "canonical_name": label["name"],
            "canonical_unit": label["unit_name"],
            **{f"foods_{source}": int(food_counts.get(source, 0)) for source in REFERENCE_SOURCES},
            "reference_sources": " | ".join(sorted(membership)),
            "reference_source_count": len(membership),
        })
    axes = pd.DataFrame(rows)

    foodb_rows = []
    for row in reviewed.itertuples(index=False):
        source = foodb_nutrients if row.foodb_axis_kind == "nutrient" else foodb_compounds
        match = source[source["name"].eq(row.foodb_axis_name)]
        if len(match) > 1:
            raise ValueError(f"Expected exactly one FooDB {row.foodb_axis_kind} match for {row.foodb_axis_name!r}, found {len(match)}")
        # A reviewed chemical mapping is not automatically an observed
        # training axis. Source-level reconstruction can exclude records that
        # lack a stable provenance key; such axes get zero FooDB coverage.
        if match.empty:
            foodb_rows.append({
                "canonical_usda_code": int(row.canonical_usda_code),
                "foodb_mapping_count": 0,
                "foodb_observed_foods": 0,
                "foodb_mappings": f"{row.foodb_axis_kind}:{row.foodb_axis_name}",
                "foodb_mapping_relations": row.mapping_relation,
            })
            continue
        match = match.iloc[0]
        foodb_rows.append({
            "canonical_usda_code": int(row.canonical_usda_code),
            "foodb_mapping_count": 1,
            "foodb_observed_foods": int(match.observed_foods),
            "foodb_mappings": f"{row.foodb_axis_kind}:{row.foodb_axis_name}",
            "foodb_mapping_relations": row.mapping_relation,
        })
    foodb_mapping = pd.DataFrame(foodb_rows).groupby("canonical_usda_code").agg(
        foodb_mapping_count=("foodb_mapping_count", "sum"),
        foodb_observed_foods=("foodb_observed_foods", "max"),
        foodb_mappings=("foodb_mappings", " | ".join),
        foodb_mapping_relations=("foodb_mapping_relations", " | ".join),
    ).reset_index()
    axes = axes.merge(foodb_mapping, on="canonical_usda_code", how="left")
    axes["foodb_mapping_count"] = axes["foodb_mapping_count"].fillna(0).astype(int)
    axes["foodb_observed_foods"] = axes["foodb_observed_foods"].fillna(0).astype(int)
    for col in ("foodb_mappings", "foodb_mapping_relations"):
        axes[col] = axes[col].fillna("")
    axes["source_count"] = axes["reference_source_count"] + axes["foodb_mapping_count"].gt(0).astype(int)
    axes["source_membership"] = axes.apply(
        lambda row: " | ".join((["foodb"] if row.foodb_mapping_count else []) + row.reference_sources.split(" | ")), axis=1
    )
    axes["eligible_by_source_overlap"] = axes["source_count"] >= 2
    per_source_food_columns = [f"foods_{source}" for source in REFERENCE_SOURCES] + ["foodb_observed_foods"]
    axes["sources_with_50_foods"] = axes[per_source_food_columns].ge(50).sum(axis=1)
    axes["recommended_initial_training"] = axes["eligible_by_source_overlap"] & axes["sources_with_50_foods"].ge(2)
    axes = axes.sort_values(["source_count", "reference_source_count", "canonical_usda_code"], ascending=[False, False, True])
    eligible = axes[axes["eligible_by_source_overlap"]].copy()
    axes.to_csv(AUDIT_DIR / "all_reference_axes_with_overlap.csv", index=False)
    eligible.to_csv(AUDIT_DIR / "shared_training_axis_candidates.csv", index=False)
    name_candidates.to_csv(AUDIT_DIR / "foodb_compound_name_candidates.csv", index=False)

    source_rows = []
    raw_foodb_axis_count = len(foodb_nutrients) + len(foodb_compounds)
    for source in ALL_SOURCES:
        has_axis = axes["foodb_mapping_count"].gt(0) if source == "foodb" else axes[f"foods_{source}"].gt(0)
        source_rows.append({
            "source": source,
            "observed_numeric_axes": int(raw_foodb_axis_count if source == "foodb" else reference["source"].eq(source).sum()),
            "axes_in_reviewed_canonical_registry": int(has_axis.sum()),
            "axes_present_in_at_least_two_datasets": int((has_axis & axes["eligible_by_source_overlap"]).sum()),
            "axes_passing_50_food_two_source_gate": int((has_axis & axes["recommended_initial_training"]).sum()),
        })
    pd.DataFrame(source_rows).to_csv(AUDIT_DIR / "shared_axis_source_comparison.csv", index=False)

    overlap = eligible.groupby("source_count").agg(axes=("canonical_usda_code", "size"), recommended_initial_training=("recommended_initial_training", "sum")).reset_index()
    supported = eligible[eligible["foodb_mapping_count"].gt(0)]
    axes_by_overlap = []
    for source_count in sorted(eligible["source_count"].unique(), reverse=True):
        group = eligible[eligible["source_count"].eq(source_count)]
        rendered = "; ".join(f"`{row.canonical_usda_code}` {row.canonical_name}" for row in group.itertuples(index=False))
        axes_by_overlap.extend([
            f"### Present in {source_count} datasets ({len(group)} axes)",
            "",
            rendered,
            "",
        ])
    low_coverage = eligible[~eligible["recommended_initial_training"]]
    report = [
        "# Shared Training Axis Audit",
        "",
        "## Decision Rule",
        "",
        "An axis is retained only when it has numeric observations in at least two distinct datasets. Missing in a source remains missing; it is never filled with zero.",
        "USDA Foundation, SR Legacy, FNDDS, and CNF use the official USDA nutrient number as the exact canonical key. FooDB contributes only through `foodb_component_crosswalk_review.csv`, a manually reviewed chemical-definition mapping. FooDB coverage is calculated on source-level observations (`citation + orig_food_id + orig_food_part`), while FooDB nutrient and compound records count as one source for the overlap rule.",
        "",
        "## Result",
        "",
        f"- Reference-source axes with numeric values: {len(axes)}",
        f"- Retained by the two-dataset rule: {len(eligible)}",
        f"- Retained axes with at least 50 foods in at least two sources: {int(eligible['recommended_initial_training'].sum())}",
        f"- Retained canonical axes with a reviewed FooDB mapping: {len(supported)}",
        f"- Reviewed FooDB mappings: {len(reviewed)} ({int((reviewed.foodb_axis_kind == 'nutrient').sum())} nutrient; {int((reviewed.foodb_axis_kind == 'compound').sum())} compound)",
        f"- Strict FooDB compound name candidates requiring review: {int((name_candidates.review_status == 'name_candidate_only').sum())}",
        "",
        "## Overlap Distribution",
        "",
        "| Total source count | Canonical axes | Meet 50-food gate |",
        "| ---: | ---: | ---: |",
        *[f"| {int(row.source_count)} | {int(row.axes)} | {int(row.recommended_initial_training)} |" for row in overlap.itertuples(index=False)],
        "",
        "## Retained Axes by Overlap",
        "",
        *axes_by_overlap,
        "## Deferred by Coverage Gate",
        "",
        "These axes satisfy the two-dataset rule but do not have 50 observed foods in two sources. They remain in the registry but are not initial training targets: "
        + "; ".join(f"`{row.canonical_usda_code}` {row.canonical_name}" for row in low_coverage.itertuples(index=False))
        + ".",
        "",
        "## Interpretation",
        "",
        "The active vocabulary should be `shared_training_axis_candidates.csv` filtered to `recommended_initial_training=true`. The source-overlap rule protects against source-private labels; the coverage gate prevents rare overlap from becoming a high-variance target. A FooDB compound mapping establishes chemical-axis compatibility, not value compatibility: values still require an explicit unit/basis and duplicate-record aggregation step before they enter a shared training matrix.",
        "",
        "## Files",
        "",
        "- `shared_training_axis_candidates.csv`: every retained axis, source membership, per-source food coverage, and FooDB mappings.",
        "- `all_reference_axes_with_overlap.csv`: full reference catalog, including source-private axes that are excluded now.",
        "- `foodb_compound_name_candidates.csv`: strict-name candidates that are deliberately not promoted without review.",
        "- `foodb_component_crosswalk_review.csv`: reviewed FooDB nutrient/compound mappings used in this audit.",
        "- `shared_axis_source_comparison.csv`: source-by-source numeric-axis and shared-axis coverage summary.",
    ]
    (AUDIT_DIR / "SHARED_TRAINING_AXIS_AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("Retained by >=2 source rule:", len(eligible))
    print("Recommended after >=50-food gate:", int(eligible["recommended_initial_training"].sum()))
    print("Reviewed FooDB-supported axes:", len(supported))


if __name__ == "__main__":
    main()
