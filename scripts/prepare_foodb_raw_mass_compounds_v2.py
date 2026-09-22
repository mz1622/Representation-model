#!/usr/bin/env python3
"""Prepare the complete mass-normalizable FooDB compound candidate set.

This is deliberately an *input-candidate* preparation step, not the final
pretraining filter.  Every FooDB compound with at least one numeric
``mg/100g`` record is retained, including rare compounds.  Later corpus
construction may assign a low-support axis to context-only or remove it under
an explicitly recorded support rule; this script does not silently impose a
support threshold.

FooDB's compound registry contains many entities with no quantitative food
content record.  The meaningful raw candidate pool is therefore defined by
``Content.csv`` rather than by the registry size alone.  Only ``mg/100g`` is
accepted here because the current model contract requires values converted to
g/100g without a unit token.  Molar concentrations and activity equivalents
are preserved in an exclusion audit rather than coerced into mass values.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OBSERVATION_DIR = ROOT / "data" / "processed" / "fooddb_observations_v1"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed" / "fooddb_raw_mass_compounds_v2"
CHUNK_SIZE = 200_000


def normalized_unit(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.lower().str.replace(" ", "", regex=False)


def build_axis_registry(content_path: Path, compound_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all mass-compatible axes and a unit-level exclusion audit."""
    support: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    unit_summary: dict[tuple[str, int], dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for chunk in pd.read_csv(
        content_path,
        usecols=["source_id", "source_type", "standard_content", "orig_unit"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        compounds = chunk[chunk["source_type"].eq("Compound")].copy()
        compounds["standard_content"] = pd.to_numeric(compounds["standard_content"], errors="coerce")
        compounds = compounds[compounds["standard_content"].notna()]
        compounds["unit"] = normalized_unit(compounds["orig_unit"])
        for (unit, source_id), group in compounds.groupby(["unit", "source_id"], sort=False):
            key = (str(unit), int(source_id))
            unit_summary[key]["records"] += int(len(group))
            unit_summary[key]["positive_records"] += int((group["standard_content"] > 0).sum())
            unit_summary[key]["zero_records"] += int((group["standard_content"] == 0).sum())
            # A mass fraction cannot exceed 100 g per 100 g.  FooDB contains
            # a small set of mg/100g-labelled values above this bound; those
            # records are retained in a separate audit but must not define
            # training support.
            if unit == "mg/100g" and (group["standard_content"] <= 100_000).all():
                axis = support[int(source_id)]
                axis["records"] += int(len(group))
                axis["positive_records"] += int((group["standard_content"] > 0).sum())
                axis["zero_records"] += int((group["standard_content"] == 0).sum())
            elif unit == "mg/100g":
                valid = group[group["standard_content"] <= 100_000]
                if not valid.empty:
                    axis = support[int(source_id)]
                    axis["records"] += int(len(valid))
                    axis["positive_records"] += int((valid["standard_content"] > 0).sum())
                    axis["zero_records"] += int((valid["standard_content"] == 0).sum())

    compounds = pd.read_csv(compound_path, usecols=["id", "name"], low_memory=False).rename(
        columns={"id": "foodb_compound_id", "name": "axis_name"}
    )
    rows = []
    for compound_id, counts in support.items():
        rows.append(
            {
                "foodb_compound_id": compound_id,
                "axis_id": f"foodb_compound:{compound_id}",
                "model_unit": "g/100g",
                "raw_unit": "mg/100g",
                "observed_records": counts["records"],
                "positive_records": counts["positive_records"],
                "explicit_zero_records": counts["zero_records"],
                "initial_candidate": True,
                "selection_reason": "All FooDB compounds with an observed numeric mg/100g value are retained before support filtering.",
            }
        )
    registry = compounds.merge(pd.DataFrame(rows), on="foodb_compound_id", how="inner", validate="one_to_one")
    registry["support_tier_before_row_deduplication"] = pd.cut(
        registry["observed_records"],
        bins=[-1, 19, 49, float("inf")],
        labels=["rare_lt20", "context_candidate_20_49", "mask_candidate_ge50"],
    ).astype(str)
    registry = registry.sort_values(["axis_name", "foodb_compound_id"], kind="stable").reset_index(drop=True)

    unit_rows = []
    for (unit, compound_id), counts in unit_summary.items():
        unit_rows.append(
            {
                "foodb_compound_id": compound_id,
                "raw_unit": unit or "<missing>",
                "numeric_records": counts["records"],
                "positive_records": counts["positive_records"],
                "explicit_zero_records": counts["zero_records"],
                "current_decision": "retain" if unit == "mg/100g" else "exclude_non_mass_or_noncomparable_unit",
                "reason": (
                    "Directly convertible to g/100g by division by 1,000."
                    if unit == "mg/100g"
                    else "Cannot be placed on the current mass-only scale without an axis-specific chemical or activity conversion."
                ),
            }
        )
    unit_audit = pd.DataFrame(unit_rows).merge(compounds, on="foodb_compound_id", how="left", validate="many_to_one")
    return registry, unit_audit.sort_values(["raw_unit", "axis_name", "foodb_compound_id"], kind="stable")


def write_values(content_path: Path, output_path: Path, invalid_output_path: Path, eligible_ids: set[int]) -> tuple[int, int]:
    """Write one observed FooDB compound value per provenance-level food/axis."""
    first = True
    total = 0
    invalid_total = 0
    invalid_first = True
    columns = ["food_id", "axis_id", "value_g_per_100g", "source_content_id", "source_id", "orig_unit"]
    for chunk in pd.read_csv(
        content_path,
        usecols=["source_content_id", "source_id", "source_type", "food_id", "standard_content", "orig_unit"],
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        selected = chunk[chunk["source_type"].eq("Compound")].copy()
        selected["standard_content"] = pd.to_numeric(selected["standard_content"], errors="coerce")
        selected = selected[selected["standard_content"].notna() & selected["source_id"].isin(eligible_ids)]
        selected = selected[normalized_unit(selected["orig_unit"]).eq("mg/100g")].copy()
        if selected.empty:
            continue
        invalid = selected[selected["standard_content"].gt(100_000)].copy()
        if not invalid.empty:
            invalid["axis_id"] = "foodb_compound:" + invalid["source_id"].astype(int).astype(str)
            invalid["value_g_per_100g"] = invalid["standard_content"] / 1000.0
            invalid["exclusion_reason"] = "Mass fraction exceeds 100 g/100g despite mg/100g label."
            invalid.to_csv(
                invalid_output_path,
                mode="w" if invalid_first else "a",
                header=invalid_first,
                index=False,
                columns=columns + ["exclusion_reason"],
            )
            invalid_first = False
            invalid_total += len(invalid)
        selected = selected[selected["standard_content"] <= 100_000].copy()
        if selected.empty:
            continue
        selected["axis_id"] = "foodb_compound:" + selected["source_id"].astype(int).astype(str)
        selected["value_g_per_100g"] = selected["standard_content"] / 1000.0
        if (selected["value_g_per_100g"] < 0).any() or (selected["value_g_per_100g"] > 100).any():
            raise AssertionError("Mass-bound filtering failed for FooDB compound values.")
        selected.to_csv(output_path, mode="w" if first else "a", header=first, index=False, columns=columns)
        first = False
        total += len(selected)
    if first:
        raise ValueError("No mass-normalizable FooDB compound values were found.")
    return total, invalid_total


def write_readme(output_dir: Path, axis_count: int, value_count: int, invalid_count: int, unit_audit: pd.DataFrame) -> None:
    total_numeric_axes = int(unit_audit["foodb_compound_id"].nunique())
    excluded_axes = int(
        unit_audit.loc[unit_audit["current_decision"].ne("retain"), "foodb_compound_id"].nunique()
    )
    output_dir.joinpath("README.md").write_text(
        f"""# Raw FooDB Mass Compound Candidates v2

This directory retains the complete raw FooDB compound pool that can be put on
the current mass-only model scale without an unverified conversion.

- Quantitatively observed compound entities in FooDB: {total_numeric_axes:,}.
- Retained mass-normalizable candidates: {axis_count:,} (`mg/100g` -> `g/100g`).
- Numeric entities excluded from the mass-only model: {excluded_axes:,}; see
  `compound_unit_audit.csv` for `uM`, IU, RE, NE, and alpha-TE records.
- Retained observed values: {value_count:,}.
- Values excluded for impossible mass fractions: {invalid_count:,}; see
  `invalid_mass_values.csv`.

No support threshold is applied here.  All {axis_count:,} axes remain in
`compound_axis_registry.csv`, with a recorded support tier.  The final corpus
builder must make the target-selection rule separately and audibly: rare axes
cannot be reconstructed reliably, 20--49-observation axes are candidates for
context-only use, and only sufficiently supported axes can be masked targets.

The input food entities are the provenance-level FooDB observations from
`fooddb_observations_v1`, not umbrella FooDB food IDs. Missing values remain
absent: this is a sparse observed-value table, not a zero-filled matrix.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION_DIR)
    parser.add_argument("--compound-catalog", type=Path, default=ROOT / "foodb_2020_04_07_csv" / "Compound.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    observation_dir = args.observation_dir.resolve()
    output_dir = args.output_dir.resolve()
    content_path = observation_dir / "Content.csv"
    if not content_path.exists():
        raise FileNotFoundError(f"Processed FooDB observation content is missing: {content_path}")
    if not args.compound_catalog.exists():
        raise FileNotFoundError(f"Raw FooDB compound catalog is missing: {args.compound_catalog}")
    output_dir.mkdir(parents=True, exist_ok=True)

    registry, unit_audit = build_axis_registry(content_path, args.compound_catalog)
    eligible_ids = set(registry["foodb_compound_id"].astype(int))
    registry.to_csv(output_dir / "compound_axis_registry.csv", index=False)
    unit_audit.to_csv(output_dir / "compound_unit_audit.csv", index=False)
    value_count, invalid_count = write_values(
        content_path,
        output_dir / "observed_compound_values.csv",
        output_dir / "invalid_mass_values.csv",
        eligible_ids,
    )
    write_readme(output_dir, len(registry), value_count, invalid_count, unit_audit)

    print(f"Retained raw FooDB mass compound axes: {len(registry):,}")
    print(f"Retained observed compound values: {value_count:,}")
    print(f"Excluded impossible mass values: {invalid_count:,}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
