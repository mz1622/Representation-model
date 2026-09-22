#!/usr/bin/env python3
"""Audit raw food-composition sources before any cross-database merge.

This script intentionally does not create training matrices or impute missing
values.  It writes source-level inventories, observed-value distributions, and
conservative nutrient-name crosswalk candidates to ``data/audits``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "data" / "audits"


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label: str
    release: str
    value_semantics: str
    evidence_tier: str
    url: str
    food_path: Path
    nutrient_path: Path | None
    food_nutrient_path: Path | None
    food_subset_path: Path | None = None


SOURCES = (
    SourceSpec(
        key="foodb_2020",
        label="FooDB",
        release="2020-04-07 export",
        value_semantics="Heterogeneous curated/literature content records; values may have different source methods and units.",
        evidence_tier="Curated composition and compound reference; not a harmonized analytical panel.",
        url="https://foodb.ca/",
        food_path=ROOT / "foodb_2020_04_07_csv" / "Food.csv",
        nutrient_path=ROOT / "foodb_2020_04_07_csv" / "Nutrient.csv",
        food_nutrient_path=None,
    ),
    SourceSpec(
        key="usda_foundation_2026_04",
        label="USDA FoodData Central Foundation Foods",
        release="2026-04-30",
        value_semantics="Food-level nutrient summaries with sample, acquisition, and analytical metadata available in companion tables.",
        evidence_tier="Analytical/reference composition data.",
        url="https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2026-04-30.zip",
        food_path=ROOT / "data/raw/usda/foundation_2026_04_30" / "food.csv",
        nutrient_path=ROOT / "data/raw/usda/foundation_2026_04_30" / "nutrient.csv",
        food_nutrient_path=ROOT / "data/raw/usda/foundation_2026_04_30" / "food_nutrient.csv",
        food_subset_path=ROOT / "data/raw/usda/foundation_2026_04_30" / "foundation_food.csv",
    ),
    SourceSpec(
        key="usda_sr_legacy_2018_04",
        label="USDA FoodData Central SR Legacy",
        release="2018-04 (final release)",
        value_semantics="Historic compiled food-component values from analyses, calculations, and literature.",
        evidence_tier="Curated reference composition data; frozen historical release.",
        url="https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_sr_legacy_food_csv_2018-04.zip",
        food_path=ROOT / "data" / "raw" / "usda" / "sr_legacy_2018" / "FoodData_Central_sr_legacy_food_csv_2018-04" / "food.csv",
        nutrient_path=ROOT / "data" / "raw" / "usda" / "sr_legacy_2018" / "FoodData_Central_sr_legacy_food_csv_2018-04" / "nutrient.csv",
        food_nutrient_path=ROOT / "data" / "raw" / "usda" / "sr_legacy_2018" / "FoodData_Central_sr_legacy_food_csv_2018-04" / "food_nutrient.csv",
    ),
    SourceSpec(
        key="usda_fndds_2021_2023",
        label="USDA FoodData Central FNDDS",
        release="2021-2023, downloaded 2024-10-31",
        value_semantics="Survey foods and beverages with composition values and recipe/input-food relationships.",
        evidence_tier="Derived survey/composite composition data; not independent laboratory samples.",
        url="https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_survey_food_csv_2024-10-31.zip",
        food_path=ROOT / "data" / "raw" / "usda" / "fndds_2021_2023" / "FoodData_Central_survey_food_csv_2024-10-31" / "food.csv",
        nutrient_path=ROOT / "data" / "raw" / "usda" / "fndds_2021_2023" / "FoodData_Central_survey_food_csv_2024-10-31" / "nutrient.csv",
        food_nutrient_path=ROOT / "data" / "raw" / "usda" / "fndds_2021_2023" / "FoodData_Central_survey_food_csv_2024-10-31" / "food_nutrient.csv",
    ),
)


def normalize_name(value: object) -> str:
    text = str(value).lower()
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(total|by calculation|retinol activity equivalents?|dietary)\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def quantiles(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: None for key in ("min", "p01", "p05", "median", "p95", "p99", "max")}
    probabilities = (0, 0.01, 0.05, 0.5, 0.95, 0.99, 1)
    names = ("min", "p01", "p05", "median", "p95", "p99", "max")
    return {name: float(value) for name, value in zip(names, np.quantile(array, probabilities))}


def file_status(path: Path | None) -> dict[str, object]:
    return {
        "path": str(path.relative_to(ROOT)) if path and path.exists() else str(path) if path else None,
        "present": bool(path and path.exists()),
        "size_bytes": path.stat().st_size if path and path.exists() else None,
    }


def audit_fdc_source(spec: SourceSpec) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    assert spec.nutrient_path and spec.food_nutrient_path
    foods = pd.read_csv(spec.food_path, low_memory=False)
    if spec.food_subset_path:
        subset = pd.read_csv(spec.food_subset_path, usecols=["fdc_id"])
        foods = foods.merge(subset.drop_duplicates(), on="fdc_id", how="inner")
    nutrients = pd.read_csv(spec.nutrient_path, low_memory=False)
    values = pd.read_csv(spec.food_nutrient_path, low_memory=False)
    values = values[values["fdc_id"].isin(foods["fdc_id"])]
    values["amount"] = pd.to_numeric(values["amount"], errors="coerce")
    if spec.key == "usda_fndds_2021_2023":
        nutrient_join_column = "nutrient_nbr"
        nutrients[nutrient_join_column] = pd.to_numeric(nutrients[nutrient_join_column], errors="raise")
        values["_nutrient_join_key"] = pd.to_numeric(values["nutrient_id"], errors="raise").astype(float)
        join_catalog = nutrients[["id", "name", "unit_name", nutrient_join_column]].rename(columns={"id": "axis_catalog_id"}).dropna(subset=[nutrient_join_column])
        enriched = values.merge(
            join_catalog,
            left_on="_nutrient_join_key",
            right_on=nutrient_join_column,
            how="left",
            validate="many_to_one",
        )
    else:
        nutrient_join_column = "id"
        join_catalog = nutrients[["id", "name", "unit_name"]].rename(columns={"id": "axis_catalog_id"})
        enriched = values.merge(
            join_catalog, left_on="nutrient_id", right_on="axis_catalog_id", how="left", validate="many_to_one"
        )
    valid = enriched["amount"].notna()
    observed = enriched.loc[valid, "amount"]
    distribution_rows = []
    for unit, group in enriched.loc[valid].groupby("unit_name", dropna=False):
        row = {
            "source": spec.key,
            "value_table": "food_nutrient",
            "unit": "<missing>" if pd.isna(unit) else str(unit),
            "records": int(len(group)),
            "positive": int((group["amount"] > 0).sum()),
            "explicit_zero": int((group["amount"] == 0).sum()),
            "negative": int((group["amount"] < 0).sum()),
        }
        row.update(quantiles(group["amount"]))
        distribution_rows.append(row)
    catalog = nutrients.copy()
    catalog["source"] = spec.key
    catalog["axis_id"] = catalog["id"].astype(str)
    catalog["axis_name"] = catalog["name"]
    catalog["unit"] = catalog["unit_name"]
    coverage = enriched.groupby("axis_catalog_id", dropna=False).agg(
        observed_records=("amount", lambda x: int(x.notna().sum())),
        positive_records=("amount", lambda x: int((x > 0).sum())),
        zero_records=("amount", lambda x: int((x == 0).sum())),
        min_value=("amount", "min"),
        median_value=("amount", "median"),
        max_value=("amount", "max"),
    ).reset_index()
    catalog = catalog.merge(coverage, left_on="id", right_on="axis_catalog_id", how="left")
    catalog["observed_records"] = catalog["observed_records"].fillna(0).astype(int)
    summary = {
        "source": spec.key,
        "food_records": int(len(foods)),
        "food_ids": int(foods["fdc_id"].nunique()),
        "food_categories": int(foods["food_category_id"].nunique()) if "food_category_id" in foods else None,
        "axis_catalog_size": int(len(nutrients)),
        "axes_with_observations": int(enriched["nutrient_id"].nunique()),
        "measurement_records": int(len(enriched)),
        "observed_numeric_records": int(valid.sum()),
        "explicit_zero_records": int((observed == 0).sum()),
        "positive_records": int((observed > 0).sum()),
        "negative_records": int((observed < 0).sum()),
        "missing_numeric_records": int((~valid).sum()),
        "nutrient_join_key": nutrient_join_column,
        "numeric_value_statistics": quantiles(observed),
    }
    return summary, catalog, pd.DataFrame(distribution_rows)


def audit_foodb(spec: SourceSpec, chunksize: int) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    root = spec.food_path.parent
    foods = pd.read_csv(spec.food_path, low_memory=False)
    nutrients = pd.read_csv(root / "Nutrient.csv", low_memory=False)
    compounds = pd.read_csv(root / "Compound.csv", low_memory=False)
    axis_catalog = pd.concat(
        [
            pd.DataFrame({"source": spec.key, "axis_kind": "nutrient", "axis_id": nutrients["id"].astype(str), "axis_name": nutrients["name"]}),
            pd.DataFrame({"source": spec.key, "axis_kind": "compound", "axis_id": compounds["id"].astype(str), "axis_name": compounds["name"]}),
        ],
        ignore_index=True,
    )
    counters: Counter[tuple[str, str]] = Counter()
    units: Counter[tuple[str, str]] = Counter()
    unit_numeric_counts: Counter[tuple[str, str, str]] = Counter()
    samples: dict[tuple[str, str], list[float]] = defaultdict(list)
    max_sample = 100_000
    per_axis: dict[tuple[str, int], dict[str, float]] = defaultdict(lambda: {"records": 0, "observed": 0, "positive": 0, "zero": 0, "negative": 0, "min": math.inf, "max": -math.inf})
    content_path = root / "Content.csv"
    usecols = ["source_id", "source_type", "standard_content", "orig_unit"]
    for chunk in pd.read_csv(content_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        chunk["standard_content"] = pd.to_numeric(chunk["standard_content"], errors="coerce")
        chunk["source_type"] = chunk["source_type"].fillna("<missing>").astype(str)
        chunk["orig_unit"] = chunk["orig_unit"].fillna("<missing>").astype(str)
        numeric = chunk["standard_content"].notna()
        for (kind, unit), group in chunk.groupby(["source_type", "orig_unit"], dropna=False):
            values = group["standard_content"]
            valid_values = values[values.notna()].to_numpy(dtype=float)
            key = (kind, unit)
            counters[(kind, "records")] += int(len(group))
            counters[(kind, "observed")] += int(len(valid_values))
            counters[(kind, "positive")] += int((valid_values > 0).sum())
            counters[(kind, "zero")] += int((valid_values == 0).sum())
            counters[(kind, "negative")] += int((valid_values < 0).sum())
            units[key] += int(len(group))
            unit_numeric_counts[(kind, unit, "positive")] += int((valid_values > 0).sum())
            unit_numeric_counts[(kind, unit, "zero")] += int((valid_values == 0).sum())
            unit_numeric_counts[(kind, unit, "negative")] += int((valid_values < 0).sum())
            remaining = max_sample - len(samples[key])
            if remaining > 0:
                samples[key].extend(valid_values[:remaining].tolist())
        for (kind, source_id), group in chunk.groupby(["source_type", "source_id"], dropna=False):
            values = group["standard_content"].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            key = (str(kind), int(source_id))
            item = per_axis[key]
            item["records"] += len(values)
            item["observed"] += len(finite)
            item["positive"] += int((finite > 0).sum())
            item["zero"] += int((finite == 0).sum())
            item["negative"] += int((finite < 0).sum())
            if len(finite):
                item["min"] = min(item["min"], float(finite.min()))
                item["max"] = max(item["max"], float(finite.max()))
    distribution_rows = []
    for (kind, unit), count in sorted(units.items()):
        row = {
            "source": spec.key,
            "value_table": f"Content:{kind}",
            "unit": unit,
            "records": count,
            "positive": unit_numeric_counts[(kind, unit, "positive")],
            "explicit_zero": unit_numeric_counts[(kind, unit, "zero")],
            "negative": unit_numeric_counts[(kind, unit, "negative")],
        }
        row.update(quantiles(samples[(kind, unit)]))
        distribution_rows.append(row)
    coverage_rows = []
    names = {
        "Nutrient": nutrients.set_index("id")["name"].to_dict(),
        "Compound": compounds.set_index("id")["name"].to_dict(),
    }
    for (kind, source_id), item in per_axis.items():
        coverage_rows.append(
            {
                "source": spec.key,
                "axis_kind": kind.lower(),
                "axis_id": str(source_id),
                "axis_name": names.get(kind, {}).get(source_id, "<unresolved source_id>"),
                "measurement_records": int(item["records"]),
                "observed_records": int(item["observed"]),
                "positive_records": int(item["positive"]),
                "zero_records": int(item["zero"]),
                "negative_records": int(item["negative"]),
                "min_value": None if math.isinf(item["min"]) else item["min"],
                "max_value": None if math.isinf(item["max"]) else item["max"],
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    axis_catalog = axis_catalog.merge(coverage, on=["source", "axis_kind", "axis_id", "axis_name"], how="left")
    for column in ("measurement_records", "observed_records", "positive_records", "zero_records", "negative_records"):
        axis_catalog[column] = axis_catalog[column].fillna(0).astype(int)
    summary = {
        "source": spec.key,
        "food_records": int(len(foods)),
        "food_ids": int(foods["id"].nunique()),
        "food_categories": int(foods["food_group"].nunique()),
        "axis_catalog_size": int(len(axis_catalog)),
        "axes_with_observations": int((axis_catalog["observed_records"] > 0).sum()),
        "measurement_records": int(sum(counters[(kind, "records")] for kind in ("Nutrient", "Compound"))),
        "observed_numeric_records": int(sum(counters[(kind, "observed")] for kind in ("Nutrient", "Compound"))),
        "explicit_zero_records": int(sum(counters[(kind, "zero")] for kind in ("Nutrient", "Compound"))),
        "positive_records": int(sum(counters[(kind, "positive")] for kind in ("Nutrient", "Compound"))),
        "negative_records": int(sum(counters[(kind, "negative")] for kind in ("Nutrient", "Compound"))),
        "missing_numeric_records": int(sum(counters[(kind, "records")] - counters[(kind, "observed")] for kind in ("Nutrient", "Compound"))),
        "numeric_value_statistics": None,
    }
    return summary, axis_catalog, pd.DataFrame(distribution_rows)


def make_crosswalk(catalog: pd.DataFrame) -> pd.DataFrame:
    candidates = catalog.copy()
    candidates["normalized_name"] = candidates["axis_name"].map(normalize_name)
    candidates["unit"] = candidates.get("unit", pd.Series(index=candidates.index, dtype=object)).fillna("<unknown>")
    grouped = candidates.groupby("normalized_name", dropna=False)
    rows = []
    for name, group in grouped:
        source_set = set(group["source"])
        if "foodb_2020" not in source_set or not any(source.startswith("usda_") for source in source_set):
            continue
        rows.append(
            {
                "normalized_name": name,
                "sources": " | ".join(sorted(group["source"].unique())),
                "source_axis_names": " | ".join(f"{r.source}:{r.axis_name}" for r in group.itertuples()),
                "units": " | ".join(sorted({str(unit) for unit in group["unit"]})),
                "candidate_type": "name-normalized only; requires definition and unit review",
            }
        )
    return pd.DataFrame(rows).sort_values("normalized_name") if rows else pd.DataFrame()


def make_usda_axis_comparison(catalog: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare USDA distributions by nutrient identity without claiming food equivalence."""
    usda = catalog[catalog["source"].str.startswith("usda_")].copy()
    columns = ["source", "axis_id", "axis_name", "unit", "observed_records", "positive_records", "zero_records", "min_value", "median_value", "max_value"]
    usda = usda[columns]
    metadata = usda.groupby("axis_id", as_index=False).agg(
        axis_name=("axis_name", "first"),
        unit=("unit", "first"),
    )
    wide_parts = []
    for metric in ("observed_records", "positive_records", "zero_records", "min_value", "median_value", "max_value"):
        pivot = usda.pivot(index="axis_id", columns="source", values=metric).add_prefix(f"{metric}__").reset_index()
        wide_parts.append(pivot)
    comparison = metadata
    for part in wide_parts:
        comparison = comparison.merge(part, on="axis_id", how="left")
    observed_columns = [column for column in comparison if column.startswith("observed_records__")]
    observed_matrix = comparison[observed_columns].fillna(0).gt(0)
    comparison["sources_with_observations"] = observed_matrix.sum(axis=1)
    comparison["common_to_all_three"] = observed_matrix.all(axis=1)
    source_coverage = pd.DataFrame(
        [
            {
                "comparison": "USDA nutrients observed in all three sources",
                "axis_count": int(comparison["common_to_all_three"].sum()),
            },
            {
                "comparison": "USDA nutrients observed in at least two sources",
                "axis_count": int((comparison["sources_with_observations"] >= 2).sum()),
            },
            {
                "comparison": "USDA nutrients observed in exactly one source",
                "axis_count": int((comparison["sources_with_observations"] == 1).sum()),
            },
        ]
    )
    return comparison.sort_values(["common_to_all_three", "axis_name"], ascending=[False, True]), source_coverage


def load_food_names(spec: SourceSpec) -> pd.DataFrame:
    if spec.key == "foodb_2020":
        foods = pd.read_csv(spec.food_path, usecols=["id", "name"], low_memory=False)
        return foods.rename(columns={"id": "food_id", "name": "food_name"}).assign(source=spec.key)
    foods = pd.read_csv(spec.food_path, usecols=["fdc_id", "description"], low_memory=False)
    if spec.food_subset_path:
        subset = pd.read_csv(spec.food_subset_path, usecols=["fdc_id"])
        foods = foods.merge(subset.drop_duplicates(), on="fdc_id", how="inner")
    return foods.rename(columns={"fdc_id": "food_id", "description": "food_name"}).assign(source=spec.key)


def make_food_name_overlap(specs: Iterable[SourceSpec]) -> tuple[pd.DataFrame, pd.DataFrame]:
    food_names = pd.concat([load_food_names(spec) for spec in specs], ignore_index=True)
    food_names["normalized_food_name"] = food_names["food_name"].map(normalize_name)
    food_names = food_names[food_names["normalized_food_name"].ne("")]
    foo = food_names[food_names["source"] == "foodb_2020"]
    candidates = []
    summaries = []
    for source, group in food_names[food_names["source"] != "foodb_2020"].groupby("source"):
        overlap = foo.merge(group, on="normalized_food_name", suffixes=("_foodb", "_other"))
        summaries.append(
            {
                "left_source": "foodb_2020",
                "right_source": source,
                "exact_normalized_name_pairs": int(len(overlap)),
                "unique_foodb_foods": int(overlap["food_id_foodb"].nunique()),
                "unique_other_foods": int(overlap["food_id_other"].nunique()),
                "interpretation": "Candidate only; names do not establish same product, preparation, sample, or assay basis.",
            }
        )
        if len(overlap):
            candidates.append(
                overlap[["normalized_food_name", "food_id_foodb", "food_name_foodb", "food_id_other", "food_name_other"]].assign(other_source=source))
    candidate_frame = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
    return pd.DataFrame(summaries), candidate_frame


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small audit table without requiring pandas' optional tabulate dependency."""
    if frame.empty:
        return "_No rows._"
    prepared = frame.fillna("").astype(str).applymap(lambda value: value.replace("|", "\\|"))
    columns = prepared.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in prepared.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *rows])


def build_markdown(
    inventory: pd.DataFrame,
    paths: list[dict[str, object]],
    crosswalk: pd.DataFrame,
    usda_coverage: pd.DataFrame,
    name_overlap: pd.DataFrame,
) -> str:
    lines = [
        "# Raw Food-Composition Dataset Audit",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Scope",
        "",
        "This report describes raw sources only. It does not create a merged training matrix, impute missing values, or treat name matches as equivalence.",
        "",
        "## Source Inventory",
        "",
        markdown_table(inventory),
        "",
        "## Immediate Data-Construction Decisions",
        "",
        "1. Keep FoodDB, USDA Foundation, SR Legacy, FNDDS, Branded, and Open Food Facts as separate provenance strata.",
        "2. Store explicit numeric zero separately from an absent measurement. Neither should be inferred from the other.",
        "3. Do not pool raw values across units or databases before a reviewed axis-definition and unit crosswalk exists.",
        "4. Treat FoodDB compounds as sparse heterogeneous observations, not as a dense negative-label matrix.",
        "5. Treat FNDDS recipes and Branded/Open Food Facts label products as derived or label data, not independent analytical replicates.",
        "",
        "## Candidate Cross-Database Nutrient Matches",
        "",
        f"The candidate file contains {len(crosswalk)} normalized-name overlaps. Every row still requires definition, unit, basis, and provenance review before values can be compared or merged.",
        "",
        "## USDA Within-Family Axis Coverage",
        "",
        markdown_table(usda_coverage),
        "",
        "These are coverage overlaps, not agreement estimates: Foundation, SR Legacy, and FNDDS describe different food populations and provenance levels.",
        "",
        "## FoodDB--USDA Exact Name Candidates",
        "",
        markdown_table(name_overlap),
        "",
        "## Acquisition Status",
        "",
        markdown_table(pd.DataFrame(paths)),
        "",
        "## Next Audit Gates",
        "",
        "1. Review all candidate nutrient matches and record exact / aggregate / incompatible mappings.",
        "2. Convert values only to explicit mass or energy bases, retaining original unit and conversion rule.",
        "3. Audit duplicate and near-duplicate foods across FoodDB and FDC before defining food-level splits.",
        "4. Run physical plausibility checks by nutrient and unit before any model training.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=250_000)
    args = parser.parse_args()
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    inventory_rows = []
    catalog_parts = []
    distribution_parts = []
    manifest_rows = []
    for spec in SOURCES:
        paths = [spec.food_path, spec.nutrient_path, spec.food_nutrient_path, spec.food_subset_path]
        present = all(path is None or path.exists() for path in paths)
        manifest_rows.append(
            {
                "source": spec.key,
                "release": spec.release,
                "available": present,
                "evidence_tier": spec.evidence_tier,
                "url": spec.url,
                "paths": json.dumps([file_status(path) for path in paths]),
            }
        )
        if not present:
            continue
        if spec.key == "foodb_2020":
            summary, catalog, distributions = audit_foodb(spec, args.chunksize)
        else:
            summary, catalog, distributions = audit_fdc_source(spec)
            catalog["axis_kind"] = "nutrient"
        inventory_rows.append(
            {
                "source": spec.key,
                "label": spec.label,
                "release": spec.release,
                "evidence_tier": spec.evidence_tier,
                **{key: value for key, value in summary.items() if key not in {"source", "numeric_value_statistics"}},
            }
        )
        catalog_parts.append(catalog)
        distribution_parts.append(distributions)
    manifest = pd.DataFrame(manifest_rows)
    inventory = pd.DataFrame(inventory_rows)
    catalog = pd.concat(catalog_parts, ignore_index=True, sort=False)
    distributions = pd.concat(distribution_parts, ignore_index=True, sort=False)
    crosswalk = make_crosswalk(catalog)
    usda_comparison, usda_coverage = make_usda_axis_comparison(catalog)
    name_overlap, name_candidates = make_food_name_overlap(SOURCES)
    manifest.to_csv(AUDIT_DIR / "source_manifest.csv", index=False)
    inventory.to_csv(AUDIT_DIR / "dataset_inventory.csv", index=False)
    catalog.to_csv(AUDIT_DIR / "axis_catalog_and_coverage.csv", index=False)
    distributions.to_csv(AUDIT_DIR / "value_distributions_by_unit.csv", index=False)
    crosswalk.to_csv(AUDIT_DIR / "nutrient_name_crosswalk_candidates.csv", index=False)
    usda_comparison.to_csv(AUDIT_DIR / "usda_axis_distribution_comparison.csv", index=False)
    usda_coverage.to_csv(AUDIT_DIR / "usda_axis_coverage_summary.csv", index=False)
    name_overlap.to_csv(AUDIT_DIR / "food_name_exact_overlap_summary.csv", index=False)
    name_candidates.to_csv(AUDIT_DIR / "food_name_exact_overlap_candidates.csv", index=False)
    (AUDIT_DIR / "raw_dataset_audit.md").write_text(
        build_markdown(inventory, manifest_rows, crosswalk, usda_coverage, name_overlap), encoding="utf-8"
    )
    print(f"Wrote audit artifacts to {AUDIT_DIR}")
    print(inventory.to_string(index=False))
    print(f"Candidate normalized-name overlaps: {len(crosswalk)}")


if __name__ == "__main__":
    main()
