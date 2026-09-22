"""Freeze a prediction-axis panel and audit source food records without pooling.

This module intentionally stops before train/validation construction.  It
creates exact-name food groups as reviewable deduplication candidates, retains
every eligible source measurement, and records rather than resolves within-cell
cross-source disagreement.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import re
import shutil
from typing import Any, Iterable
import unicodedata

import numpy as np
import pandas as pd

from .util import require_columns, sha256_file, stable_id, write_csv, write_json


PANEL_FREEZE_VERSION = "frozen_prediction_axis_panel_v1"
FOOD_AUDIT_VERSION = "frozen_prediction_panel_food_audit_v1"

# These aliases only resolve a documented spelling or source-label convention
# to a target already present in the frozen panel.  They are not fuzzy matches.
EXACT_LABEL_TO_TARGET_NAME = {
    "fat": "Fat, total",
    "fat total": "Fat, total",
    "fat total lipids": "Fat, total",
    "total lipid fat": "Fat, total",
    "total lipids fat": "Fat, total",
    "lipids": "Fat, total",
    "protein": "Protein, total",
    "proteins": "Protein, total",
    "protein total": "Protein, total",
    "dietary fibre": "Dietary fibre, total",
    "dietary fiber": "Dietary fibre, total",
    "fiber dietary": "Dietary fibre, total",
    "total dietary fibre": "Dietary fibre, total",
    "total dietary fiber": "Dietary fibre, total",
    "moisture": "Water",
    "water g 100g": "Water",
    "ash total": "Ash",
    "ash g 100g": "Ash",
    "alcohol": "Ethanol",
    "alcohol ethyl": "Ethanol",
}

# `Carbohydrate, by difference` is intentionally absent.  It is a calculated
# expression and is not the direct mass target in the frozen panel.
INFOODS_TAG_TO_TARGET_NAME = {
    "PROCNT": "Protein, total",
    "FAT": "Fat, total",
    "CHO-": "Carbohydrate, total",
    "CHOAVL": "Available carbohydrate",
    "FIBTG": "Dietary fibre, total",
    "WATER": "Water",
    "ASH": "Ash",
    "ALC": "Ethanol",
}

FOODB_SOURCE_ID_TO_TARGET_NAME = {
    "Nutrient:1": "Fat, total",
    "Nutrient:2": "Protein, total",
    "Nutrient:3": "Carbohydrate, total",
    "Nutrient:5": "Dietary fibre, total",
    "Nutrient:39": "Ash",
    "Nutrient:31": "Docosanoic acid (22:0)",
    "Nutrient:36": "Docosapentaenoic acid (DPA; 22:5n-3)",
}

EXCLUDED_EXPRESSION_KEYS = {
    "carbohydrate by difference",
    "carbohydrate total by difference",
    "energy",
    "energy kilocalories",
    "energy kcal",
    "energy regulation eu no 1169 2011 kj 100g",
    "energy regulation eu no 1169 2011 kcal 100g",
}


def exact_label_key(value: Any) -> str:
    """Normalize only typography for an exact lexical identity comparison."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def exact_food_name_key(value: Any) -> str:
    """Use only Unicode, case, and whitespace normalization for food names."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"\s+", " ", text)


def _split_panel_aliases(value: Any) -> Iterable[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [item.strip() for item in re.split(r"\s*[;|]\s*", str(value)) if item.strip()]


def _target_id_by_name(panel: pd.DataFrame) -> dict[str, str]:
    result = dict(zip(panel.canonical_name.map(exact_label_key), panel.target_axis_id, strict=True))
    if len(result) != len(panel):
        raise ValueError("Frozen panel canonical names are not unique after exact normalization")
    return result


def build_exact_component_mapping(panel: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    """Map source components to one frozen axis using only exact evidence."""
    require_columns(panel, ["target_axis_id", "canonical_name", "original_axis_names", "aliases_for_review"], "panel")
    require_columns(
        components,
        ["component_observation_id", "source_key", "source_component_id", "original_name", "infoods_tag"],
        "source component catalogue",
    )
    by_name: dict[str, set[str]] = defaultdict(set)
    for row in panel.itertuples(index=False):
        for field in ("canonical_name", "original_axis_names", "aliases_for_review"):
            for alias in _split_panel_aliases(getattr(row, field)):
                by_name[exact_label_key(alias)].add(row.target_axis_id)
    target_by_name = _target_id_by_name(panel)
    for alias, target_name in EXACT_LABEL_TO_TARGET_NAME.items():
        target_key = exact_label_key(target_name)
        if target_key in target_by_name:
            by_name[exact_label_key(alias)].add(target_by_name[target_key])

    rows: list[dict[str, Any]] = []
    for row in components.fillna("").itertuples(index=False):
        label_key = exact_label_key(row.original_name)
        target_ids = set(by_name.get(label_key, set()))
        mapping_basis = "panel_exact_name_or_registered_alias"

        if label_key in EXCLUDED_EXPRESSION_KEYS:
            target_ids = set()
            mapping_basis = "excluded_non_direct_expression"
        elif row.source_key == "foodb" and row.source_component_id in FOODB_SOURCE_ID_TO_TARGET_NAME:
            target_key = exact_label_key(FOODB_SOURCE_ID_TO_TARGET_NAME[row.source_component_id])
            if target_key in target_by_name:
                target_ids = {target_by_name[target_key]}
                mapping_basis = "reviewed_foodb_source_component_id"
        elif row.infoods_tag in INFOODS_TAG_TO_TARGET_NAME:
            target_key = exact_label_key(INFOODS_TAG_TO_TARGET_NAME[row.infoods_tag])
            if target_key in target_by_name:
                target_ids = {target_by_name[target_key]}
                mapping_basis = "reviewed_INFOODS_tag"

        if len(target_ids) == 1:
            target_axis_id = next(iter(target_ids))
            status = "mapped_exact"
        elif len(target_ids) > 1:
            target_axis_id = ""
            status = "ambiguous_exact_candidate_requires_review"
        else:
            target_axis_id = ""
            status = "not_mapped_by_exact_rule"

        rows.append(
            {
                "component_observation_id": row.component_observation_id,
                "source_key": row.source_key,
                "source_component_id": row.source_component_id,
                "source_original_name": row.original_name,
                "source_infoods_tag": row.infoods_tag,
                "source_axis_label_key": label_key,
                "target_axis_id": target_axis_id,
                "mapping_status": status,
                "mapping_basis": mapping_basis,
                "candidate_target_axis_ids": ";".join(sorted(target_ids)),
            }
        )
    mapped = pd.DataFrame(rows)
    if mapped.component_observation_id.duplicated().any():
        raise AssertionError("Source component mapping must have one row per component observation")
    return mapped


def build_exact_name_food_groups(foods: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create exact-name groups and retain every original food observation."""
    require_columns(foods, ["food_observation_id", "source_key", "source_food_id", "original_name"], "food observations")
    result = foods.copy().fillna("")
    result["exact_food_name_key"] = result.original_name.map(exact_food_name_key)
    blank = result.exact_food_name_key.eq("")
    result["exact_name_group_id"] = result.exact_food_name_key.map(
        lambda key: stable_id("food_exact_name", key) if key else ""
    )
    result.loc[blank, "exact_name_group_id"] = result.loc[blank].apply(
        lambda row: stable_id("food_unmatched_name", row.source_key, row.food_observation_id), axis=1
    )
    result["exact_name_group_status"] = np.where(
        blank,
        "blank_name_not_deduplicated",
        "exact_name_candidate_group",
    )

    facet_columns = [
        column for column in (
            "scientific_name", "food_group", "food_subgroup", "food_type", "part", "maturity",
            "processing", "cooking", "preservation", "physical_state", "packing_medium", "geography",
            "cultivar", "recipe_status", "brand_status", "edible_status",
        ) if column in result.columns
    ]

    def nonempty_unique(values: pd.Series) -> list[str]:
        return sorted({str(value).strip() for value in values if str(value).strip()})

    groups: list[dict[str, Any]] = []
    for group_id, frame in result.groupby("exact_name_group_id", sort=True):
        row: dict[str, Any] = {
            "exact_name_group_id": group_id,
            "exact_food_name_key": frame.exact_food_name_key.iloc[0],
            "display_name": sorted(nonempty_unique(frame.original_name), key=str.casefold)[0] if nonempty_unique(frame.original_name) else "",
            "food_observation_count": len(frame),
            "source_count": frame.source_key.nunique(),
            "source_keys": ";".join(sorted(frame.source_key.unique())),
            "food_observation_ids": ";".join(sorted(frame.food_observation_id)),
            "source_food_ids": ";".join(sorted(frame.source_food_id.astype(str))),
            "original_names": " | ".join(nonempty_unique(frame.original_name)),
        }
        disagreements = []
        for column in facet_columns:
            values = nonempty_unique(frame[column])
            row[f"{column}_values"] = " | ".join(values)
            if len(values) > 1:
                disagreements.append(column)
        row["facet_disagreement_fields"] = ";".join(disagreements)
        row["has_facet_disagreement"] = bool(disagreements)
        groups.append(row)
    group_frame = pd.DataFrame(groups)
    return result, group_frame


def _as_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.casefold().eq("true")


def eligible_direct_mass_measurements(frame: pd.DataFrame) -> pd.Series:
    """Select observed, directly normalized fresh-weight mass measurements only."""
    required = [
        "normalized_value_g_per_100g", "value_status", "is_censored", "is_range_only",
        "conversion_status", "measurement_modality",
    ]
    require_columns(frame, required, "measurement table")
    value = pd.to_numeric(frame.normalized_value_g_per_100g, errors="coerce")
    observed = frame.value_status.astype(str).isin(["observed", "explicit_zero"])
    return (
        value.notna()
        & value.ge(0)
        & observed
        & ~_as_bool(frame.is_censored)
        & ~_as_bool(frame.is_range_only)
        & frame.conversion_status.astype(str).eq("converted_exact_mass")
        & frame.measurement_modality.astype(str).eq("mass_fraction_fresh_weight")
    )


def _distinct_value_count(values: pd.Series, *, rtol: float = 1e-9, atol: float = 1e-12) -> int:
    sorted_values = sorted(pd.to_numeric(values, errors="coerce").dropna().tolist())
    representatives: list[float] = []
    for value in sorted_values:
        if not representatives or not np.isclose(value, representatives[-1], rtol=rtol, atol=atol):
            representatives.append(value)
    return len(representatives)


def summarize_exact_name_axis_cells(measurements: pd.DataFrame) -> pd.DataFrame:
    """Describe but never resolve measurements sharing a name-group and target axis."""
    require_columns(
        measurements,
        ["exact_name_group_id", "target_axis_id", "source_key", "normalized_value_g_per_100g", "measurement_id"],
        "mapped measurements",
    )
    rows: list[dict[str, Any]] = []
    for (food_group_id, target_axis_id), frame in measurements.groupby(
        ["exact_name_group_id", "target_axis_id"], sort=True
    ):
        values = pd.to_numeric(frame.normalized_value_g_per_100g, errors="raise")
        source_count = frame.source_key.nunique()
        distinct_count = _distinct_value_count(values)
        minimum, maximum, median = float(values.min()), float(values.max()), float(values.median())
        if median == 0:
            relative_span = float("inf") if maximum > minimum else 0.0
        else:
            relative_span = (maximum - minimum) / abs(median)
        disagreement = source_count > 1 and distinct_count > 1
        rows.append(
            {
                "exact_name_group_id": food_group_id,
                "target_axis_id": target_axis_id,
                "measurement_count": len(frame),
                "source_count": source_count,
                "source_keys": ";".join(sorted(frame.source_key.unique())),
                "measurement_ids": ";".join(sorted(frame.measurement_id)),
                "minimum_g_per_100g": minimum,
                "median_g_per_100g": median,
                "maximum_g_per_100g": maximum,
                "tolerance_distinct_value_count": distinct_count,
                "relative_span_vs_median": relative_span,
                "cell_status": (
                    "cross_source_value_disagreement_preserved" if disagreement
                    else "cross_source_same_value_preserved" if source_count > 1
                    else "single_source_measurement_preserved"
                ),
            }
        )
    return pd.DataFrame(rows)


def _source_distribution(foods: pd.DataFrame, mapped_measurements: pd.DataFrame) -> pd.DataFrame:
    covered_foods = set(mapped_measurements.food_observation_id)
    rows: list[dict[str, Any]] = []
    for source_key, frame in foods.groupby("source_key", sort=True):
        observed = mapped_measurements[mapped_measurements.source_key.eq(source_key)]
        rows.append(
            {
                "source_key": source_key,
                "raw_food_observations": len(frame),
                "food_observations_with_frozen_axis_measurement": int(frame.food_observation_id.isin(covered_foods).sum()),
                "exact_name_food_groups": frame.exact_name_group_id.nunique(),
                "exact_name_groups_with_frozen_axis_measurement": observed.exact_name_group_id.nunique(),
                "mapped_measurements": len(observed),
                "frozen_target_axes_observed": observed.target_axis_id.nunique(),
            }
        )
    return pd.DataFrame(rows)


def _food_group_distribution(foods: pd.DataFrame, mapped_measurements: pd.DataFrame) -> pd.DataFrame:
    covered_foods = set(mapped_measurements.food_observation_id)
    result = foods.copy()
    result["food_group_display"] = result.get("food_group", "").fillna("").astype(str).str.strip().replace("", "<not reported>")
    result["has_frozen_axis_measurement"] = result.food_observation_id.isin(covered_foods)
    summary = (
        result.groupby(["source_key", "food_group_display"], dropna=False)
        .agg(
            food_observations=("food_observation_id", "size"),
            food_observations_with_frozen_axis_measurement=("has_frozen_axis_measurement", "sum"),
            exact_name_food_groups=("exact_name_group_id", "nunique"),
        )
        .reset_index()
        .sort_values(["source_key", "food_observations", "food_group_display"], ascending=[True, False, True])
    )
    return summary


def _category_metadata_coverage(foods: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "food_group", "food_subgroup", "food_type", "scientific_name", "part", "maturity", "processing",
        "cooking", "preservation", "physical_state", "packing_medium", "geography", "cultivar", "recipe_status",
    ]
    rows = []
    for source_key, frame in foods.groupby("source_key", sort=True):
        row = {"source_key": source_key, "food_observations": len(frame)}
        for field in fields:
            row[f"{field}_present"] = int(
                frame.get(field, pd.Series("", index=frame.index)).fillna("").astype(str).str.strip().ne("").sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, limit: int = 30) -> str:
    view = frame.head(limit).copy()
    if view.empty:
        return "_No rows._"
    view = view.replace({np.nan: ""})
    columns = list(view.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in view.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|").replace("\n", " ") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_report(
    report_path: Path,
    *,
    freeze_manifest: dict[str, Any],
    source_distribution: pd.DataFrame,
    group_summary: pd.DataFrame,
    conflict_summary: pd.DataFrame,
    food_group_distribution: pd.DataFrame,
    mapping: pd.DataFrame,
    mapped_measurements: pd.DataFrame,
) -> None:
    conflict_count = int(conflict_summary.cell_status.eq("cross_source_value_disagreement_preserved").sum())
    lines = [
        "# Frozen Prediction Panel Food Audit",
        "",
        "## Scope",
        "",
        "This audit freezes the current prediction-axis panel before any food-level modelling, split construction, numerical pooling, or source-priority selection.",
        "Food records are grouped only when their names are identical after Unicode normalization, case-folding, and whitespace normalization. Punctuation, word order, translations, taxonomy, food facets, and semantic similarity are not used for deduplication.",
        "",
        f"- Frozen panel version: `{freeze_manifest['panel_version']}`",
        f"- Frozen panel SHA-256: `{freeze_manifest['source_panel_sha256']}`",
        f"- Direct prediction axes: {freeze_manifest['direct_prediction_axes']}",
        f"- Exact-mapped source component observations: {int(mapping.mapping_status.eq('mapped_exact').sum())}",
        f"- Preserved direct mass measurements on frozen axes: {len(mapped_measurements):,}",
        f"- Exact-name food groups: {len(group_summary):,}",
        f"- Cross-source food-axis disagreements retained without resolution: {conflict_count:,}",
        "",
        "## Source Distribution",
        "",
        _markdown_table(source_distribution),
        "",
        "## Food Categories",
        "",
        "Food-group labels are shown as supplied by each source. They are not treated as a cross-source ontology in this audit; a future FoodOn/LanguaL mapping must be reviewed separately.",
        "",
        _markdown_table(food_group_distribution, limit=30),
        "",
        "## Exact-Name Deduplication and Conflicts",
        "",
        "An exact-name group retains all member food records and their measurements. If a group has distinct food facets, that fact is flagged but does not change this first-pass grouping. A conflict means that two or more sources report non-equal normalized values for the same exact-name group and frozen target axis. Both values and all source provenance remain in `target_measurement_observations.csv.gz`; no source is selected as the winner.",
        "",
        _markdown_table(
            conflict_summary[conflict_summary.cell_status.eq("cross_source_value_disagreement_preserved")],
            limit=20,
        ),
        "",
        "## Deliverables",
        "",
        "- `source_component_to_frozen_target_axis.csv.gz`: exact identity mapping ledger; no semantic/fuzzy mappings are accepted.",
        "- `food_observation_to_exact_name_group.csv.gz` and `exact_name_food_groups.csv`: first-pass food deduplication ledger.",
        "- `target_measurement_observations.csv.gz`: every retained measurement, including conflicting values and source metadata.",
        "- `exact_name_food_axis_cell_summary.csv.gz` and `cross_source_axis_conflicts.csv.gz`: conflict ledger without numerical arbitration.",
        "- `source_distribution.csv`, `food_group_distribution_by_source.csv`, and `food_category_metadata_coverage.csv`: source and food-category distributions.",
        "",
        "This is an audit artifact, not a train/validation dataset and not a pooled canonical profile.",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def freeze_prediction_panel(source_panel_path: Path, freeze_dir: Path) -> dict[str, Any]:
    """Create or verify a byte-identical immutable panel snapshot."""
    source_panel_path = source_panel_path.resolve()
    source_hash = sha256_file(source_panel_path)
    frozen_panel = freeze_dir / "frozen_prediction_axis_registry.csv"
    manifest_path = freeze_dir / "freeze_manifest.json"
    if manifest_path.exists() or frozen_panel.exists():
        if not manifest_path.exists() or not frozen_panel.exists():
            raise FileExistsError(f"Incomplete frozen panel directory: {freeze_dir}")
        existing = pd.read_json(manifest_path, typ="series").to_dict()
        if existing.get("source_panel_sha256") != source_hash or sha256_file(frozen_panel) != source_hash:
            raise ValueError("Existing frozen panel does not match the requested source panel")
        return existing

    panel = pd.read_csv(source_panel_path)
    require_columns(panel, ["target_axis_id", "direct_prediction_target"], "source prediction panel")
    if not panel.target_axis_id.is_unique or not panel.direct_prediction_target.astype(bool).all():
        raise ValueError("Prediction panel must contain unique direct target axes")
    freeze_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(source_panel_path, frozen_panel)
    manifest = {
        "panel_version": PANEL_FREEZE_VERSION,
        "source_panel_path": str(source_panel_path),
        "source_panel_sha256": source_hash,
        "frozen_panel_sha256": sha256_file(frozen_panel),
        "direct_prediction_axes": len(panel),
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_policy": "Do not modify this file. Create a new panel version for any target-axis change.",
    }
    write_json(manifest, manifest_path)
    return manifest


def audit_frozen_prediction_panel_foods(
    *,
    source_panel_path: Path,
    freeze_dir: Path,
    staging_dir: Path,
    output_dir: Path,
    report_dir: Path,
) -> dict[str, Any]:
    """Run the non-pooling food audit against an immutable axis snapshot."""
    if output_dir.exists():
        raise FileExistsError(f"Audit output already exists: {output_dir}")
    freeze_manifest = freeze_prediction_panel(source_panel_path, freeze_dir)
    panel = pd.read_csv(freeze_dir / "frozen_prediction_axis_registry.csv")

    food_parts, component_parts = [], []
    source_files: dict[str, dict[str, str]] = {}
    for source_dir in sorted(path for path in staging_dir.iterdir() if path.is_dir()):
        food_path = source_dir / "food_observation.csv.gz"
        component_path = source_dir / "component_observation.csv.gz"
        if not food_path.exists() or not component_path.exists():
            raise FileNotFoundError(f"Missing required staging catalogue under {source_dir}")
        foods = pd.read_csv(food_path, low_memory=False, keep_default_na=False, na_values=[""])
        components = pd.read_csv(component_path, low_memory=False, keep_default_na=False, na_values=[""])
        if foods.source_key.nunique() != 1 or foods.source_key.iloc[0] != source_dir.name:
            raise ValueError(f"Food catalogue source mismatch in {food_path}")
        if components.source_key.nunique() != 1 or components.source_key.iloc[0] != source_dir.name:
            raise ValueError(f"Component catalogue source mismatch in {component_path}")
        food_parts.append(foods)
        component_parts.append(components)
        source_files[source_dir.name] = {
            "food_observation": str(food_path),
            "food_observation_sha256": sha256_file(food_path),
            "component_observation": str(component_path),
            "component_observation_sha256": sha256_file(component_path),
        }

    foods, exact_name_groups = build_exact_name_food_groups(pd.concat(food_parts, ignore_index=True))
    components = pd.concat(component_parts, ignore_index=True)
    component_mapping = build_exact_component_mapping(panel, components)
    mapped_components = component_mapping[component_mapping.mapping_status.eq("mapped_exact")].copy()
    mapped_component_ids = set(mapped_components.component_observation_id)
    target_names = panel.set_index("target_axis_id").canonical_name

    observations: list[pd.DataFrame] = []
    measurement_input_count = 0
    direct_mass_input_count = 0
    for source_dir in sorted(path for path in staging_dir.iterdir() if path.is_dir()):
        parts = sorted(source_dir.glob("measurement_part_*.csv.gz"))
        if not parts:
            raise FileNotFoundError(f"No measurement parts under {source_dir}")
        source_files[source_dir.name]["measurement_parts"] = []
        for part in parts:
            source_files[source_dir.name]["measurement_parts"].append({"path": str(part), "sha256": sha256_file(part)})
            for chunk in pd.read_csv(part, low_memory=False, chunksize=250_000, keep_default_na=False, na_values=[""]):
                measurement_input_count += len(chunk)
                direct = eligible_direct_mass_measurements(chunk)
                direct_mass_input_count += int(direct.sum())
                chunk = chunk.loc[direct & chunk.component_observation_id.isin(mapped_component_ids)].copy()
                if chunk.empty:
                    continue
                chunk = chunk.merge(
                    mapped_components[["component_observation_id", "target_axis_id", "mapping_basis"]],
                    on="component_observation_id",
                    how="inner",
                    validate="many_to_one",
                )
                chunk["target_axis_name"] = chunk.target_axis_id.map(target_names)
                observations.append(chunk)

    mapped_measurements = pd.concat(observations, ignore_index=True) if observations else pd.DataFrame()
    if mapped_measurements.empty:
        raise ValueError("No direct mass measurements mapped to the frozen target panel")
    mapped_measurements = mapped_measurements.merge(
        foods[["food_observation_id", "exact_name_group_id", "exact_food_name_key"]],
        on="food_observation_id",
        how="left",
        validate="many_to_one",
    )
    if mapped_measurements.exact_name_group_id.isna().any():
        raise ValueError("Mapped measurement has no food observation in the audit catalogue")
    if mapped_measurements.measurement_id.duplicated().any():
        raise ValueError("Measurement IDs are duplicated after source staging concatenation")

    cell_summary = summarize_exact_name_axis_cells(mapped_measurements)
    conflicts = cell_summary[cell_summary.cell_status.eq("cross_source_value_disagreement_preserved")].copy()
    source_distribution = _source_distribution(foods, mapped_measurements)
    food_group_distribution = _food_group_distribution(foods, mapped_measurements)
    category_coverage = _category_metadata_coverage(foods)
    unmapped_components = component_mapping[~component_mapping.mapping_status.eq("mapped_exact")].copy()

    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(panel, output_dir / "frozen_prediction_axis_registry.csv")
    write_csv(component_mapping, output_dir / "source_component_to_frozen_target_axis.csv.gz")
    write_csv(unmapped_components, output_dir / "unmapped_or_ambiguous_source_components.csv.gz")
    write_csv(foods, output_dir / "food_observation_to_exact_name_group.csv.gz")
    write_csv(exact_name_groups, output_dir / "exact_name_food_groups.csv")
    write_csv(mapped_measurements, output_dir / "target_measurement_observations.csv.gz")
    write_csv(cell_summary, output_dir / "exact_name_food_axis_cell_summary.csv.gz")
    write_csv(conflicts, output_dir / "cross_source_axis_conflicts.csv.gz")
    write_csv(source_distribution, output_dir / "source_distribution.csv")
    write_csv(food_group_distribution, output_dir / "food_group_distribution_by_source.csv")
    write_csv(category_coverage, output_dir / "food_category_metadata_coverage.csv")

    manifest = {
        "audit_version": FOOD_AUDIT_VERSION,
        "status": "food_audit_only_no_pooling_no_splits_no_model_training",
        "frozen_panel": freeze_manifest,
        "staging_dir": str(staging_dir.resolve()),
        "source_input_files": source_files,
        "food_observations": len(foods),
        "exact_name_food_groups": len(exact_name_groups),
        "exact_name_groups_with_multiple_observations": int(exact_name_groups.food_observation_count.gt(1).sum()),
        "exact_name_groups_with_facet_disagreement": int(exact_name_groups.has_facet_disagreement.sum()),
        "source_component_observations": len(component_mapping),
        "source_component_observations_exact_mapped": len(mapped_components),
        "raw_measurement_records": measurement_input_count,
        "direct_mass_measurement_records_before_axis_mapping": direct_mass_input_count,
        "preserved_target_measurement_records": len(mapped_measurements),
        "observed_frozen_target_axes": int(mapped_measurements.target_axis_id.nunique()),
        "exact_name_food_axis_cells": len(cell_summary),
        "cross_source_axis_conflicts_preserved": len(conflicts),
        "value_conflict_policy": "retain every source measurement; report disagreement; do not select, average, or pool values",
        "food_deduplication_policy": "exact normalized original name only; no fuzzy, semantic, translated, or taxonomy-based merge",
        "axis_mapping_policy": "exact frozen-panel name/registered alias, reviewed source ID, or reviewed INFOODS tag only; no semantic mapping",
    }
    write_json(manifest, output_dir / "audit_manifest.json")
    _write_report(
        report_dir / "FOOD_AUDIT_REPORT_ZH.md",
        freeze_manifest=freeze_manifest,
        source_distribution=source_distribution,
        group_summary=exact_name_groups,
        conflict_summary=cell_summary,
        food_group_distribution=food_group_distribution,
        mapping=component_mapping,
        mapped_measurements=mapped_measurements,
    )
    return manifest
