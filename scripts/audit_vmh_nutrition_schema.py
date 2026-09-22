#!/usr/bin/env python3
"""Audit VMH as a nutrition ontology and provenance reference.

VMH exposes a broad nutrition vocabulary, but its food rows may overlap the
USDA data already used by this repository.  This script makes that distinction
explicit.  It downloads/caches the public VMH nutrient and food catalogues,
checks whether VMH-only BLS nutrient identifiers have food-level observations,
and maps only exact USDA nutrient identifiers to the local mass corpus.

The output is an audit and crosswalk.  It does not modify a training corpus or
turn missing values into zeros.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
VMH_BASE_URL = "https://www.vmh.life/_api"
DEFAULT_CACHE_DIR = ROOT / "data/raw/reference/vmh_2026_08"
DEFAULT_OUTPUT_DIR = ROOT / "data/audits/vmh_nutrition_schema_2026_08"
V4_DATA_DIR = ROOT / "data/processed/multisource_mass_v4"
FOODB_CROSSWALK = ROOT / "data/audits/foodb_component_crosswalk_review.csv"

MASS_UNITS = {"g", "mg", "microg"}
MACRO_CODES = {"203", "204", "205", "291"}
CORE_MICRO_CODES = {
    "301", "302", "303", "304", "305", "306", "307", "309", "310", "312", "313", "314", "315", "316", "317",
    "319", "323", "328", "401", "404", "405", "406", "410", "415", "416", "417", "418", "421", "430",
}
ACTIVITY_EQUIVALENT_CODES = {"318", "320", "324", "435"}
STRUCTURAL_CODES = {"207", "221", "255"}


def fetch_json(url: str, attempts: int = 4) -> dict:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 FoodNutrition research audit"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(request, timeout=60) as response:
                return json.load(response)
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504}:
                raise RuntimeError(f"VMH request failed ({exc.code}): {url}") from exc
            last_error = exc
        except URLError as exc:
            last_error = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"VMH request failed after {attempts} attempts: {url}") from last_error


def cache_or_fetch(cache_path: Path, url: str, refresh: bool) -> dict:
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    payload = fetch_json(url)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def normalize_name(value: object) -> str:
    text = "" if pd.isna(value) else str(value).lower()
    text = text.replace("α", "alpha").replace("β", "beta")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def vmh_code(row: pd.Series) -> str | None:
    value = str(row.get("vmh_nutrient_id", row.get("nut_no", "")))
    match = re.fullmatch(r"USDA(\d+(?:\.\d+)?)", value)
    return match.group(1) if match else None


def role_for(row: pd.Series) -> str:
    code = vmh_code(row)
    unit = str(row.get("vmh_unit", row.get("unit", ""))).strip().lower()
    category = row.get("vmh_category", row.get("category"))
    if code in MACRO_CODES:
        return "core_macro_target"
    if code in CORE_MICRO_CODES:
        return "core_micronutrient_target"
    if code in ACTIVITY_EQUIVALENT_CODES or unit == "iu":
        return "activity_equivalent_exclude_from_mass_task"
    if code in STRUCTURAL_CODES:
        return "structural_component_context_not_target"
    if unit in MASS_UNITS:
        if category == "Lipids":
            return "nutrient_chemical_form_lipid_candidate"
        if category == "Proteins":
            return "nutrient_chemical_form_amino_acid_candidate"
        if category in {"Carbohydrates", "Dietary fibers", "Dietary Fibers"}:
            return "nutrient_chemical_form_carbohydrate_candidate"
        if category == "Vitamins":
            return "vitamin_form_candidate_review_aggregation"
        if category == "Minerals and trace elements":
            return "mineral_candidate_review_definition"
        return "mass_composition_candidate_review"
    if unit in {"kcal", "kj"}:
        return "energy_separate_unit_aware_task"
    return "not_compatible_with_mass_only_task"


def observed_count(url: str) -> int:
    payload = fetch_json(url)
    return int(payload["count"])


def write_markdown(
    output_path: Path,
    raw_catalogue_count: int,
    nutrients: pd.DataFrame,
    foods: pd.DataFrame,
    crosswalk: pd.DataFrame,
    bls_counts: pd.DataFrame,
) -> None:
    category_counts = nutrients.groupby("vmh_category").size().sort_values(ascending=False)
    role_counts = crosswalk.groupby("recommended_role").size().sort_values(ascending=False)
    direct = crosswalk[crosswalk["local_axis_status"].eq("exact_v4_axis")]
    food_sources = foods["source_description"].value_counts()
    bls_total = int(bls_counts["vmh_observed_food_values"].sum())
    accepted_foodb = int(crosswalk["foodb_reviewed_mapping"].eq(True).sum())

    lines = [
        "# VMH Nutrition Schema Audit",
        "",
        "## Decision",
        "",
        "VMH should be used as a **nutrition ontology, crosswalk reference, and future nutrition-to-metabolism interface**. It must not be appended as a new food-level pretraining source in the current corpus. The VMH food catalogue contains "
        f"{len(foods):,} foods, and every food identifier is USDA-prefixed. Its provenance field identifies USDA Standard Reference Release 28. These are therefore overlapping historical USDA records, not independent observations.",
        "",
        "VMH does broaden the target vocabulary: its public nutrient endpoint contains "
        f"{raw_catalogue_count:,} rows. One row has no nutrient identifier and is excluded from the usable schema; the remaining {len(nutrients):,} entries span {len(category_counts)} categories. That vocabulary is useful for cleaning the current nutrient/compound boundary, especially where FooDB records a vitamin or mineral under its Compound table.",
        "",
        "## VMH Catalogue",
        "",
        "| Category | Entries |",
        "|---|---:|",
        *[f"| {category} | {count:,} |" for category, count in category_counts.items()],
        "",
        f"Of these entries, {int(nutrients['is_usda_identifier'].sum()):,} have an explicit `USDA<nutrient-code>` identity and {int((~nutrients['is_usda_identifier']).sum()):,} use a BLS identifier. The public nutrition-data endpoint returns {bls_total:,} food-level values across all {len(bls_counts):,} BLS identifiers in this audit; therefore the BLS catalogue labels are not a source of additional VMH food composition labels.",
        "",
        "## Crosswalk to the Current Corpus",
        "",
        f"The local corpus has exact identifier-level links for {len(direct):,} VMH entries. {accepted_foodb:,} VMH entries have an accepted FooDB-to-USDA mapping, including FooDB compounds that should be reclassified as nutrients. Exact identifier agreement does not by itself make an axis a training target: definition and aggregation still control the decision.",
        "",
        "| Recommended role | VMH entries |",
        "|---|---:|",
        *[f"| {role} | {count:,} |" for role, count in role_counts.items()],
        "",
        "## Training Schema",
        "",
        "Use four explicit layers rather than the old nutrient-versus-compound binary:",
        "",
        "1. **Core nutrition targets:** 4 macronutrients and 29 chemical-mass micronutrients. Use only definition-compatible `g/100g` values. Do not include activity equivalents such as RAE, DFE, TE, or IU.",
        "2. **Nutrient chemical forms:** fatty acids, amino acids, sugars, fibre fractions, carotenoids, tocopherol forms, and vitamin forms. Keep component/aggregate relationships in an explicit graph and prevent simultaneous visibility of a target and a deterministic aggregate/component.",
        "3. **Bioactive chemistry:** FooDB compounds such as polyphenols and flavour volatiles. They are a separate sparse-assay task, not interchangeable with essential nutrients.",
        "4. **Structural or derived components:** water, ash, alcohol, calculated salt, energy, activity equivalents. Do not place these in the same mass regression task without a specific objective.",
        "",
        "## Required v5 Work Before Retraining",
        "",
        "1. Replace the current `target_kind` rule with the reviewed VMH-linked schema in `vmh_to_local_crosswalk.csv`.",
        "2. Reclassify FooDB calcium, iron, iodine, fluoride, molybdenum, retinol, alpha-tocopherol, and vitamin C candidates as nutrient forms only after the existing collision screening; reclassify ash as structural, not compound.",
        "3. Preserve source, analytical/recipe/borrowed/imputed provenance, basis, unit, and censoring code for CoFID and AFCD before admitting those sources. Do not turn CoFID `Tr` or `N` into zero.",
        "4. Use VMH after training for metabolic interpretation and diet-level evaluation, not to duplicate USDA rows in model fitting.",
        "",
        "## Artifacts",
        "",
        "- `vmh_nutrient_catalog.csv`: cached VMH nutrient vocabulary and task role.",
        "- `vmh_to_local_crosswalk.csv`: VMH-to-v4/FooDB identifier-level audit.",
        "- `vmh_food_catalogue_provenance.csv`: evidence that VMH food rows are USDA-derived.",
        "- `vmh_bls_observation_counts.csv`: food-level value check for every BLS nutrient identifier.",
        "",
        "Sources: [VMH API](https://www.vmh.life/_api/docs/); [Noronha et al., 2019, VMH database paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6323901/).",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh", action="store_true", help="Refresh cached VMH catalogues and BLS count queries.")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    nutrients_payload = cache_or_fetch(args.cache_dir / "nutrients.json", f"{VMH_BASE_URL}/nutrients/?page_size=500", args.refresh)
    foods_payload = cache_or_fetch(args.cache_dir / "foods.json", f"{VMH_BASE_URL}/foods/?page_size=10000", args.refresh)
    raw_catalogue_count = int(nutrients_payload["count"])
    nutrients = pd.DataFrame(nutrients_payload["results"])
    foods = pd.DataFrame(foods_payload["results"])
    if len(nutrients) != int(nutrients_payload["count"]):
        raise RuntimeError("VMH nutrient API result is paginated unexpectedly; refusing an incomplete schema audit.")
    if len(foods) != int(foods_payload["count"]):
        raise RuntimeError("VMH food API result is paginated unexpectedly; refusing an incomplete provenance audit.")

    nutrients = nutrients.rename(columns={"nut_no": "vmh_nutrient_id", "common_name": "vmh_common_name", "description": "vmh_description", "unit": "vmh_unit", "category": "vmh_category", "subcategory": "vmh_subcategory"})
    nutrients = nutrients[nutrients["vmh_nutrient_id"].notna()].copy()
    nutrients["vmh_nutrient_id"] = nutrients["vmh_nutrient_id"].astype(str).str.strip()
    nutrients = nutrients[nutrients["vmh_nutrient_id"].ne("")].copy()
    nutrients["fdc_code"] = nutrients.apply(vmh_code, axis=1)
    nutrients["is_usda_identifier"] = nutrients["fdc_code"].notna()
    nutrients["recommended_role"] = nutrients.apply(role_for, axis=1)
    nutrients["mass_unit_compatible"] = nutrients["vmh_unit"].astype(str).str.strip().str.lower().isin(MASS_UNITS)
    nutrients = nutrients[["vmh_nutrient_id", "vmh_common_name", "vmh_description", "vmh_unit", "vmh_category", "vmh_subcategory", "fdc_code", "is_usda_identifier", "mass_unit_compatible", "recommended_role", "sr_order", "mets"]].sort_values("vmh_nutrient_id", kind="stable")

    axes = pd.read_csv(V4_DATA_DIR / "axis_registry.csv")
    values = pd.read_csv(V4_DATA_DIR / "observed_axis_values.csv", usecols=["canonical_food_id", "axis_id"])
    local_foods = pd.read_csv(V4_DATA_DIR / "food_entities.csv", usecols=["canonical_food_id", "primary_source"])
    values = values.merge(local_foods, on="canonical_food_id", how="inner", validate="many_to_one")
    local_coverage = values.groupby(["primary_source", "axis_id"]).size().rename("observed_food_axis_values").reset_index()
    local_coverage = local_coverage.rename(columns={"primary_source": "source"})
    local_wide = local_coverage.pivot_table(index="axis_id", columns="source", values="observed_food_axis_values", aggfunc="sum", fill_value=0).reset_index()
    local_wide.columns.name = None
    crosswalk = nutrients.copy()
    crosswalk["axis_id"] = crosswalk["fdc_code"].map(lambda code: f"fdc:{code}" if pd.notna(code) else pd.NA)
    crosswalk = crosswalk.merge(axes[["axis_id", "axis_name", "target_kind", "axis_class", "mask_policy"]], on="axis_id", how="left")
    crosswalk["local_axis_status"] = crosswalk["axis_name"].notna().map({True: "exact_v4_axis", False: "not_in_v4_by_exact_identifier"})
    crosswalk = crosswalk.merge(local_wide, on="axis_id", how="left")
    for source in ("foodb", "sr_legacy", "cnf"):
        if source not in crosswalk:
            crosswalk[source] = 0
        crosswalk[source] = crosswalk[source].fillna(0).astype(int)

    reviewed = pd.read_csv(FOODB_CROSSWALK)
    reviewed = reviewed[reviewed["review_status"].eq("accepted")].copy()
    reviewed["axis_id"] = "fdc:" + reviewed["canonical_usda_code"].astype(str)
    reviewed = reviewed.groupby("axis_id", as_index=False).agg(
        foodb_reviewed_mapping=("axis_id", "size"),
        foodb_mapped_names=("foodb_axis_name", lambda names: "; ".join(sorted(set(names)))),
        foodb_mapping_relation=("mapping_relation", lambda values: "; ".join(sorted(set(values)))),
    )
    crosswalk = crosswalk.merge(reviewed, on="axis_id", how="left")
    crosswalk["foodb_reviewed_mapping"] = crosswalk["foodb_reviewed_mapping"].notna()
    crosswalk["foodb_mapped_names"] = crosswalk["foodb_mapped_names"].fillna("")
    crosswalk["foodb_mapping_relation"] = crosswalk["foodb_mapping_relation"].fillna("")
    crosswalk = crosswalk.sort_values(["recommended_role", "vmh_category", "vmh_nutrient_id"], kind="stable")

    bls = nutrients[~nutrients["is_usda_identifier"]][["vmh_nutrient_id", "vmh_common_name", "vmh_unit", "vmh_category"]].copy()
    count_cache_path = args.cache_dir / "bls_nutritiondata_counts.json"
    counts: dict[str, int] = {} if args.refresh or not count_cache_path.exists() else json.loads(count_cache_path.read_text())
    if len(counts) < len(bls):
        missing_identifiers = [identifier for identifier in bls["vmh_nutrient_id"] if identifier not in counts]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(observed_count, f"{VMH_BASE_URL}/nutritiondata/?nutrient={identifier}&page_size=1"): identifier
                for identifier in missing_identifiers
            }
            for index, future in enumerate(as_completed(futures), start=1):
                identifier = futures[future]
                counts[identifier] = future.result()
                count_cache_path.write_text(json.dumps(counts, indent=2, sort_keys=True))
                if index % 25 == 0:
                    print(f"Checked VMH BLS food-level coverage: {index}/{len(futures)} newly checked; {len(counts)}/{len(bls)} cached")
    bls["vmh_observed_food_values"] = bls["vmh_nutrient_id"].map(counts).astype(int)

    foods_out = foods.rename(columns={"food_id": "vmh_food_id", "name": "food_name", "sources": "source_description"})
    foods_out["food_id_prefix"] = foods_out["vmh_food_id"].str.extract(r"^([A-Za-z]+)", expand=False)
    foods_out = foods_out[["vmh_food_id", "food_id_prefix", "food_name", "product_type", "source_description", "survey"]]

    nutrients.to_csv(args.output_dir / "vmh_nutrient_catalog.csv", index=False)
    crosswalk.to_csv(args.output_dir / "vmh_to_local_crosswalk.csv", index=False)
    bls.to_csv(args.output_dir / "vmh_bls_observation_counts.csv", index=False)
    foods_out.to_csv(args.output_dir / "vmh_food_catalogue_provenance.csv", index=False)
    write_markdown(args.output_dir / "VMH_NUTRITION_SCHEMA_AUDIT.md", raw_catalogue_count, nutrients, foods_out, crosswalk, bls)

    print(f"VMH nutrients: {len(nutrients):,}")
    print(f"VMH foods: {len(foods_out):,}; ID prefixes: {dict(Counter(foods_out['food_id_prefix']))}")
    print(f"VMH BLS food-level observations: {int(bls['vmh_observed_food_values'].sum()):,}")
    print(f"Wrote audit: {args.output_dir}")


if __name__ == "__main__":
    main()
