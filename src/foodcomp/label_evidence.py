"""Resolve bibliographic evidence before treating a compiled value as a label."""

from __future__ import annotations

from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pandas as pd


def reference_catalogs(root: Path) -> dict[str, dict[str, dict]]:
    frida = pd.read_excel(root / "data/raw/candidates/frida_6_1/FCDB_6.1_Dataset.xlsx",
                          sheet_name="Source", keep_default_na=False)
    frida_rows = {}
    for row in frida.itertuples(index=False):
        frida_rows[str(row.SourceID)] = {
            "title": row.TitleEnglish if row.TitleEnglish not in {"", "NULL"} else row.TitleOriginal,
            "reference_type": row.EurofirRefType, "url": "" if row.URL == "NULL" else row.URL,
        }
    ciqual_rows = {}
    for element in ET.parse(root / "data/raw/candidates/ciqual_2025/sources.xml").getroot():
        ciqual_rows[element.findtext("source_code", "").strip()] = {
            "title": element.findtext("ref_citation", "").strip(), "reference_type": "official_source_citation", "url": "",
        }
    return {"frida": frida_rows, "ciqual": ciqual_rows}


def reference_ids(raw) -> list[str]:
    parts = [x.strip() for x in str(raw).split(",") if x.strip()]
    return [str(int(float(part))) if re.fullmatch(r"\d+(?:\.0+)?", part) else part for part in parts]


def inspect_reference(source: str, raw, catalogs: dict) -> dict:
    ids = reference_ids(raw)
    entries = [catalogs.get(source, {}).get(identifier) for identifier in ids]
    unresolved = not ids or any(entry is None or not entry["title"] for entry in entries)
    titles = [entry["title"] for entry in entries if entry]
    types = [entry["reference_type"] for entry in entries if entry]
    combined = " | ".join(titles)
    lower = combined.casefold()
    issues = []
    if unresolved:
        issues.append("unresolved_reference")
    if source == "frida":
        if "1655" in ids:
            issues.append("assumed_zero_not_analyzed")
        if "E" in types or set(ids) & {"2007", "2054", "2133", "2296"}:
            issues.append("estimated_calculated_or_imputed")
        if set(ids) & {"1342", "1344", "1353", "1355", "1357", "1358", "1541", "1938", "1960",
                       "2132", "2135", "2136", "2138", "2140", "2141", "2143", "2187", "2190", "2284", "2286", "2288"}:
            issues.append("copied_database_lineage_unresolved")
    elif source == "ciqual":
        if re.search(r"calcul[éees]|imput[ée]|ajust[ée]", lower):
            issues.append("estimated_calculated_or_imputed")
        if "usda" in lower or "u.s. department of agriculture" in lower or "table de composition" in lower:
            issues.append("copied_database_lineage_unresolved")
        if ("oqali" in lower and "analys" not in lower) or "industrielles" in lower:
            issues.append("label_or_mixed_survey_requires_review")
    return {"reference_ids": ";".join(ids), "resolved_reference_titles": combined,
            "reference_types": ";".join(types), "evidence_issues": ";".join(issues),
            "strict_evidence_rejected": bool(issues),
            "interpretation": "Bibliographic check, not independent laboratory or expert certification"}


def gate_reference_evidence(frame: pd.DataFrame, catalogs: dict) -> pd.DataFrame:
    """Fail closed on explicit contrary evidence; preserve every raw value."""
    result = frame.copy()
    for source in ("frida", "ciqual"):
        mask = result.source_key.eq(source)
        if not mask.any():
            continue
        lookup = {raw: inspect_reference(source, raw, catalogs) for raw in result.loc[mask, "source_reference"].unique()}
        checks = result.loc[mask, "source_reference"].map(lookup)
        result.loc[mask, "quality_evidence"] = checks.map(lambda item: item["resolved_reference_titles"] + " | " + item["evidence_issues"])
        rejected = checks[checks.map(lambda item: item["strict_evidence_rejected"])].index
        result.loc[rejected, ["main_value_eligible", "strict_validation_eligible", "independent_evidence"]] = False
        result.loc[rejected, "quality_tier"] = "D"
        result.loc[rejected, "data_layer"] = "reference_evidence_hold"
        result.loc[rejected, "exclusion_reason"] = checks.loc[rejected].map(lambda item: item["evidence_issues"])
        result.loc[rejected, "value_origin"] = "bibliography_contradicts_independent_assay_or_remains_unresolved"
        assumed = checks[checks.map(lambda item: "assumed_zero_not_analyzed" in item["evidence_issues"])].index
        result.loc[assumed, "value_status"] = "assumed_zero"
        result.loc[assumed, "is_explicit_zero"] = False
        result.loc[assumed, "zero_semantics"] = "not_analyzed_natural_zero_assumed_by_compiler"
    sr = result.source_key.eq("usda_sr_legacy")
    nonanalytical = sr & result.method_expression.fillna("").str.contains(r"Recipe|Based on physical composition", case=False, regex=True)
    if nonanalytical.any():
        result.loc[nonanalytical, ["main_value_eligible", "strict_validation_eligible", "independent_evidence"]] = False
        result.loc[nonanalytical, "data_layer"] = "calculated_reference_auxiliary"
        result.loc[nonanalytical, "quality_tier"] = "D"
        result.loc[nonanalytical, "exclusion_reason"] = "SR_derivation_is_recipe_or_physical_composition_not_direct_analysis"
    return result


def foundation_result_links(base: Path) -> pd.DataFrame:
    """Join actual lab results through aggregate -> sample -> subsample lineage."""
    def read(name, **kwargs):
        return pd.read_csv(base / name, keep_default_na=False, low_memory=False, **kwargs)

    inputs = read("input_food.csv", usecols=["fdc_id", "fdc_of_input_food"]).rename(columns={"fdc_id": "parent_fdc_id", "fdc_of_input_food": "sample_fdc_id"})
    children = read("sub_sample_food.csv").rename(columns={"fdc_id": "subsample_fdc_id", "fdc_id_of_sample_food": "sample_fdc_id"})
    values = read("food_nutrient.csv", usecols=["id", "fdc_id", "nutrient_id"]).rename(columns={"id": "food_nutrient_id", "fdc_id": "subsample_fdc_id"})
    results = read("sub_sample_result.csv")
    methods = read("lab_method.csv").rename(columns={"id": "lab_method_id"})
    linked = inputs.merge(children, on="sample_fdc_id").merge(values, on="subsample_fdc_id").merge(results, on="food_nutrient_id", validate="many_to_one")
    linked = linked.merge(methods[["lab_method_id", "description", "technique"]], on="lab_method_id", how="left", validate="many_to_one")
    linked["method_label"] = linked.description.fillna("") + " | " + linked.technique.fillna("")
    return linked.groupby(["parent_fdc_id", "nutrient_id"]).agg(
        actual_linked_result_count=("food_nutrient_id", "nunique"),
        actual_linked_sample_count=("sample_fdc_id", "nunique"),
        actual_linked_method_labels=("method_label", lambda values: "; ".join(sorted(set(values)))),
    ).reset_index()
