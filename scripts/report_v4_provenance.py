"""Read-only provenance report for the v4 candidate; run from the repo root.

Dependencies: pip install numpy pandas openpyxl
This script never builds, reclassifies, or changes the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from foodcomp.constants import AMINO_ACID_NAMES, AMINO_ACID_TAGS, MINERAL_TAGS, NUTRITIONAL_ROLE_BY_TAG, VITAMIN_TAG_PREFIXES
from foodcomp.harmonize import _role_from_component
from foodcomp.util import normalize_text


SOURCES = {
    "usda_sr_legacy": ("USDA SR Legacy", "April 2018", "usda"),
    "usda_foundation": ("USDA Foundation", "April 2026", "usda"),
    "cnf": ("CNF", "2026", "cnf"),
    "foodb": ("FooDB", "April 2020 CSV", "foodb"),
    "frida": ("Frida", "6.1", "frida"),
    "ciqual": ("CIQUAL", "2025", "ciqual"),
    "afcd": ("AFCD", "Release 3", "afcd"),
    "cofid": ("CoFID", "2021", "cofid"),
    "norway": ("Norway", "6 Sep 2026 snapshot", "norway"),
}

ROLES = {
    "macronutrient": "Macronutrient-associated expressions",
    "micronutrient_mineral": "Mineral / trace-element expressions",
    "micronutrient_vitamin": "Vitamin-associated expressions and forms",
    "essential_nutrient_choline": "Choline",
    "nutrient_constituent_fatty_acid": "Fatty acids and fatty-acid sums",
    "nutrient_constituent_amino_acid": "Amino-acid-associated expressions",
    "nutrient_constituent_carbohydrate": "Sugars, starch and related constituents",
    "other_nutritional_component": "Other proximate / nutritional expressions",
    "other_food_component": "Other food components / unresolved roles",
}

RULES = {
    "R1": "Choline tag CHOLN or an exact choline name. Essential-nutrient interpretation informed by NIH; a separate reporting group is a project choice.",
    "R2": "Exact canonical proximate name (protein, total protein, total fat, carbohydrate or dietary fibre/fiber variants). The project supplies the macronutrient label.",
    "R3": "Explicit tag-to-role dictionary: protein, fat, carbohydrate and fibre tags; WATER, ASH and ALC are placed in Other nutritional components. This dictionary is project-maintained.",
    "R4": "Mineral-tag list or a mineral-name keyword in the canonical name or any source-group alias. This does not independently establish essentiality for every element or chemical form.",
    "R5": "Vitamin-related tag prefix, vitamin name, or vitamin keyword in the name or any source group. Broad rule; forms and non-essential co-reported chemicals require review.",
    "R6": "Fatty-acid tag prefix or the words fatty acid in the canonical name or source groups. Chemical-constituent grouping, not a claim that each fatty acid is essential.",
    "R7": "Amino-acid tag/name or the words amino acid in the name or source groups. Group-driven assignments can include non-amino-acid items such as total nitrogen.",
    "R8": "Sugar, glucose, fructose, sucrose, starch or oligosaccharide keyword in the canonical name or source groups. A project carbohydrate-constituent grouping.",
    "R9": "Fallback Other food component; chemical identity only and no nutritional-role assertion. The absence of a role is not proof of non-nutritional status.",
}

FOOD_FIELDS = {
    "usda_sr_legacy": "food_category.csv description joined to food.csv; food_type is the source data_type (sr_legacy_food). These are native FDC food groups, not model-derived classes.",
    "usda_foundation": "The same FDC food_category join, restricted by foundation_food.csv. food_type denotes Foundation Foods; the data type is not a biological food family.",
    "cnf": "Food_Name joined to CNF_Food_Group on CNF_Food_Group_Code; CNF_Food_Group_Description_EN supplies the group. French names and scientific names remain source metadata.",
    "foodb": "Food.csv food_group, food_subgroup and food_type are inherited by content-level observations. Native labels such as Type 1 are retained; they are not reinterpreted as processing classes.",
    "frida": "Food sheet FoodGroup supplies the native group; EurofirFoodGroup is stored as food_type. Source FoodOntology, FoodEx2 and LanguaL fields are retained when present.",
    "ciqual": "alim_grp_nom_eng, alim_ssgrp_nom_eng and alim_ssssgrp_nom_eng map to group, subgroup and type. These represent three native hierarchy levels.",
    "afcd": "Classification column of the nutrient profile supplies the numeric food_group code. The separate official Food Group Information workbook provides broader two-digit parents; no fuzzy recoding is used.",
    "cofid": "GROUP column supplies the original one-, two- or three-letter food code. Meanings are defined in the User Guide Appendix B. NAME and DESC are preserved separately.",
    "norway": "foodGroupId from foods.json is resolved against food-groups.json. The API supplies a native hierarchy and per-food LanguaL metadata.",
}


def tex(value) -> str:
    value = re.sub(r"\s+", " ", str(value)).strip()
    escapes = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(escapes.get(c, c) for c in value)


def identifier(value: str) -> str:
    if any(c in value for c in "{}\\"):
        return tex(value)
    return r"\path{" + value + "}"


def table(headers, rows, widths=None, caption="", label="", compact=False) -> str:
    columns = "".join(f"P{{{w}\\textwidth}}" for w in widths) if widths else "l" * len(headers)
    if compact:
        out = [r"\par\vspace{3pt}\begingroup\footnotesize", r"\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.12}", r"\begin{tabular}{@{}" + columns + "@{}}", r"\toprule", " & ".join(r"\textbf{" + tex(x) + "}" for x in headers) + r" \\", r"\midrule"]
        out.extend(" & ".join(row) + r" \\" for row in rows)
        out.extend([r"\bottomrule", r"\end{tabular}\endgroup\par"])
        return "\n".join(out)
    out = [r"\begingroup" + (r"\footnotesize" if compact else r"\small"), r"\setlength{\tabcolsep}{4pt}", r"\setlength{\LTpre}{4pt}\setlength{\LTpost}{5pt}", r"\renewcommand{\arraystretch}{1.12}", r"\begin{longtable}{@{}" + columns + "@{}}"]
    if caption:
        out.append(r"\caption{" + caption + "}" + (r"\label{" + label + "}" if label else "") + r"\\")
    header = " & ".join(r"\textbf{" + tex(x) + "}" for x in headers) + r" \\"
    out += [r"\toprule", header, r"\midrule", r"\endfirsthead", r"\toprule", header, r"\midrule", r"\endhead", r"\midrule", r"\multicolumn{" + str(len(headers)) + r"}{r}{\footnotesize Continued on next page}\\", r"\endfoot", r"\bottomrule", r"\endlastfoot"]
    for row in rows:
        out.append(" & ".join(row) + r" \\")
    out += [r"\end{longtable}", r"\endgroup"]
    return "\n".join(out)


def rule_for(row, source_groups: str) -> tuple[str, str]:
    tag, name = str(row.infoods_tag).upper(), row.canonical_name
    plain = normalize_text(name)
    normalized = normalize_text(f"{name} {source_groups}")
    predicted = _role_from_component(tag, name, source_groups)
    assert predicted[0] == row.nutritional_role, f"Stored role cannot be reproduced: {row.component_concept_id}"
    assert predicted[3] == row.nutritional_role_review_status
    if tag == "CHOLN" or plain in {"choline", "choline total"}:
        return "R1", f"Choline identity: {tag or name}."
    if plain in {"protein", "total protein", "fat total", "total fat", "carbohydrate", "dietary fibre", "dietary fiber"}:
        return "R2", f"Exact proximate name: {name}."
    if tag in NUTRITIONAL_ROLE_BY_TAG:
        return "R3", f"Explicit project tag dictionary: {tag}."
    minerals = ("calcium", "iron", "magnesium", "phosphorus", "potassium", "sodium", "zinc", "copper", "manganese", "selenium", "iodine", "molybdenum", "chloride")
    if tag in MINERAL_TAGS or any(x in normalized for x in minerals):
        token = tag if tag in MINERAL_TAGS else next(x for x in minerals if x in normalized)
        return "R4", f"Mineral tag/keyword: {token}."
    vitamins = ("vitamin", "thiamin", "riboflavin", "niacin", "folate", "biotin", "pantothenic", "cobalamin", "tocopherol")
    if tag.startswith(VITAMIN_TAG_PREFIXES) or any(x in normalized for x in vitamins):
        token = next((x for x in VITAMIN_TAG_PREFIXES if tag.startswith(x)), "") or next(x for x in vitamins if x in normalized)
        in_group_only = not tag.startswith(VITAMIN_TAG_PREFIXES) and not any(x in plain for x in vitamins)
        return "R5", f"Vitamin prefix/keyword: {token}." + (" Trigger is in a source group, not the component name." if in_group_only else "")
    if tag.startswith(("FA", "FASAT", "FAMS", "FAPU")) or "fatty acid" in normalized:
        return "R6", f"Fatty-acid prefix/group: {tag or 'fatty acid'}."
    if tag in AMINO_ACID_TAGS or tag.startswith("AA") or plain in AMINO_ACID_NAMES or "amino acid" in normalized:
        group_only = not (tag in AMINO_ACID_TAGS or tag.startswith("AA") or plain in AMINO_ACID_NAMES or "amino acid" in plain)
        return "R7", ("Source-group amino acid keyword; component name itself did not match an amino-acid identity." if group_only else f"Amino-acid tag/name rule: {tag or name}.")
    carbs = ("sugar", "glucose", "fructose", "sucrose", "starch", "oligosaccharide")
    if any(x in normalized for x in carbs):
        return "R8", "Carbohydrate keyword: " + next(x for x in carbs if x in normalized) + "."
    return "R9", "No nutritional-role rule matched."


def render_report(root: Path, output: Path) -> Path:
    summary = json.loads((output / "audit_summary.json").read_text())
    assert (summary["foods"], summary["axes"], summary["cells"], summary["foods_without_selected_cells"]) == (14228, 499, 641427, 793), "The narrative is pinned to the audited v4 snapshot; review it before using another build"
    axes = read(output / "axis_registry_snapshot.csv")
    foods = read(output / "food_provenance.csv")
    source_counts = read(output / "source_contribution_counts.csv").set_index("source_key")
    source_axes = read(output / "axis_source_provenance.csv")
    aliases = read(output / "axis_aliases_including_noncontributors.csv")
    source_axis_groups = read(output / "source_axis_categories.csv")
    known = set(read(output / "known_component_identity_review.csv").component_concept_id)
    placeholders = {}
    placeholders["SOURCE_COUNTS"] = table(
        ["Source / version", "Food concepts", "Train", "Val.", "Axes", "Supported cells"],
        [[tex(SOURCES[s][0]) + r"\newline\footnotesize " + tex(SOURCES[s][1]) + r" \cite{" + SOURCES[s][2] + "}", *[f"{int(source_counts.loc[s, k]):,}" for k in ["foods", "train_foods", "validation_foods", "axes", "supported_cells"]]] for s in SOURCES],
        [0.25, 0.12, 0.10, 0.08, 0.07, 0.16],
        "Actual selected numerical contributions. Source rows overlap and are not additive.", "tab:foodsource",
    )
    placeholders["FOOD_FIELDS"] = table(
        ["Source", "Native category fields and their interpretation"],
        [[tex(SOURCES[s][0]) + r" \cite{" + SOURCES[s][2] + "}", tex(FOOD_FIELDS[s])] for s in SOURCES], [0.19, 0.75],
        "Source food categories as imported in v4.", "tab:foodfields",
    )
    example_rows = []
    example_ids = {"usda_sr_legacy": "169831", "usda_foundation": "2768188", "cnf": "5282", "foodb": "DUKE|474|Leaf|raw", "frida": "1957", "ciqual": "12112", "afcd": "F000002", "cofid": "14-331", "norway": "4.341"}
    for s, source_id in example_ids.items():
        candidate = foods[foods.source_key.eq(s) & foods.source_food_id.astype(str).eq(source_id)]
        assert len(candidate) >= 1
        x = candidate.iloc[0]
        native = " / ".join(str(x[k]) for k in ["food_group", "food_subgroup", "food_type"] if x[k] != "")
        example_rows.append([tex(SOURCES[s][0]) + r"\newline " + identifier(str(x.source_food_id)), tex(x.original_name), tex(native)])
    placeholders["FOOD_EXAMPLES"] = table(["Source / ID", "Original food name", "Native group / subgroup / type"], example_rows, [0.22, 0.36, 0.34])
    axis_source_rows = []
    for s in SOURCES:
        for x in source_axis_groups[source_axis_groups.source_key.eq(s)].itertuples(index=False):
            kind = "native table" if s == "foodb" else "native worksheet" if s == "cofid" else "native parameter group" if s == "frida" else "adapter umbrella"
            axis_source_rows.append([tex(SOURCES[s][0]), tex(x.source_component_group), tex(kind), f"{x.axes:,}"])
    placeholders["AXIS_SOURCE_CATEGORIES"] = table(["Source", "Stored source category", "Category provenance", "Axes"], axis_source_rows, [0.16, 0.41, 0.25, 0.08], "Native component groups versus adapter umbrella labels. Counts overlap across sources.")
    role_rows = []
    for role, label in ROLES.items():
        a = axes[axes.nutritional_role.eq(role)]
        role_rows.append([tex(label), str(len(a)), str(a.training_role.eq("maskable_target").sum()), str(a.training_role.eq("context_only").sum())])
    role_rows.append([r"\textbf{Total}", r"\textbf{499}", r"\textbf{158}", r"\textbf{341}"])
    placeholders["ROLE_COUNTS"] = table(["Current project reporting role", "Axes", "Target", "Context"], role_rows, [0.59, 0.10, 0.10, 0.10], "Provisional classification of retained expressions, not counts of essential nutrients.")
    placeholders["RULE_TABLE"] = table(["Rule", "Actual assignment rule, in precedence order"], [[tex(k), tex(v)] for k, v in RULES.items()], [0.08, 0.85])
    meaning = {"INFOODS": "Identity anchored to an accepted tag; role remains a separate rule.", "SOURCE": "Kept source-specific; no verified common cross-database identity asserted.", "CHEBI": "ChEBI identity anchor, not automatic human essentiality.", "USDA_CNF_NUTRIENT_NBR": "Shared number-based identity fallback; decimal-code review is required.", "EUROFIR_EFSA": "Source-supplied component expression/identifier.", "INCHIKEY": "Chemical structure key, not a nutritional-role classification."}
    placeholders["IDENTITY_TABLE"] = table(["Primary identity namespace", "Axes", "What the field establishes"], [[identifier(k), str(v), tex(meaning[k])] for k, v in summary["authority_namespace_counts"].items()], [0.32, 0.08, 0.51])
    afcd_book = root / "data/raw/expansion_2026_09_06/afcd/04_AFCD Release 3 - Food group information.xlsx"
    afcd_groups = pd.read_excel(afcd_book, sheet_name="Food group information", header=3, dtype=str).fillna("")
    afcd_parents = dict(zip(afcd_groups["Food group ID"], afcd_groups["Food group name"]))
    native_groups = foods.groupby(["source_key", "food_group"]).food_concept_id.nunique()
    group_tables = []
    for s in SOURCES:
        group_tables.append(r"\subsection{" + tex(SOURCES[s][0]) + "}")
        rows = []
        for name, count in native_groups[s].sort_index().items():
            label = str(name)
            parent = afcd_parents.get(label[:2], "") if s == "afcd" else ""
            if parent:
                label += " (official parent: " + parent + ")"
            rows.append([tex(label or "[not supplied]"), str(count)])
        group_tables.append(table(["Original food-group label / code", "Contributing foods"], rows, [0.75, 0.19]))
    placeholders["FOOD_GROUPS"] = "\n".join(group_tables)
    status_short = {"authority_supported": "rule marked authority-supported (not expert sign-off)", "name_rule_pending_expert_review": "name/group rule; expert review pending", "source_definition_role_pending_expert_review": "source-definition role; expert review pending", "no_nutritional_role_asserted": "no nutritional role asserted"}
    catalogue, classification_rows = [], []
    count = 0
    for role, label in ROLES.items():
        catalogue.append(r"\subsection{" + tex(label) + "}")
        for row in axes[axes.nutritional_role.eq(role)].sort_values(["canonical_name", "component_concept_id"], key=lambda x: x.str.casefold()).itertuples(index=False):
            count += 1
            code = f"A{count:03d}"
            registered = aliases[aliases.component_concept_id.eq(row.component_concept_id)]
            current = source_axes[source_axes.component_concept_id.eq(row.component_concept_id)]
            rule, explanation = rule_for(row, unique_text(registered.source_component_group))
            warnings = []
            if row.component_concept_id in known:
                warnings.append("KNOWN IDENTITY ALERT: distinct decimal nutrient codes are mapped to this axis; total/form-specific equivalence is not certified.")
            if "nitrogen" in normalize_text(row.canonical_name) and role == "nutrient_constituent_amino_acid":
                warnings.append("CLASSIFICATION ALERT: total nitrogen is not an amino acid; the source-group keyword caused this assignment.")
            if "lycopene" in normalize_text(row.canonical_name) and role == "micronutrient_vitamin":
                warnings.append("CLASSIFICATION ALERT: source vitamin-group membership does not make lycopene an essential vitamin.")
            if any(x in normalize_text(row.canonical_name) for x in ["cholesterol"]) and row.component_family == "carbohydrate_family":
                warnings.append("MASK-FAMILY ALERT: the current CHO prefix rule assigns cholesterol to the carbohydrate family; this is not a chemical-class conclusion.")
            if normalize_text(row.canonical_name) in {"lead", "cadmium", "mercury"}:
                warnings.append("SCOPE ALERT: contaminant/safety measurement, not a nutritional requirement.")
            classification_rows.append(dict(axis_number=code, component_concept_id=row.component_concept_id, rule=rule, rule_explanation=explanation, actual_source_keys=unique_text(current.source_key), category_rule_source_groups=unique_text(registered.source_component_group), nutritional_role=row.nutritional_role, training_role=row.training_role, review_status=row.nutritional_role_review_status, warnings=" ".join(warnings)))
            catalogue.append(r"\par\addvspace{10pt}\noindent\begin{minipage}{\textwidth}\raggedright")
            catalogue.append(r"{\normalsize\bfseries " + code + ". " + tex(row.canonical_name) + r"}\par\vspace{4pt}")
            use = "Target" if row.training_role == "maskable_target" else "Context"
            catalogue.append(identifier(row.component_concept_id) + r"\hfill\textbf{" + use + "}; train/val: " + f"{row.train_count:,}/{row.validation_count:,}.\n")
            catalogue.append(r"\textbf{Identity:} " + identifier(row.authority_namespace + ":" + str(row.authority_id)) + "; expression: " + tex(row.expression_variant.replace("_", " ")) + ". Name basis: " + tex(row.canonical_name_basis) + ".\n")
            catalogue.append(r"\textbf{Our category:} " + tex(label) + ". " + rule + ": " + tex(explanation) + " Review: " + tex(status_short[row.nutritional_role_review_status]) + ".\n")
            chemical = "Unresolved in the current chemical-class field."
            if row.chemical_class != "unresolved_chemical_class":
                chemical = "Ontology/source class field present; " + ("ChEBI " + row.chebi_id + "." if row.chebi_id else "source chemical-class label.")
            if row.cdno_id:
                chemical += " CDNO exact-label link: " + row.cdno_id + "."
            catalogue.append(r"\textbf{Chemical evidence:} " + tex(chemical) + "\n")
            actual_sources = set(current.source_key)
            extra = sorted(set(registered.source_key) - actual_sources)
            if extra:
                catalogue.append(r"\textbf{Registry only, no selected value:} " + tex(", ".join(SOURCES.get(x, (x,))[0] for x in extra)) + ".\n")
            if warnings:
                catalogue.append(r"\textbf{Review warning:} " + tex(" ".join(warnings)) + "\n")
            source_rows = []
            for x in current.sort_values(["source_key", "original_name", "source_component_id"]).itertuples(index=False):
                source_rows.append([tex(SOURCES[x.source_key][0]), tex(x.original_name) + r"\newline{\footnotesize ID: " + identifier(str(x.source_component_id)) + "}", tex(x.source_component_group), str(x.contributing_foods)])
            catalogue.append(table(["Supplier", "Original component name / ID", "Original group or adapter label", "Foods"], source_rows, [0.15, 0.39, 0.30, 0.06], compact=True))
            catalogue.append(r"\end{minipage}\par")
    assert count == len(axes) == 499
    pd.DataFrame(classification_rows).to_csv(output / "axis_classification_rule_ledger.csv", index=False)
    placeholders["AXIS_CATALOGUE"] = "\n".join(catalogue)
    text = (root / "reports/scientific_food_composition_v4/provenance_report_template.tex").read_text()
    for name, content in placeholders.items():
        marker = f"@@{name}@@"
        assert text.count(marker) == 1, marker
        text = text.replace(marker, content)
    assert not re.search(r"@@[A-Z_]+@@", text)
    target = output / "dataset_provenance_classification_v4.tex"
    target.write_text(text, encoding="utf-8")
    print(f"Wrote {target}: {count} axis entries, {len(native_groups)} source food groups")
    return target

def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False, low_memory=False)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def unique_text(values) -> str:
    return "; ".join(sorted({str(x).strip() for x in values if str(x).strip()}))


def analyze(root: Path, output: Path) -> dict:
    release = root / "data/processed/scientific_food_composition_v4/release"
    inputs = sorted(release.iterdir())
    before = {str(p.relative_to(root)): sha256(p) for p in inputs if p.is_file()}
    matrix = np.load(release / "canonical_profile_matrix.npz", allow_pickle=False)
    ids, axes = matrix["food_ids"].astype(str), matrix["component_ids"].astype(str)
    profiles = read(release / "canonical_profile.csv.gz")
    profiles = profiles[profiles.food_concept_id.isin(ids) & profiles.component_concept_id.isin(axes) & profiles.aggregation_status.eq("accepted")].copy()
    assert not profiles.duplicated(["food_concept_id", "component_concept_id"]).any()
    assert len(profiles) == int(matrix["observed"].sum())
    row = pd.Index(ids).get_indexer(profiles.food_concept_id)
    col = pd.Index(axes).get_indexer(profiles.component_concept_id)
    assert matrix["observed"][row, col].all()
    np.testing.assert_allclose(matrix["values"][row, col], profiles.canonical_value_g_per_100g.astype(float), rtol=2e-6, atol=1e-12)
    links = profiles[["food_concept_id", "component_concept_id", "partition", "measurement_ids"]].copy()
    links["measurement_id"] = links.measurement_ids.str.split(";")
    links = links.drop(columns="measurement_ids").explode("measurement_id")
    measurements = read(release / "measurement_main_eligible.csv.gz")
    assert measurements.measurement_id.is_unique
    values = links.merge(measurements, on="measurement_id", validate="many_to_one", indicator=True)
    assert values["_merge"].eq("both").all()
    values = values.drop(columns="_merge")
    foods = read(release / "food_observation.csv.gz")
    components = read(release / "component_observation.csv.gz")
    food_map = read(release / "food_observation_to_concept.csv.gz")
    axis_map = read(release / "component_observation_to_concept.csv.gz")
    for field, mapping, concept in [("food_observation_id", food_map, "food_concept_id"), ("component_observation_id", axis_map, "component_concept_id")]:
        assert mapping[field].is_unique
        assert values[field].map(mapping.set_index(field)[concept]).eq(values[concept]).all()
    for field, observations in [("food_observation_id", foods), ("component_observation_id", components)]:
        assert values[field].map(observations.set_index(field).source_key).eq(values.source_key).all()
    food_links = values[["food_concept_id", "food_observation_id", "source_key", "partition"]].drop_duplicates()
    food_links = food_links.merge(foods.drop(columns="source_key"), on="food_observation_id", validate="many_to_one")
    component_links = values[["component_concept_id", "component_observation_id", "source_key"]].drop_duplicates()
    component_links = component_links.merge(components.drop(columns="source_key"), on="component_observation_id", validate="many_to_one")
    counts = values.groupby(["component_concept_id", "component_observation_id"]).agg(selected_records=("measurement_id", "nunique"), contributing_foods=("food_concept_id", "nunique")).reset_index()
    component_links = component_links.merge(counts, validate="one_to_one")
    axis_details = read(release / "component_concept.csv.gz").set_index("component_concept_id").loc[axes].reset_index()
    partitions = read(release / "ml_partition.csv").set_index("food_concept_id").loc[ids].reset_index()
    food_details = read(release / "food_concept.csv.gz").set_index("food_concept_id").loc[ids].reset_index()
    source_rows = []
    for source, group in values.groupby("source_key"):
        source_rows.append(dict(source_key=source, foods=group.food_concept_id.nunique(), train_foods=group.loc[group.partition.eq("train"), "food_concept_id"].nunique(), validation_foods=group.loc[group.partition.eq("validation"), "food_concept_id"].nunique(), source_food_observations=group.food_observation_id.nunique(), axes=group.component_concept_id.nunique(), selected_records=group.measurement_id.nunique(), supported_cells=len(group[["food_concept_id", "component_concept_id"]].drop_duplicates())))
    source_counts = pd.DataFrame(source_rows)
    groups = food_links.groupby(["source_key", "food_group", "food_subgroup", "food_type"], dropna=False).agg(foods=("food_concept_id", "nunique"), source_observations=("food_observation_id", "nunique")).reset_index()
    axis_groups = component_links.groupby(["source_key", "source_component_group"], dropna=False).agg(axes=("component_concept_id", "nunique"), source_component_records=("component_observation_id", "nunique")).reset_index()
    all_aliases = axis_map[axis_map.component_concept_id.isin(axes)].merge(components, on="component_observation_id", validate="many_to_one")
    all_aliases["has_selected_value"] = all_aliases.component_observation_id.isin(component_links.component_observation_id)
    hierarchy = json.loads((release / "food_hierarchy.json").read_text())
    known = pd.read_csv(root / "reports/scientific_food_composition_v4/known_component_identity_review.csv", keep_default_na=False)
    summary = dict(
        version="scientific_food_composition_v4", foods=len(ids), axes=len(axes), cells=len(profiles), missing=int(matrix["observed"].size - matrix["observed"].sum()),
        source_count=len(source_counts), partitions=partitions.partition.value_counts().to_dict(),
        food_sources_per_concept=food_links.groupby("food_concept_id").source_key.nunique().value_counts().sort_index().to_dict(),
        source_categories_per_concept=food_links.groupby("food_concept_id").food_group.nunique().value_counts().sort_index().to_dict(),
        foods_without_selected_cells=int((~partitions.food_concept_id.isin(profiles.food_concept_id)).sum()),
        foods_with_foodon=int(food_details.foodon_id.ne("").sum()),
        hierarchy_edges_model_children=sum(e["child_food_concept_id"] in set(ids) for e in hierarchy["edges"]),
        authority_namespace_counts=axis_details.authority_namespace.value_counts().to_dict(),
        role_status_counts=axis_details.nutritional_role_review_status.value_counts().to_dict(),
        chemical_class_available=int(axis_details.chemical_class.ne("unresolved_chemical_class").sum()),
        role_counts=axis_details.groupby(["nutritional_role", "training_role"]).size().unstack(fill_value=0).to_dict("index"),
        source_category_counts=food_links.groupby("source_key").food_group.nunique().to_dict(),
        inputs_sha256=before,
    )
    output.mkdir(parents=True, exist_ok=True)
    outputs = {"food_provenance": food_links, "axis_source_provenance": component_links, "source_contribution_counts": source_counts, "source_food_categories": groups, "source_axis_categories": axis_groups, "axis_registry_snapshot": axis_details, "axis_aliases_including_noncontributors": all_aliases, "food_registry_snapshot": food_details, "known_component_identity_review": known}
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    # Keep every selected measurement ID with its original record IDs; no numeric labels are copied into the report sidecars.
    values[["food_concept_id", "component_concept_id", "partition", "measurement_id", "source_key", "food_observation_id", "component_observation_id", "source_reference", "lineage_source_key"]].to_csv(output / "selected_value_provenance.csv.gz", index=False)
    after = {str(p.relative_to(root)): sha256(p) for p in inputs if p.is_file()}
    assert before == after, "Report generation must not modify any dataset artifact"
    (output / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "inputs_sha256"}, indent=2))
    print(source_counts.to_string(index=False))
    print(axis_groups.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/scientific_food_composition_v4/provenance_report"))
    parser.add_argument("--render-only", action="store_true", help="Reuse checked provenance ledgers, then regenerate the self-contained LaTeX")
    args = parser.parse_args()
    if not args.render_only:
        analyze(args.root, args.output)
    render_report(args.root, args.output)


if __name__ == "__main__":
    main()
