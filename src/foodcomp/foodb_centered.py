"""FooDB-first identity and provenance rules for the v5 candidate release.

Raw source records are never edited. A measurement identity is distinct from
the chemical catalogue entry to which a historical FooDB import pointed.
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from .harmonize import _component_family, _role_from_component
from .ontology import parse_obo
from .schema import COMPONENT_COLUMNS
from .util import stable_id

EXCLUDED_SOURCES = frozenset({"afcd", "cofid", "norway"})
ALLOWED_SOURCES = frozenset({"foodb", "usda_sr_legacy", "usda_foundation", "cnf", "frida", "ciqual"})
INFOODS_URL = "https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/"
FRIDA_DEFINITION_URL = "https://doi.org/10.11583/DTU.32312844.v1"

# FCDB 6.1 Appendix A identifies these straight-chain fatty acids explicitly.
# This bridges EuroFIR F4:0 and INFOODS F4D0 without conflating isomers.
SATURATED_ACID_TAGS = {
    "CHEBI:30772": "F4D0", "CHEBI:30776": "F6D0", "CHEBI:28837": "F8D0",
    "CHEBI:30813": "F10D0", "CHEBI:30805": "F12D0", "CHEBI:45919": "F13D0",
    "CHEBI:28875": "F14D0", "CHEBI:42504": "F15D0", "CHEBI:15756": "F16D0",
    "CHEBI:32365": "F17D0", "CHEBI:28842": "F18D0", "CHEBI:28822": "F20D0",
    "CHEBI:39248": "F21D0", "CHEBI:28941": "F22D0", "CHEBI:42394": "F23D0",
    "CHEBI:28866": "F24D0",
}
SPECIFIC_ACID_TAGS = {
    "CHEBI:16196": "F18D1CN9", "CHEBI:17351": "F18D2CN6",
    "CHEBI:27432": "F18D3CN3", "CHEBI:73731": "F20D2CN6",
    "CHEBI:28792": "F22D1CN9", "CHEBI:32428": "F22D1CN11",
}
# The source documentation supersedes illustrative molecular cross-references.
# These totals must not be labelled as a single vitamer's chemical mass.
FRIDA_EQUIVALENT_EXPRESSIONS = {
    "37": "vitamin B1 activity, expressed as thiamine chloride",
    "40": "vitamin B6 activity, expressed as pyridoxine hydrochloride",
    "143": "folate activity, expressed as folic acid equivalent",
    "38": "vitamin B12 activity, expressed as cyanocobalamin equivalent",
    "294": "preformed niacin, expressed as nicotinic acid equivalent",
}

# Explicit equivalent labels, not a fuzzy matcher. Totals, equivalents, free
# forms, fatty-acid isomers and carbohydrate expressions are not collapsed.
TAG_ALIASES = {
    "PROCNT": ["protein", "proteins", "total protein"],
    "FAT": ["fat", "total lipid (fat)", "fat, total", "total fat"],
    "CHO-": ["carbohydrate"],
    "CHOCDF": ["carbohydrate, by difference", "carbohydrate by difference"],
    "CHOAVL": ["carbohydrates, total available"],
    "ASH": ["ash"], "WATER": ["water", "moisture"], "NT": ["nitrogen", "nitrogen, total"],
    "STARCH": ["starch", "starch, total"], "ALC": ["alcohol, ethyl", "ethanol"],
    "MG": ["magnesium", "magnesium, mg"], "CA": ["calcium", "calcium, ca"],
    "FE": ["iron", "iron, fe"], "P": ["phosphorus", "phosphorus, p"],
    "K": ["potassium", "potassium, k"], "NA": ["sodium", "sodium, na"],
    "ZN": ["zinc", "zinc, zn"], "CU": ["copper", "copper, cu"],
    "MN": ["manganese", "manganese, mn"], "SE": ["selenium", "selenium, se"],
    "CLD": ["chloride"], "ID": ["iodine"], "MO": ["molybdenum"], "FD": ["fluoride", "fluoride, f"],
    "THIA": ["thiamin", "thiamine", "thiamin|thiamine"], "RIBF": ["riboflavin", "riboflavine"],
    "PANTAC": ["pantothenic acid"], "BIOT": ["biotin"], "ASCL": ["l-ascorbic acid"],
    "ASCDL": ["vitamin c, l-dehydroascorbic acid", "l-dehydroascorbic acid"],
    "FOLAC": ["folic acid"], "CHOLN": ["choline, total", "total choline"],
    "TOCPHA": ["alpha-tocopherol", "tocopherol, alpha"],
    "TOCPHB": ["beta-tocopherol", "tocopherol, beta"],
    "TOCPHG": ["gamma-tocopherol", "tocopherol, gamma"],
    "TOCPHD": ["delta-tocopherol", "tocopherol, delta"],
    "CHOCAL": ["cholecalciferol"], "ERGCAL": ["ergocalciferol"],
    "CARTA": ["alpha-carotene"], "CARTB": ["beta-carotene"], "LYCPN": ["lycopene"],
    "LUTN": ["lutein"], "ZEA": ["zeaxanthin"],
    "ALA": ["alanine", "l-alanine"], "ARG": ["arginine", "l-arginine"],
    "ASP": ["aspartic acid", "l-aspartic acid"], "GLU": ["glutamic acid", "l-glutamic acid"],
    "GLY": ["glycine"], "HIS": ["histidine", "l-histidine"],
    "ILE": ["isoleucine", "l-isoleucine"], "LEU": ["leucine", "l-leucine"],
    "LYS": ["lysine", "l-lysine"], "MET": ["methionine", "l-methionine"],
    "PHE": ["phenylalanine", "l-phenylalanine"], "PRO": ["proline", "l-proline"],
    "SER": ["serine", "l-serine"], "THR": ["threonine", "l-threonine"],
    "TRP": ["tryptophan", "l-tryptophan"], "TYR": ["tyrosine", "l-tyrosine"],
    "VAL": ["valine", "l-valine"],
    "GLUS": ["glucose"], "FRUS": ["fructose"], "SUCS": ["sucrose"],
    "LACS": ["lactose"], "MALS": ["maltose"], "GALS": ["galactose"],
    "F4D0": ["butyric acid", "butanoic acid"],
    "F14D0": ["myristic acid", "tetradecanoic acid", "14:0"],
    "F16D0": ["palmitic acid", "hexadecanoic acid", "16:0"],
    "F18D0": ["stearic acid", "octadecanoic acid", "18:0"],
    "F20D0": ["arachidic acid", "icosanoic acid", "20:0"],
    "F22D0": ["behenic acid", "docosanoic acid", "22:0"],
    "F24D0": ["lignoceric acid", "tetracosanoic acid", "24:0"],
    "FASAT": ["fatty acids, total saturated", "sfa"],
    "FAMS": ["fatty acids, total monounsaturated"],
    "FAPU": ["fatty acids, total polyunsaturated"],
}


def name_key(value):
    text = unicodedata.normalize("NFKC", str(value)).strip().casefold()
    return re.sub(r"\s+", " ", text)


def nutrient_number(definition):
    match = re.search(r"(?:number|code)\s+([0-9]+(?:\.[0-9]+)?)", str(definition), re.I)
    return match.group(1) if match else ""


def recover_orphans(observations, inventory, root):
    """Recover a source identity; a ChEBI ion is not silently made an element."""
    rows, decisions = [], []
    chebi = parse_obo(root / "data/raw/ontology/chebi_lite.obo.gz")
    for row in inventory.fillna("").itertuples(index=False):
        ids = [x for x in row.chebi_ids.split(";") if x]
        original = row.most_frequent_original_name
        name = original or (chebi[ids[0]]["name"] if len(ids) == 1 and ids[0] in chebi else "")
        if not name and row.curated_synonyms:
            # Synonyms establish a local identity, not permission to mix it
            # with a stereochemically different or aggregate food analyte.
            name = sorted(row.curated_synonyms.split(" | "), key=lambda x: (len(x), x))[0]
        status = "recovered_original_content_identity" if name and str(row.source_id) != "0" else "unresolved_no_identity"
        entry = {col: "" for col in COMPONENT_COLUMNS}
        entry.update(component_observation_id=row.component_observation_id, source_key="foodb",
                     source_component_id="Compound:" + str(row.source_id), original_name=name or "Unresolved FooDB compound " + str(row.source_id),
                     source_component_group="FooDB Compound (Content-linked dictionary recovery)",
                     source_definition="Original FooDB Content component; original names: " + row.original_names,
                     chebi_id=ids[0] if len(ids) == 1 else "", authority_status=status,
                     recovery_evidence_url=row.example_evidence_url, recovery_original_names=row.original_names)
        rows.append(entry)
        decisions.append({**row._asdict(), "recovered_name": name, "identity_decision": status,
                          "decision_basis": "Content.source_id/orig_source_name; CompoundSynonym; CompoundExternalDescriptor; ChEBI snapshot",
                          "individual_web_review": False})
    combined = pd.concat([observations, pd.DataFrame(rows)], ignore_index=True).fillna("")
    if combined.component_observation_id.duplicated().any():
        raise ValueError("Recovered component ID already exists in source registry")
    return combined, pd.DataFrame(decisions)


def prepare_identities(observations, infoods):
    """Explicit label-to-tag crosswalk; original fields stay in the release."""
    result = observations.copy().fillna("")
    result["harmonization_name"] = result.original_name
    result["harmonization_tag"] = result.infoods_tag
    result["harmonization_chebi"] = result.get("chebi_id", pd.Series("", index=result.index))
    result["identity_rule"] = "original_formal_identifier_or_source_specific"
    result["identity_evidence_url"] = ""
    result["harmonization_exclusion_reason"] = ""
    aliases = {name_key(n): tag for tag, names in TAG_ALIASES.items() if tag in infoods for n in names}
    for index, row in result.iterrows():
        tag = aliases.get(name_key(row.original_name))
        original_tag = row.infoods_tag
        if str(row.get("inchikey", "")) == "LQJBNNIYVWPHFW-QXMHVHEDSA-N":
            result.at[index, "harmonization_chebi"] = "CHEBI:32419"
            result.at[index, "identity_rule"] = "PubChem_5282767_InChIKey_to_CHEBI_32419_not_11Z_isomer"
            result.at[index, "identity_evidence_url"] = "https://pubchem.ncbi.nlm.nih.gov/compound/5282767"
        if tag and (not original_tag or original_tag == tag or original_tag not in infoods):
            result.at[index, "harmonization_tag"] = tag
            result.at[index, "identity_rule"] = "reviewed_exact_analyte_label_to_INFOODS"
        chemical_tag = {**SATURATED_ACID_TAGS, **SPECIFIC_ACID_TAGS}.get(str(row.get("chebi_id", "")))
        if chemical_tag in infoods and (not original_tag or original_tag not in infoods):
            result.at[index, "harmonization_tag"] = chemical_tag
            result.at[index, "identity_rule"] = "FCDB_appendix_A_CHEBI_identity_to_INFOODS_expression"
            result.at[index, "identity_evidence_url"] = FRIDA_DEFINITION_URL
        if row.source_key == "frida" and str(row.source_component_id) == "115":
            result.at[index, "harmonization_tag"] = "CHOLE"
            result.at[index, "identity_rule"] = "FCDB_section_8_2_chromatographic_cholesterol"
            result.at[index, "identity_evidence_url"] = FRIDA_DEFINITION_URL
        if row.source_key == "frida" and str(row.source_component_id) in FRIDA_EQUIVALENT_EXPRESSIONS:
            result.at[index, "harmonization_name"] = FRIDA_EQUIVALENT_EXPRESSIONS[str(row.source_component_id)]
            result.at[index, "harmonization_tag"] = ""
            result.at[index, "harmonization_exclusion_reason"] = "documented_vitamer_equivalent_not_single_chemical_mass"
            result.at[index, "identity_rule"] = "FCDB_sections_6_5_to_6_11_measured_vitamer_total"
            result.at[index, "identity_evidence_url"] = FRIDA_DEFINITION_URL
        if str(row.source_component_id) == "Compound:3519" and row.source_key == "foodb":
            result.at[index, "harmonization_tag"] = "MG"
            result.at[index, "identity_rule"] = "Content_USDA_304_and_DUKE_MAGNESIUM_measurement_not_salt_mass"
        if str(row.source_component_id) == "Compound:8426" and row.source_key == "foodb":
            result.at[index, "harmonization_tag"] = "THIA"
            result.at[index, "identity_rule"] = "Content_THIAMIN_with_CHEBI_18385_identity"
        if row.source_key == "foodb" and row.source_component_id == "Compound:21595":
            result.at[index, "authority_status"] = "unresolved_mixed_MUFA_PUFA_import_id"
        if row.source_key == "foodb" and row.source_component_id == "Nutrient:4":
            result.at[index, "authority_status"] = "unresolved_unspecified_fatty_acid_aggregate"
    return result


def virtual_import_components(measurements, observations, metadata):
    """Resolve FooDB USDA imports using their original nutrient number, not ID proximity."""
    result = measurements.copy()
    result["original_component_observation_id"] = result.component_observation_id
    fields = ["measurement_id", "orig_source_id", "orig_source_name", "source_id", "source_type", "id"]
    result = result.merge(metadata[fields], on="measurement_id", how="left", validate="one_to_one")
    usda = observations[observations.source_key.eq("usda_sr_legacy")].copy()
    usda["nutrient_number"] = usda.source_definition.map(nutrient_number)
    usda = usda[usda.nutrient_number.ne("")].set_index("nutrient_number")
    if usda.index.duplicated().any():
        raise ValueError("USDA nutrient-number registry is ambiguous")
    rows = []
    selected = result.source_key.eq("foodb") & result.source_reference.eq("USDA") & result.orig_source_id.isin(usda.index)
    for (old_id, number), group in result[selected].groupby(["component_observation_id", "orig_source_id"]):
        native = usda.loc[number]
        ident = stable_id("compobs", "foodb", old_id, "original_USDA_nutrient", number)
        entry = native.to_dict()
        entry.update(component_observation_id=ident, source_key="foodb",
                     source_component_id="import:" + old_id + ":USDA:" + number,
                     original_name=group.orig_source_name.dropna().iloc[0] if group.orig_source_name.notna().any() else native.original_name,
                     source_component_group="FooDB " + group.source_type.iloc[0] + " (USDA-coded Content expression)",
                     parent_component_observation_id=old_id,
                     source_definition="FooDB Content.orig_source_id = USDA nutrient number " + number,
                     identity_rule="record_level_original_USDA_nutrient_number",
                     recovery_evidence_url="https://foodb.ca/contents/" + str(group.id.iloc[0]))
        rows.append(entry)
        result.loc[group.index, "component_observation_id"] = ident
    return result, pd.concat([observations, pd.DataFrame(rows)], ignore_index=True).fillna("")


def harmonization_view(observations):
    view = observations.copy()
    view["original_name"] = view.harmonization_name
    view["infoods_tag"] = view.harmonization_tag
    view["chebi_id"] = view.harmonization_chebi
    equivalent = view.harmonization_exclusion_reason.ne("")
    view.loc[equivalent, ["chebi_id", "eurofir_component_id", "inchikey"]] = ""
    return view


def retain_internal_reference_values(measurements):
    """Accept source-published citations without re-adjudicating their origin.

    Only the copy-specific gate is relaxed. Missing, invalid, censored and
    non-mass values keep their existing exclusions and provenance flags.
    """
    result = measurements.copy()
    internal = result.exclusion_reason.eq("external_reference_requires_origin_resolution_and_deduplication")
    numeric = pd.to_numeric(result.normalized_value_g_per_100g, errors="coerce")
    valid = (numeric.between(0, 100) & result.value_status.isin(["observed", "explicit_zero"])
             & result.conversion_status.eq("converted_exact_mass")
             & result.measurement_modality.eq("mass_fraction_fresh_weight"))
    if (internal & ~valid).any():
        raise ValueError("Internal-citation policy cannot admit an invalid or non-mass value")
    result.loc[internal, "main_value_eligible"] = True
    result.loc[internal, "validation_reference_eligible"] = True
    result.loc[internal, "exclusion_reason"] = ""
    result.loc[internal, "source_policy_decision"] = "admitted_source_published_internal_citation"
    result.loc[internal, "source_policy_reason"] = "user_2026_09_09_internal_citations_preserved_without_origin_resolution"
    result.loc[internal, "validation_evidence_basis"] = "trusted_database_reference_not_individual_assay_certification"
    return result


def merge_equivalent_concepts(concepts, mapping, observations, infoods):
    """Make formal INFOODS identity independent of spelling-derived variants."""
    c = concepts.copy()
    m = mapping.copy()
    # INFOODS already specifies the analytical expression. A spelling-derived
    # 'by difference' flag must not create a second index for the same tag.
    remap = {}
    for (ns, authority), group in c.groupby(["authority_namespace", "authority_id"]):
        if ns == "INFOODS" and len(group) > 1:
            winner = sorted(group.component_concept_id)[0]
            remap.update({x: winner for x in group.component_concept_id})
    m["component_concept_id"] = m.component_concept_id.map(lambda x: remap.get(x, x))
    c["component_concept_id"] = c.component_concept_id.map(lambda x: remap.get(x, x))
    c = c.drop_duplicates("component_concept_id").set_index("component_concept_id")
    links = m.merge(observations, on="component_observation_id", validate="one_to_one")
    for identifier, group in links.groupby("component_concept_id"):
        c.at[identifier, "source_keys"] = ";".join(sorted(set(group.source_key)))
        c.at[identifier, "source_count"] = group.source_key.nunique()
        if group.authority_status.str.startswith("unresolved_").all():
            c.at[identifier, "identity_status"] = "unresolved_component_identity"
        tag = c.at[identifier, "infoods_tag"]
        name = c.at[identifier, "canonical_name"]
        role, basis, url, status = _role_from_component(tag, name, "")
        if tag in {"NT", "ASH", "WATER"}:
            role, basis, url, status = "other_nutritional_component", "INFOODS proximate component; not an amino acid", INFOODS_URL, "authority_supported"
        if tag in {"LYCPN", "LUTN", "ZEA", "CARTA", "CARTB"}:
            role, basis, url, status = "other_food_component", "Carotenoid chemical constituent; not a standalone essential vitamin", INFOODS_URL, "authority_supported"
        if role == "other_food_component" and c.at[identifier, "chemical_class"] != "unresolved_chemical_class":
            basis, url = "Chemical class retained independently of nutritional role", "https://www.ebi.ac.uk/chebi/"
        for key, value in zip(["nutritional_role", "nutritional_role_basis", "nutritional_role_evidence_url", "nutritional_role_review_status"], [role, basis, url, status]):
            c.at[identifier, key] = value
        c.at[identifier, "component_family"] = _component_family(tag, name, role)
        if name_key(name) == "cholesterol":
            c.at[identifier, "component_family"] = "lipid_and_fatty_acid_family"
    c = c.reset_index()
    return c, m


def foo_first_selection(measurements, mapping, component_mapping, partitions, *, resolve_internal_copies=True):
    """Same-origin copies never increase aggregation weight or sample count."""
    m = measurements.merge(mapping[["food_observation_id", "food_concept_id"]], on="food_observation_id", validate="many_to_one")
    m = m.merge(component_mapping[["component_observation_id", "component_concept_id"]], on="component_observation_id", how="left", validate="many_to_one")
    if m.component_concept_id.isna().any():
        raise ValueError("Admitted measurement has no component identity mapping")
    m = m.merge(partitions[["food_concept_id", "partition", "source_holdout"]], on="food_concept_id", how="inner", validate="many_to_one")
    m["origin_dataset"] = m.source_key
    refs = {"USDA": "usda_sr_legacy", "DTU": "frida", "FRIDA": "frida", "PHENOL EXPLORER": "phenol_explorer"}
    copy = m.source_key.eq("foodb") & m.source_reference.isin(refs)
    m.loc[copy, "origin_dataset"] = m.loc[copy, "source_reference"].map(refs)
    admitted = m.main_value_eligible.astype(str).str.casefold().eq("true")
    m["selection_decision"] = np.where(admitted, "eligible", "existing_record_level_exclusion")
    # Resolve an imported value only against the same food concept, same
    # measurement identity and original dataset. Other sources cannot certify it.
    if resolve_internal_copies:
        direct = m[~m.source_key.eq("foodb")].groupby(["food_concept_id", "component_concept_id", "source_key"]).normalized_value_g_per_100g.median().rename("origin_reference_value_g_per_100g").reset_index().rename(columns={"source_key": "origin_dataset"})
        m = m.merge(direct, on=["food_concept_id", "component_concept_id", "origin_dataset"], how="left", validate="many_to_one")
        copy = m.source_key.eq("foodb") & m.source_reference.isin(refs)
        linked = m.origin_reference_value_g_per_100g.notna()
        agrees = np.isclose(m.normalized_value_g_per_100g, m.origin_reference_value_g_per_100g, rtol=1e-4, atol=1e-8)
        m.loc[copy & linked & agrees, "selection_decision"] = "copied_value_resolved_to_original"
        m.loc[copy & linked & ~agrees, "selection_decision"] = "copied_value_differs_use_original_review"
        m.loc[copy & ~linked, "selection_decision"] = "unresolved_copy_archive_not_new_label"
    m.loc[m.source_holdout & ~m.source_key.eq("usda_foundation"), "selection_decision"] = "source_holdout_requires_foundation"
    eligible = m.selection_decision.eq("eligible")
    foo_cells = set(map(tuple, m.loc[eligible & m.source_key.eq("foodb"), ["food_concept_id", "component_concept_id"]].to_numpy()))
    # FooDB is the anchor. Source-published FooDB records provide a cell; supplementary
    # sources fill gaps, with the existing quality/heterogeneity aggregation.
    supplemental_overlap = eligible & ~m.source_key.eq("foodb") & pd.Series(
        [tuple(x) in foo_cells for x in m[["food_concept_id", "component_concept_id"]].to_numpy()], index=m.index)
    m.loc[supplemental_overlap, "selection_decision"] = "supplement_not_used_foodb_cell_present"
    keep = set(m.loc[m.selection_decision.eq("eligible"), "measurement_id"])
    ledger_columns = ["measurement_id", "food_concept_id", "component_concept_id", "source_key", "source_reference",
                      "origin_dataset", "selection_decision", "normalized_value_g_per_100g", "origin_reference_value_g_per_100g"]
    for column in ledger_columns:
        if column not in m:
            m[column] = np.nan
    return measurements[measurements.measurement_id.isin(keep)].copy(), m[ledger_columns]
