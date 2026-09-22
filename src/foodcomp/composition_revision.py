"""Reviewed analyte-level aliases and the requested v6 eligibility policy.

These are explicit decisions, not name-similarity rules. Original method,
identity, units and source labels remain in component_observation.
"""

import pandas as pd

from .util import stable_id


ROLE_POLICY = dict(minimum_train=50, minimum_validation=5,
                   require_verified_identity=False, exclude_expression_labels=False)

INFOODS = "https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/"
FIBRE = "https://www.fao.org/4/y5022e/y5022e03.htm"
CHEBI = "https://www.ebi.ac.uk/chebi/"


def group(key, name, representative, members, reason, refs):
    return dict(key=key, name=name, representative=representative,
                members=members.split(), reason=reason, evidence_urls=refs)


MERGES = [
    group("chromium_total", "Chromium, total", "INFOODS:CR",
          "INFOODS:CR CHEBI:49544", "FooDB Compound:3517 is source-defined Chromium (chemical element), not a species-specific Cr(III) assay. Its dictionary ion must not create an extra prediction axis; combine with elemental Cr and preserve the original structure metadata.", ["https://foodb.ca/", "https://frida.fooddata.dk/food/lists/parameters?lang=en", INFOODS]),
    group("nickel_total", "Nickel, total", "INFOODS:NI",
          "INFOODS:NI CHEBI:49786", "FooDB Compound:13447 is source-defined Nickel, with no species-specific Ni(II) assay. Combine its elemental-content observations with Ni, without interpreting the dictionary ion as a separate measurement.", ["https://foodb.ca/", "https://frida.fooddata.dk/food/lists/parameters?lang=en", INFOODS]),
    group("dietary_fibre_total", "Dietary fibre, total", "INFOODS:FIBTG",
          "INFOODS:FIBTG INFOODS:FIB- EUROFIR_EFSA:RF-00000284-NTR SOURCE:foodb:Nutrient:5",
          "One broad total dietary-fibre target (including old C001/C002); analytical methods remain distinct provenance, not asserted analytically identical. Excludes isolated fibre fractions.", [FIBRE, INFOODS]),
    group("cholesterol", "Cholesterol", "INFOODS:CHOLE",
          "INFOODS:CHOLE INFOODS:CHOL- INCHIKEY:HVYWMOMLDIMFJA-UHFFFAOYSA-N",
          "Same cholesterol analyte; enzymatic/chromatographic/unspecified methods are retained as measurement metadata.", [INFOODS, CHEBI + "CHEBI:16113"]),
    group("vitamin_b6_total", "Vitamin B6, total", "INFOODS:VITB6A",
          "INFOODS:VITB6A INFOODS:VITB6-", "Two total B6 expressions differing in method specificity; pyridoxine alone remains separate.", [INFOODS]),
    group("pantothenic_acid", "Pantothenic acid (vitamin B5)", "INFOODS:PANTAC",
          "INFOODS:PANTAC SOURCE:foodb:Compound:11820", "Vitamin B5 denotes pantothenic acid; no activity-equivalent conversion is performed.", [INFOODS, "https://ods.od.nih.gov/factsheets/PantothenicAcid-HealthProfessional/"]),
    group("vitamin_d3", "Cholecalciferol (vitamin D3)", "INFOODS:CHOCAL",
          "INFOODS:CHOCAL INCHIKEY:QYSXJUFSXHHAJI-FWSOMWAYSA-N", "Exact D3 synonym and stereochemical identity; unspecified vitamin D and D2 remain separate.", [INFOODS, CHEBI + "CHEBI:28940"]),
    group("vitamin_k1", "Phylloquinone (vitamin K1)", "INFOODS:VITK1",
          "INFOODS:VITK1 INCHIKEY:MBWXNTAXLNYFJB-LKUDQCMESA-N", "Phytomenadione/phylloquinone denotes vitamin K1; K2 homologues remain separate.", [INFOODS, CHEBI + "CHEBI:18067"]),
    group("alpha_tocotrienol", "Alpha-tocotrienol", "INFOODS:TOCTRA",
          "INFOODS:TOCTRA USDA_CNF_NUTRIENT_NBR:344", "Same alpha-tocotrienol analyte, not alpha-tocopherol or total vitamin E.", [INFOODS, "https://fdc.nal.usda.gov/food-data-downloads/"]),
    group("cystine", "Cystine", "INFOODS:CYS",
          "INFOODS:CYS CHEBI:16283", "INFOODS CYS is cystine, matching the source L-cystine identity; cysteine is not included.", [INFOODS, CHEBI + "CHEBI:16283"]),
    group("calcidiol", "25-Hydroxycholecalciferol (calcidiol)", "INFOODS:CHOCALOH",
          "INFOODS:CHOCALOH CHEBI:17933", "Calcidiol is the 25-hydroxy D3 metabolite, not unhydroxylated vitamin D3.", [INFOODS, CHEBI + "CHEBI:17933"]),
    group("epa", "Eicosapentaenoic acid (EPA; 20:5 n-3)", "INFOODS:F20D5N3",
          "INFOODS:F20D5N3 CHEBI:28364", "The standard food EPA expression corresponds to all-cis 20:5 n-3.", [INFOODS, CHEBI + "CHEBI:28364"]),
    group("dha", "Docosahexaenoic acid (DHA; 22:6 n-3)", "INFOODS:F22D6N3",
          "INFOODS:F22D6N3 CHEBI:28125 INCHIKEY:MBMBGCFOFBJSGT-KUBAVDMBSA-N", "Doconexent and DHA identify the same all-cis omega-3 acid.", [INFOODS, CHEBI + "CHEBI:28125"]),
    group("dpa_n3", "Docosapentaenoic acid (DPA; 22:5 n-3)", "INFOODS:F22D5N3",
          "INFOODS:F22D5N3 CHEBI:53488 INCHIKEY:YUFFSWGQGVEMMI-JLNKQSITSA-N SOURCE:foodb:Nutrient:36", "The 7Z,10Z,13Z,16Z,19Z identity is n-3 DPA; n-6 DPA remains separate.", [INFOODS, CHEBI + "CHEBI:53488"]),
    group("dpa_n6", "Docosapentaenoic acid (22:5 n-6)", "INFOODS:F22D5N6",
          "INFOODS:F22D5N6 CHEBI:65136", "The 4Z,7Z,10Z,13Z,16Z acid is n-6 DPA, not n-3 DPA.", [INFOODS, CHEBI + "CHEBI:65136"]),
    group("arachidonic_acid", "Arachidonic acid (20:4 n-6)", "INFOODS:F20D4N6",
          "INFOODS:F20D4N6 CHEBI:15843", "Specific all-cis n-6 arachidonic acid; unspecified 20:4 is not merged.", [INFOODS, CHEBI + "CHEBI:15843"]),
    group("stearidonic_acid", "Stearidonic acid (18:4 n-3)", "CHEBI:32389",
          "CHEBI:32389 INCHIKEY:JIWBIWFOSCKQMA-DFARDDQGSA-N", "Exact chemical identity; generic 18:4 without isomer definition remains separate.", [CHEBI + "CHEBI:32389"]),
    group("oleic_acid", "Oleic acid (18:1 cis n-9)", "INFOODS:F18D1CN9",
          "INFOODS:F18D1CN9 INCHIKEY:ZQPPMHVWECSIRJ-MDZDMXLPSA-N", "Oleic acid denotes cis-9 18:1, not the whole 18:1 or cis-18:1 aggregate.", [INFOODS, CHEBI + "CHEBI:16196"]),
    group("gamma_linolenic_acid", "Gamma-linolenic acid (18:3 n-6)", "CHEBI:28661",
          "CHEBI:28661 USDA_CNF_NUTRIENT_NBR:685", "Same all-cis n-6 18:3; not alpha-linolenic n-3 or unspecified 18:3.", [CHEBI + "CHEBI:28661", INFOODS]),
    group("cis_vaccenic_acid", "Cis-vaccenic acid (18:1 11c)", "CHEBI:50464",
          "CHEBI:50464 USDA_CNF_NUTRIENT_NBR:885", "Same cis-11 isomer; trans-vaccenic acid is distinct.", [CHEBI + "CHEBI:50464", "https://fdc.nal.usda.gov/food-data-downloads/"]),
    group("trans_18_1", "Fatty acids, trans 18:1, total", "INFOODS:F18D1T",
          "INFOODS:F18D1T EUROFIR_EFSA:RF-00000166-NTR", "Same unspecified positional trans-18:1 sum; individual positional isomers stay separate.", [INFOODS, "https://frida.fooddata.dk/food/lists/parameters?lang=en"]),
    group("trans_22_1", "Fatty acids, trans 22:1, total", "USDA_CNF_NUTRIENT_NBR:664",
          "USDA_CNF_NUTRIENT_NBR:664 SOURCE:frida:93", "Same trans-22:1 expression; no cis/trans pooling.", ["https://frida.fooddata.dk/food/lists/parameters?lang=en", "https://fdc.nal.usda.gov/food-data-downloads/"]),
    group("conjugated_linoleic_acids", "Conjugated linoleic acids (CLA), total", "USDA_CNF_NUTRIENT_NBR:670",
          "USDA_CNF_NUTRIENT_NBR:670 EUROFIR_EFSA:RF-00000171-NTR", "Same conjugated 18:2 family sum, not total linoleic acid or one CLA isomer.", [INFOODS, "https://frida.fooddata.dk/food/lists/parameters?lang=en"]),
]

EXCLUSIONS = {
    "SOURCE:foodb:Compound:0": "invalid_missing_source_component_id_zero_not_a_single_analyte",
    "INFOODS:CHOCDF": "user_removed_total_carbohydrate_calculated_by_difference",
    "SOURCE:foodb:Nutrient:38": "energy_is_not_chemical_mass",
    "EUROFIR_EFSA:RF-00000252-NTR": "below_detection_limit_summary_not_exact_mass_label",
    "INFOODS:VITA_RAE": "retinol_activity_equivalents_are_not_chemical_mass",
    "USDA_CNF_NUTRIENT_NBR:320": "retinol_activity_equivalents_are_not_chemical_mass",
}


def identity_key(row):
    namespace, identifier = str(row["authority_namespace"]), str(row["authority_id"])
    return identifier if identifier.startswith(namespace + ":") else namespace + ":" + identifier


def revise_compositions(components, mapping):
    result, mapping = components.copy(), mapping.copy()
    result["original_component_concept_ids"] = result.component_concept_id
    result["source_identity_keys"] = result.apply(identity_key, axis=1)
    result["deduplication_basis"] = "No supported same-analyte alias in the reviewed candidates."
    result["deduplication_evidence_urls"] = ""
    result["classification_representative_key"] = result.source_identity_keys
    touched, decisions = set(), []
    for decision in MERGES:
        found = result[result.source_identity_keys.isin(decision["members"])]
        if len(found) < 2:
            continue
        representative = found[found.source_identity_keys.eq(decision["representative"])]
        if len(representative) != 1:
            raise ValueError(f"Missing reviewed representative for {decision['key']}")
        old_ids = set(found.component_concept_id)
        if touched & old_ids:
            raise ValueError("Overlapping composition merge groups")
        touched |= old_ids
        row = representative.iloc[0].copy()
        new_id = stable_id("component", "v6_analyte", decision["key"])
        row["component_concept_id"] = new_id
        row["canonical_name"] = decision["name"]
        row["authority_namespace"], row["authority_id"] = "PROJECT", decision["key"]
        row["infoods_tag"], row["chebi_id"], row["cdno_id"] = "", "", ""
        row["definition"] = decision["reason"]
        row["identity_status"] = "reviewed_project_analyte_mapping"
        row["canonical_name_basis"] = "reviewed_same_analyte_alias_group_with_method_provenance"
        row["expression_variant"] = "reviewed_analyte_expression"
        row["original_component_concept_ids"] = ";".join(sorted(old_ids))
        row["source_identity_keys"] = ";".join(sorted(found.source_identity_keys))
        row["deduplication_basis"] = decision["reason"]
        row["deduplication_evidence_urls"] = " | ".join(decision["evidence_urls"])
        row["classification_representative_key"] = decision["representative"]
        sources = {s for value in found.source_keys for s in str(value).split(";") if s}
        row["source_keys"], row["source_count"] = ";".join(sorted(sources)), len(sources)
        for original in found.to_dict("records"):
            decisions.append(dict(old_component_concept_id=original["component_concept_id"],
                                  old_identity=original["source_identity_keys"], old_name=original["canonical_name"],
                                  new_component_concept_id=new_id, canonical_name=decision["name"],
                                  reason=decision["reason"], evidence_urls=row["deduplication_evidence_urls"]))
        mapping.loc[mapping.component_concept_id.isin(old_ids), "component_concept_id"] = new_id
        result = pd.concat([result[~result.component_concept_id.isin(old_ids)], row.to_frame().T], ignore_index=True)
    holds = []
    for row in result.to_dict("records"):
        key = identity_key(row)
        if key in EXCLUSIONS:
            holds.append(dict(component_concept_id=row["component_concept_id"], canonical_name=row["canonical_name"], reason=EXCLUSIONS[key]))
    if not result.component_concept_id.is_unique or not mapping.component_observation_id.is_unique:
        raise AssertionError("Composition revision lost unique identities")
    return result, mapping, pd.DataFrame(decisions), pd.DataFrame(holds, columns=["component_concept_id", "canonical_name", "reason"])
