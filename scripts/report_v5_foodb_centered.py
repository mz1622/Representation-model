"""Create the self-contained LaTeX provenance report for the v5 candidate."""

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.util import sha256_file, write_csv, write_json
from foodcomp.composition_classification import (
    classify_components, classification_counts, classification_findings,
    classification_markdown, load_classification,
)
from report_v4_provenance import tex, identifier, table, SOURCES, FOOD_FIELDS

VERSION = "scientific_food_composition_v5_1"


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[""], low_memory=False).fillna("")


def main():
    release = ROOT / "data/processed" / VERSION / "release"
    audit = ROOT / "data/audits" / VERSION / "dataset"
    output = ROOT / "reports" / VERSION
    protected = [
        release / "canonical_profile_matrix.npz",
        release / "component_concept.csv.gz",
        release / "ml_partition.csv",
    ]
    before_hashes = {path.name: sha256_file(path) for path in protected}
    specification_path = ROOT / "data/reference/composition_classification_v5_1.json"
    specification = load_classification(specification_path)
    summary = json.loads((release / "dataset_summary.json").read_text())
    review_summary = json.loads((audit / "audit_summary.json").read_text())
    values = np.load(release / "canonical_profile_matrix.npz", allow_pickle=False)
    foods = read(release / "food_observation.csv.gz")
    mapping = read(release / "food_observation_to_concept.csv.gz")
    partitions = read(release / "ml_partition.csv")
    source_components = read(release / "component_observation.csv.gz")
    cm = read(release / "component_observation_to_concept.csv.gz")
    all_axes = read(release / "component_concept.csv.gz")
    axes = all_axes[all_axes.component_concept_id.isin(values["component_ids"].astype(str))]
    axes = classify_components(axes, specification)
    role_counts = classification_counts(axes, specification)
    findings = classification_findings(all_axes, axes)
    profiles = read(release / "canonical_profile.csv.gz")
    profiles = profiles[profiles.aggregation_status.eq("accepted") & profiles.food_concept_id.isin(values["food_ids"].astype(str)) & profiles.component_concept_id.isin(values["component_ids"].astype(str))]
    assert len(profiles) == values["observed"].sum()
    measurements = read(release / "measurement_main_eligible.csv.gz")
    selected_ids = set(profiles.measurement_ids.str.split(";").explode())
    measurements = measurements[measurements.measurement_id.isin(selected_ids)]
    actual = measurements.merge(mapping[["food_observation_id", "food_concept_id"]], on="food_observation_id", validate="many_to_one")
    actual = actual.merge(cm[["component_observation_id", "component_concept_id"]], on="component_observation_id", validate="many_to_one")
    actual = actual.merge(partitions[["food_concept_id", "partition", "dataset_layer"]], on="food_concept_id", validate="many_to_one")
    used_observations = set(actual.food_observation_id)
    food_ledger = foods.merge(mapping, on="food_observation_id", validate="one_to_one").merge(partitions[["food_concept_id", "partition", "dataset_layer"]], on="food_concept_id", how="left", validate="many_to_one")
    food_ledger["selected_numerical_contributor"] = food_ledger.food_observation_id.isin(used_observations)
    food_ledger["retained_nonempty_food"] = food_ledger.food_concept_id.isin(profiles.food_concept_id)
    write_csv(food_ledger, output / "food_provenance.csv")
    aliases = cm.merge(source_components, on="component_observation_id", validate="one_to_one")
    aliases = aliases[aliases.component_concept_id.isin(axes.component_concept_id)].copy()
    aliases["selected_numerical_contributor"] = aliases.component_observation_id.isin(actual.component_observation_id)
    write_csv(aliases, output / "composition_source_aliases.csv")
    native_rows = []
    for component_id, group in aliases.groupby("component_concept_id"):
        categories = []
        names = []
        for source, source_group in group.groupby("source_key"):
            categories.append(source + ": " + "; ".join(sorted(set(source_group.source_component_group))))
            names.append(source + ": " + "; ".join(sorted(set(source_group.original_name))))
        native_rows.append({
            "component_concept_id": component_id,
            "original_source_categories": " | ".join(categories),
            "original_names_by_source": " | ".join(names),
        })
    axes = axes.merge(pd.DataFrame(native_rows), on="component_concept_id", validate="one_to_one")
    write_csv(axes, output / "unique_composition_registry.csv")
    classification_columns = [
        "component_concept_id", "canonical_name", "authority_namespace", "authority_id",
        "original_names_by_source", "original_source_categories",
        "macro_micro_class", "classification_category_en", "classification_category_zh",
        "classification_subtype", "parent_nutrient_or_family", "reporting_chemical_family",
        "classification_reason_en", "classification_reason_zh", "essentiality_interpretation",
        "classification_evidence_urls", "classification_rule_id", "classification_review_status",
        "classification_limitation", "classification_recommended_action",
        "training_role", "train_count", "validation_count", "classification_version",
        "classification_is_project_mapping", "classification_expert_signed_off",
    ]
    write_csv(axes[classification_columns], output / "composition_classification.csv")
    write_csv(role_counts, output / "composition_classification_counts.csv")
    write_csv(findings, output / "classification_data_issues.csv")
    (output / "composition_classification.md").write_text(
        classification_markdown(axes, role_counts, specification), encoding="utf-8"
    )
    write_json(
        {key: value for key, value in specification.items() if key != "lookup"},
        output / "classification_evidence_and_rules.json",
    )
    write_csv(actual, output / "selected_value_provenance.csv.gz")
    selection = read(audit / "measurement_selection_ledger.csv.gz")
    copies = selection[selection.source_key.eq("foodb")]
    reference_counts = copies.groupby(["source_reference", "selection_decision"]).size().unstack(fill_value=0)
    write_csv(reference_counts.reset_index(), output / "foodb_reference_resolution.csv")
    orphan = read(audit / "orphan_compound_final_decisions.csv")
    selected_orphan = actual.merge(axes[["component_concept_id", "canonical_name", "training_role"]], on="component_concept_id", validate="many_to_one")
    selected_orphan = selected_orphan.groupby("original_component_observation_id").agg(
        selected_component_ids=("component_concept_id", lambda x: ";".join(sorted(set(x)))),
        selected_canonical_names=("canonical_name", lambda x: "; ".join(sorted(set(x)))),
        selected_training_roles=("training_role", lambda x: "; ".join(sorted(set(x)))),
    )
    orphan = orphan.merge(selected_orphan, left_on="component_observation_id", right_index=True, how="left")
    orphan["source_dictionary_canonical_name"] = orphan.canonical_name
    orphan["source_dictionary_training_role"] = orphan.training_role
    orphan["canonical_name"] = orphan.selected_canonical_names.fillna(orphan.canonical_name)
    orphan["training_role"] = orphan.selected_training_roles.fillna(orphan.training_role)
    write_csv(orphan, output / "orphan_compound_review.csv")

    sections = []
    sections.append(r"\section*{Executive Summary}")
    sections.append("This dataset is anchored on FooDB food observations and supplemented by USDA SR Legacy, CNF, Frida and CIQUAL. USDA Foundation Foods remains a source-holdout reference. AFCD, CoFID and Norway are excluded as direct numerical and descriptive sources. Internal citations within the retained databases are preserved without separately excluding or replacing those published values. This is a scope decision, not a claim that the excluded databases are scientifically unreliable. FNDDS remains outside the primary dataset.")
    n = int(values["observed"].sum())
    sections.append(f"The candidate contains \\textbf{{{summary['nonempty_food_concepts']:,} non-empty food profiles}}, including {summary['foodb_anchor_nonempty_foods']:,} FooDB-linked profiles and {summary['supplementary_nonempty_foods']:,} supplementary food identities, with \\textbf{{{len(axes)} unique retained composition expressions}} and \\textbf{{{n:,} observed cells}}. {summary['maskable_targets']} expressions meet the current maskable-target rules; {summary['context_only_components']} are context-only. Names and canonical indices are unique; original source names and IDs remain many-to-one aliases.")
    sections.append(f"The original 1,426 validation identities are preserved; {summary['validation_nonempty_foods']:,} have retained numerical values in this version. Identities without labels are explicitly not benchmark-eligible. There are {summary['train_food_concepts']:,} non-empty training profiles. No test partition, new training run, or validation outcome evaluation was produced. The release is a \\textbf{{candidate pending domain review}}, not a claim of complete biochemical coverage or individually certified laboratory ground truth.")
    sections.append("\\textbf{Classification review warning.} All 293 retained expressions now have an explicit nutritional/chemical category and an evidence-linked rationale. This is complete annotation coverage, not complete nutrient coverage. The review found an Energy expression incorrectly present in the mass-labelled matrix and nitrogen-derived total protein excluded by an inherited label-expression rule. Numerical data and partitions have not been silently rebuilt; these are release-blocking issues for an unqualified nutrition-training claim.")
    sections.extend([r"\tableofcontents", r"\clearpage", r"\section{Foods: FooDB as the Starting Point}", r"\subsection{What Is Counted as a FooDB Food?}"])
    foo = foods[foods.source_key.eq("foodb")]
    foo_linked = food_ledger[food_ledger.source_key.eq("foodb")]
    sections.append("FooDB's original Food.csv contains 992 base food entries. A content-level observation is identified by citation, original food ID, food part and preparation. These observations are not automatically independent biological samples. A local food concept can be shared with an original USDA or Frida record without making the copied value an additional observation of biological composition.")
    sections.append(table(["Counting level", "Count"], [
        ["FooDB base Food.csv entries", "992"],
        ["FooDB content-level source observations", f"{len(foo):,}"],
        ["FooDB-linked concepts with retained values", f"{summary['foodb_anchor_nonempty_foods']:,}"],
        ["Additional non-empty supplementary food concepts", f"{summary['supplementary_nonempty_foods']:,}"],
        ["Concepts with selected numerical values supplied directly through FooDB", str(actual.loc[actual.source_key.eq('foodb'), 'food_concept_id'].nunique())],
    ], [0.79, 0.14]))
    sections.append(r"\subsection{Databases Quoted Inside FooDB}")
    citation = foo.source_food_id.str.split("|").str[0].value_counts()
    reference_rows = [[tex(key), f"{citation.get(key, 0):,}"] for key in ["USDA", "DTU", "DUKE", "MANUAL", "PHENOL EXPLORER", "PATHBANK", "KNAPSACK", "PHYTOHUB", "HMDB"]]
    reference_rows.append(["Other citation strings", f"{int(citation.sum() - sum(citation.get(x, 0) for x in ['USDA','DTU','DUKE','MANUAL','PHENOL EXPLORER','PATHBANK','KNAPSACK','PHYTOHUB','HMDB'])):,}"])
    sections.append(table(["FooDB citation", "Source food observations"], reference_rows, [0.66, 0.27]))
    sections.append("These counts include observations that ultimately have no usable main-modality values. They must not be added to USDA or Frida's food counts as independent foods. FooDB's nutrient documentation identifies historical USDA, Danish and Duke sources \\cite{foodbcarb}. Immediate supplier, cited origin and canonical food identity are recorded separately.")
    sections.append(r"\subsection{How the Supplementary Databases Are Used}")
    sections.append("For an eligible food--composition cell, a FooDB-published record is the anchor, including values that FooDB cites from other databases. Supplementary sources fill cells without such a FooDB record and add clearly marked food identities absent from FooDB. Within the selected source tier, the established quality/heterogeneity aggregation is retained. This is a project integration policy, not an international source-quality ranking. Each cell's full selected measurement list is retained.")
    sections.append("Internal citations are provenance metadata, not an additional exclusion criterion. FooDB's USDA/DTU imports and citations within other retained sources are not replaced by newer originating-source values, subjected to agreement thresholds or independently rescaled. The immediate supplier, published value and cited reference are preserved. Standard food--component cell uniqueness still applies: the same canonical cell is not repeated just because a supplementary database also supplies it. Published imports are not claimed to be independent laboratory replications. Existing family and benchmark-lineage safeguards remain in place.")
    resolution_rows = [[tex(key.replace("_", " ")), f"{int(count):,}"] for key, count in copies.selection_decision.value_counts().items()]
    sections.append(table(["FooDB candidate-record disposition", "Records"], resolution_rows, [0.77, 0.16]))
    sections.append("The disposition table precedes final axis filtering and aggregation; its totals are not counts of final matrix cells. Foundation source-holdout cells can be supplied only by Foundation records. The frozen family/lineage blocking policy remains in force.")
    source_rows = []
    order = ["foodb", "usda_sr_legacy", "cnf", "frida", "ciqual", "usda_foundation"]
    for source in order:
        x = actual[actual.source_key.eq(source)]
        source_rows.append([tex(SOURCES[source][0]), f"{x.food_concept_id.nunique():,}", f"{x.component_concept_id.nunique():,}", f"{len(x):,}", f"{x[['food_concept_id','component_concept_id']].drop_duplicates().shape[0]:,}"])
    sections.append(table(["Selected supplier", "Foods", "Axes", "Records", "Cells"], source_rows, [0.30, 0.13, 0.10, 0.18, 0.18], "Actual selected numerical contributions; source food/cell counts can overlap."))
    sections.append(r"\subsection{Source Food Categories and Project Categories}")
    sections.append(table(["Source", "Native food category fields"], [[tex(SOURCES[s][0]), tex(FOOD_FIELDS[s])] for s in order], [0.19, 0.74]))
    sections.append("For a FooDB-linked concept, the canonical display name and representative source food group come from a FooDB observation. Source names, groups, descriptions and IDs remain available for all aliases. A base FooDB group inherited by several content observations is not an expert-verified category for each preparation. The project-created categories foodb\\_anchor and supplementary\\_food\\_extension describe provenance only. Existing species/part/preparation facets and family blocks preserve food identity and split continuity; food values are not inherited from parent foods. No globally adjudicated FoodOn hierarchy is claimed \\cite{infoodsmatch}.")

    sections.extend([r"\section{Composition: Unique Identity, Source Aliases and Classification}", r"\subsection{A Unique Axis Is a Unique Defined Quantity}"])
    sections.append("An axis may be a molecule, an elemental amount, a nutrient total or a defined analytical aggregate. Its identity is not its source database. Equivalent source expressions are assigned one canonical index and one unique canonical name, with all native names retained in the alias table. INFOODS tags and verified chemical identifiers support the mapping; explicitly reviewed equivalent labels bridge chemical and nutrition registries. Number parsing preserves decimal nutrient codes, so a total and a decimal-coded specific form cannot collapse because of integer truncation.")
    sections.append("The FCDB/Frida documentation provides an explicit fatty-acid synonym table. For example, C4:0, butyric acid and butanoic acid are one straight-chain saturated-fatty-acid identity, mapped to INFOODS F4D0. The same reviewed crosswalk connects other saturated acids and explicitly defined unsaturated isomers to their nutrition expressions. Source chemical identifiers and the original food-mass denominator are retained; positional or cis/trans detail is never removed merely to increase overlap \\cite{frida61}.")
    sections.append("For gadoleic acid, PubChem CID 5282767 links the FooDB InChIKey to CHEBI:32419, permitting a merge with Frida's cis-9-eicosenoic acid. The cis-11 isomer, CHEBI:32425, remains distinct. This illustrates identity-based consolidation instead of merging by the word eicosenoic alone \\cite{gadoleic}.")
    sections.append("The original source category is kept separately from our nutritional role, chemical class and training role. FooDB's Nutrient and Compound tables are provenance categories: a Compound row can represent a vitamin or mineral measurement. Conversely, chemical-class membership does not establish nutritional essentiality. The present classification is an explicit, versioned mapping of exact retained identifiers, not a fuzzy-name classifier or an inheritance of the old source-group label. Former nutritional-role fields remain in the CSV with a legacy prefix; the new classification fields are the current reporting interpretation.")
    sections.append(r"\subsection{Why Some Carbohydrate Expressions Remain Distinct}")
    carb = axes[axes.canonical_name.str.contains("carbohydrate", case=False)]
    sections.append(table(["Unique retained name", "Identity", "Selected supplier(s)"], [[tex(x.canonical_name), identifier(str(x.authority_id)), tex("; ".join(sorted(set(actual.loc[actual.component_concept_id.eq(x.component_concept_id), 'source_key']))))] for x in carb.itertuples()], [0.51, 0.12, 0.30]))
    sections.append("The two former identical display labels included a CoFID axis, which is now outside the dataset. FooDB's generic carbohydrate name does not certify a by-difference calculation. A USDA-coded FooDB Content record is assigned to the documented USDA expression; an unspecified Duke record is not. Total carbohydrate, available carbohydrate and method-unspecified carbohydrate are not interchangeable simply because the short name is similar. FAO/INFOODS distinguishes these quantities and their conversion prerequisites \\cite{infoodscarbohydrate,infoodsconversion}. No new carbohydrate quantity was calculated by assuming absent fibre to be zero.")
    sections.append(r"\subsection{Explicit Macro, Micro and Other Classes}")
    sections.append("The mutually exclusive reporting classes below are a project crosswalk grounded in authoritative definitions, not a taxonomy copied verbatim from one institution. Nutritional role, chemical family, essentiality and statistical prediction eligibility are distinct fields. A small amount in food does not make an analyte a micronutrient; a large number of observations does not make it essential.")
    sections.append(table(
        ["Reporting category", "Axes", "Target*", "Context*"],
        [[tex(x.category_en), str(x.axes), str(x.inherited_maskable_targets), str(x.inherited_context_only)]
         for x in role_counts.itertuples()],
        [0.61, 0.09, 0.11, 0.12],
        "Every retained expression is assigned exactly once. *Eligibility is inherited from the frozen candidate, not newly scientifically approved by this annotation.",
    ))
    sections.append(r"\paragraph{Macronutrients.} The broad USDA grouping includes carbohydrate, fat, protein, dietary fibre and water. We place total fat, total carbohydrate, whole-fibre expressions and water in this reporting layer. Water is non-energy-yielding. Fibre is a method-defined fraction, potentially including lignin, not a single carbohydrate molecule. Protein belongs in this class scientifically but is absent from the retained candidate; it is not added to the count merely because it should be present \cite{class_usda_macro,class_fao_components}.")
    sections.append(r"\paragraph{Micronutrients.} WHO's vitamin/mineral framework supplies the broad definition. Accepted vitamin forms stay within their parent vitamin family; 40 vitamin expressions here do not mean 40 essential vitamins. Major minerals such as calcium and potassium remain in the mineral layer, with a major-mineral subtype, rather than being mixed with protein and fat because of their larger daily requirements. Choline is an essential nutrient reported separately alongside micronutrients, not invented as another B vitamin \cite{class_who_micro,class_nih_choline,class_nasem_water}.")
    sections.append(r"\paragraph{Constituents and related forms.} Fatty acids, protein amino acids, sugars, starch and fibre fractions are subcomponents of macronutrient composition. Essential amino acids and essential fatty acids retain their indispensable status, but do not become vitamins or minerals. Vitamin-related metabolites and precursors receive a separate layer. Only alpha-tocopherol is the vitamin E form recognized to meet human requirements; non-alpha tocopherols and tocotrienols are not counted as additional core micronutrients. Provitamin A carotenoids are distinguished from lutein, zeaxanthin and lycopene, which do not convert to vitamin A \cite{class_nlm_amino,class_nih_omega3,class_nih_E,class_nih_A}.")
    sections.append(r"\paragraph{Other types.} Sterols, flavonoids and other phytochemicals, nitrogenous metabolites, organic acids, contaminants and proximate indicators are separately identified. USDA's flavonoid classes, ChEBI chemical identities, EFSA's biogenic-amine framework and FDA's contaminant categories support these assignments. A chemical classification is not a claim of a health benefit, toxicity at the observed dose or an established dietary requirement \cite{class_usda_flavonoids,class_chebi,class_efsa_amines,class_fda_elements}.")
    sections.append(r"\paragraph{Boundary cases.} Chromium has historical US intake guidance but disputed essentiality; fluoride has a caries-related AI without being an essential nutrient under EFSA's definition. Neither is treated as an undisputed core nutrient. Nickel and toxic-element monitoring expressions are not promoted to essential minerals. Ash, total nitrogen, dry matter and salt expressions are analytical indicators, not four additional nutrient species \cite{class_nih_chromium,class_efsa_fluoride,class_efsa_metals,class_fdc_docs}.")
    sections.append(r"\subsection{What the Classification Does Not Certify}")
    sections.append("Each catalogue entry below states a category, subtype, parent nutrient/family, rationale, essentiality interpretation, references and review limitations. This covers all retained expressions, including entries that should not be ordinary mass inputs. A documented family assignment is not independent expert sign-off of a source measurement, proof of equivalent analytical methods, permission to merge another identity or approval to use an axis as a prediction target.")
    sections.append(table(
        ["Expression", "Classification / data concern", "Required interpretation"],
        [[tex(x.canonical_name), tex(x.issue.replace('_', ' ')), tex(x.recommendation)] for x in findings.itertuples()],
        [0.25, 0.24, 0.44],
        "Semantic findings are preserved rather than hidden by relabelling or silently changing the frozen matrix.",
    ))

    sections.extend([r"\section{Recovery of FooDB Compound IDs}"])
    sections.append(f"The quantitative Content inventory contains {len(orphan):,} absent Compound IDs, including the non-identity placeholder ID 0. Recovery consults original Content names, CompoundSynonym, CompoundExternalDescriptor and the downloaded ChEBI ontology. Each ID has an individual decision row; original records are preserved. {summary['orphan_ids_with_selected_values']} recovered original IDs now support {summary['orphan_selected_values']:,} selected source records. This counts recovered original IDs, not necessarily newly introduced axes: many recovered records enrich an existing magnesium, thiamin or amino-acid axis.")
    sections.append("Online cross-checks confirm magnesium(2+) as CHEBI:18420, thiamine(1+) as CHEBI:18385 and L-valine as CHEBI:16414 \\cite{chebimg,chebithiamin,chebival}. The measured elemental amount is not the mass of an arbitrary magnesium salt. The original USDA component code and Content label determine the nutrition measurement identity. An important counterexample is historical Compound:21595, which contains both USDA 645 (monounsaturated) and 646 (polyunsaturated) expressions. These are separated at record level. Uncoded ambiguous aggregate rows remain excluded.")
    sections.append("Current FooDB redirects are not sufficient identity proof: the former L-valine accession FDB000465 redirects to an unspecified/racemic-valine entry. The archived ChEBI cross-reference and original Content expression are therefore preserved and checked rather than replacing stereochemistry from the redirect alone. Full individual online verification has not been claimed for all recovered IDs; the ledger distinguishes local official-source evidence from the online checks documented here. For a reused historical ID, the recovery report lists every final measurement identity selected from its record-level codes; one old ID is not assumed to mean one molecule.")
    top = orphan[orphan.selected_records.ne("")].copy()
    top["selected_records"] = pd.to_numeric(top.selected_records)
    top = top[top.selected_records.gt(0)].sort_values('selected_records',ascending=False).head(18)
    sections.append(table(["Original Compound ID", "Recovered / measurement name", "Selected records", "Final role"], [[identifier(str(x.source_id)), tex(x.canonical_name), f"{int(x.selected_records):,}", tex(str(x.training_role).replace('_',' '))] for x in top.itertuples()], [0.18,0.42,0.15,0.18]))

    sections.extend([r"\section{Eligibility, Units and Remaining Limits}"])
    sections.append("The intended main modality is g/100 g edible portion on the documented source basis. Exact mass conversions and separate zero/missing flags are stored. However, the new semantic review found that the FooDB Energy expression escaped the unit-based gate; its stored numbers must not be interpreted as grams. A below-detection-limit fatty-acid sum also needs a source-specific censoring review. Consequently, the previous all-mass/no-censoring claim is not established for every retained expression. These records are explicitly flagged in the classification issues ledger; a corrected numerical release, not relabelling alone, is required before training.")
    sections.append("The existing 2\\% source-capability-adjusted coverage screen is retained. A maskable axis additionally requires at least 100 training concepts, 30 frozen validation concepts, an accepted identity and a nondegenerate train-only robust scale. Adequately covered axes without target support remain context-only. Calculated-by-difference expressions remain context-only under the inherited protocol. These are project statistical eligibility decisions, not definitions of nutrient status. The validation-support requirement has not been bypassed by presenting insufficiently supported nutrients as proven targets.")
    sections.append("A chemical dictionary link does not override the source measurement definition. FCDB/Frida parameter 40 measures vitamin B6 vitamers together as a pyridoxine-hydrochloride expression; it is not a pyridoxal-5-phosphate assay. Analogous documented B1, B12, folate and niacin equivalent expressions are retained in the non-main-modality audit instead of being mixed into single-molecule mass labels. These are definition-based exclusions, unrelated to whether the source internally cites another database \\cite{frida61}.")
    sections.append(f"There are {summary['unresolved_conflicts']:,} unresolved within-cell heterogeneity groups before final axis filtering. Their labels are withheld. The source-capability denominator is an operational approximation, not proof that every source measured every chemical class. Internal citations remain published reference evidence and are not certified as independent assays. Expert assessment of high-risk identity, original sample basis, licences and family grouping is still required before a publication-ready benchmark. Current provenance fields do not establish that all manual entries are primary analytical measurements.")
    sections.append("All quantities and pre-normalization profiles are released with provenance. Only train-derived normalization statistics appear in the component registry. No model performance is inferred from changing the number of rows or axes. Previous releases and raw sources are unchanged.")
    sections.append(table(["Final automatic review", "Result"], [
        ["Critical structural/data checks passed", f"{review_summary['critical_checks'] - review_summary['critical_failures']}/{review_summary['critical_checks']}"],
        ["Composition-consistency screening flags", str(review_summary['composition_review_flags'])],
        ["Unresolved within-cell conflict groups", str(review_summary['unresolved_measurement_conflicts'])],
        ["Expert identity review queue items", str(review_summary['expert_identity_review_items'])],
    ], [0.74, 0.19]))
    sections.append("These are the prior structural audit results, not a new semantic certification. The Energy and protein findings demonstrate why passing the automatic gate is insufficient. Composition flags are screening observations, not automatically established errors: totals and component panels require compatible definitions. Identity-queue items also cover source mappings outside the retained matrix.")
    sections.extend([r"\section{Reproducibility Files}",r"\begin{itemize}"])
    for filename in ['food_provenance.csv','unique_composition_registry.csv','composition_source_aliases.csv','composition_classification.csv','composition_classification.md','classification_evidence_and_rules.json','classification_data_issues.csv','selected_value_provenance.csv.gz','foodb_reference_resolution.csv','orphan_compound_review.csv']:
        sections.append(r"\item " + identifier(filename))
    sections.extend([r"\end{itemize}",r"\appendix",r"\clearpage",r"\section{Complete Retained Composition Catalogue}"])
    sections.append("Every retained index appears below, including provisional and inappropriate numerical expressions. Original source categories, source aliases, selected-value suppliers and our classification are shown independently. The chemical-family label is the explicitly reviewed reporting class; unedited ontology memberships remain in the CSV. Category assignment does not assert that all remaining identities are chemically distinct or all source expressions equivalent.")
    category_order = {key: i for i, key in enumerate(specification["categories"])}
    catalogue = axes.assign(classification_order=axes.classification_category.map(category_order))
    for i, axis in enumerate(catalogue.sort_values(['classification_order','canonical_name']).itertuples(),1):
        a = aliases[aliases.component_concept_id.eq(axis.component_concept_id)]
        used = actual[actual.component_concept_id.eq(axis.component_concept_id)]
        sections.append(r"\Needspace{10\baselineskip}\subsection*{" + f"C{i:03d}. " + tex(axis.canonical_name) + "}")
        sections.append(identifier(axis.component_concept_id) + r"\par")
        sections.append(r"\textbf{Our category:} " + tex(axis.classification_category_en) + ". " + r"\textbf{Subtype:} " + tex(axis.classification_subtype) + ".")
        sections.append(r"\textbf{Nutrient/family:} " + tex(axis.parent_nutrient_or_family) + ". " + r"\textbf{Chemical family:} " + tex(axis.reporting_chemical_family) + ".")
        sections.append(r"\textbf{Why this category:} " + tex(axis.classification_reason_en) + r" \cite{" + ",".join("class_" + key for key in axis.classification_reference_keys.split(";")) + "}.")
        sections.append(r"\textbf{Essentiality:} " + tex(axis.essentiality_interpretation))
        sections.append(r"\textbf{Inherited training status:} " + tex(axis.training_role.replace('_',' ')) + f"; train/validation support: {int(axis.train_count):,}/{int(axis.validation_count):,}. Nutritional category does not change this status.")
        authority_id = str(axis.authority_id)
        namespace_prefix = str(axis.authority_namespace) + ":"
        display_id = authority_id if authority_id.startswith(namespace_prefix) else namespace_prefix + authority_id
        definition = str(axis.definition)[:450].rstrip(". ") or 'Stable source identity; detailed analytical definition not supplied'
        sections.append(r"\textbf{Registered identity / source definition:} " + identifier(display_id) + "; " + tex(definition) + ".")
        sections.append(r"\textbf{Review limit:} " + tex(axis.classification_review_status.replace('_',' ')) + ". " + tex(axis.classification_limitation))
        if axis.authority_namespace == "CHEBI":
            sections.append(r"\textbf{Exact chemical record:} \url{https://www.ebi.ac.uk/chebi/" + str(axis.authority_id) + "}.")
        source_rows=[]
        for source, group in a.groupby('source_key'):
            original_names='; '.join(sorted(set(group.original_name)))
            categories='; '.join(sorted(set(group.source_component_group)))
            source_rows.append([tex(SOURCES[source][0]),tex(categories),tex(original_names),f"{len(used[used.source_key.eq(source)]):,}"])
        sections.append(table(['Source','Native table / category','Original name(s)','Selected records'],source_rows,[0.15,0.30,0.36,0.12]))
    sections.append(r"\begin{thebibliography}{99}")
    refs=[
        ('foodbcarb','FooDB. Carbohydrate (FDBN00003), source and content references.','https://foodb.ca/nutrients/FDBN00003'),
        ('infoodsmatch','FAO/INFOODS. Guidelines for Food Matching, version 1.2.','https://www.fao.org/fileadmin/templates/food_composition/documents/Nutrition_assessment/INFOODSGuidelinesforFoodMatching_version_1_2.pdf'),
        ('infoodscarbohydrate','FAO/INFOODS. Guidelines for Checking Food Composition Data, version 1.0.','https://www.fao.org/fileadmin/templates/food_composition/documents/Guidelines_data_checking_02.pdf'),
        ('infoodsconversion','FAO/INFOODS. Guidelines for Converting Units, Denominators and Expressions, version 1.0.','https://www.fao.org/fileadmin/templates/food_composition/documents/1nutrition/Conversion_Guidelines-V1.0.pdf'),
        ('infoods','FAO/INFOODS. Food component identifiers: tagnames.','https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/'),
        ('nih','NIH Office of Dietary Supplements. Nutrient fact sheets.','https://ods.od.nih.gov/factsheets/list-all/'),
        ('nasem','National Academies. Dietary Reference Intakes collection.','https://nap.nationalacademies.org/collection/57/dietary-reference-intakes'),
        ('chebimg','ChEBI. Magnesium(2+), CHEBI:18420.','https://www.ebi.ac.uk/chebi/CHEBI:18420'),
        ('chebithiamin','ChEBI. Thiamine(1+), CHEBI:18385.','https://www.ebi.ac.uk/chebi/CHEBI:18385'),
        ('chebival','ChEBI. L-valine, CHEBI:16414.','https://www.ebi.ac.uk/chebi/CHEBI:16414'),
        ('frida61','DTU National Food Institute. Danish Food Composition Database 6.1. Documentation, sections 6 and 8, and Appendix A.','https://doi.org/10.11583/DTU.32312844.v1'),
        ('gadoleic','NIH PubChem. Gadoleic acid, CID 5282767; structure and ChEBI cross-reference.','https://pubchem.ncbi.nlm.nih.gov/compound/5282767'),
    ]
    for key,title,url in refs:
        sections.append(r"\bibitem{"+key+"}"+tex(title)+r" \url{"+url+"}.")
    for key, (title, url, _) in specification["references"].items():
        sections.append(r"\bibitem{class_"+key+"}"+tex(title)+r" \url{"+url+"}.")
    sections.extend([r"\end{thebibliography}",r"\end{document}"])
    preamble=(ROOT/'reports/scientific_food_composition_v4/provenance_report_template.tex').read_text().split(r'\begin{document}')[0]
    preamble=preamble.replace('v4 candidate','v5.1 candidate').replace('8 September 2026','9 September 2026').replace('scientific\\_food\\_composition\\_v4','scientific\\_food\\_composition\\_v5\\_1')
    preamble=preamble.replace('Sources, Original Categories, and the Basis of Our Classification','FooDB-Centred Sources, Unique Composition Identities, and Classification')
    output.mkdir(parents=True,exist_ok=True)
    target=output/'foodb_centered_dataset_report.tex'
    target.write_text(preamble+r'\begin{document}'+'\n'+r'\maketitle'+'\n'+'\n\n'.join(sections))
    after_hashes = {path.name: sha256_file(path) for path in protected}
    assert before_hashes == after_hashes, "Report generation changed frozen dataset inputs."
    write_json({
        'report_axes': len(axes), 'source_food_observations': len(foods),
        'selected_records': len(actual), 'selected_cells': len(profiles),
        'unique_names': axes.canonical_name.str.casefold().is_unique,
        'classification_coverage': int(axes.classification_category.notna().sum()),
        'classification_reference_complete': bool(axes.classification_evidence_urls.ne("").all()),
        'classification_rule_count': len(specification["rules"]),
        'classification_specification_sha256': sha256_file(specification_path),
        'classification_data_issues': len(findings),
        'frozen_input_hashes': after_hashes, 'numerical_inputs_unchanged': True,
        'classification_expert_certified': False,
    }, output/'report_checks.json')
    (output / "README.md").write_text(
        "# FooDB-centred candidate dataset\n\n"
        f"Version: `{VERSION}`. Main report: `foodb_centered_dataset_report.tex`.\n\n"
        f"{summary['nonempty_food_concepts']:,} non-empty foods; {len(axes)} retained composition expressions; "
        f"{n:,} observed cells in the nominal mass matrix (semantic exceptions flagged below). "
        f"FooDB-linked foods: {summary['foodb_anchor_nonempty_foods']:,}.\n\n"
        "AFCD, CoFID and Norway are excluded only as direct inputs. Internal citations in retained databases "
        "keep their published values and references. Food/component identity and unit checks still apply.\n\n"
        "The candidate is not yet a domain-expert-approved benchmark. Frozen validation identities remain "
        f"1,426, but only {summary['validation_nonempty_foods']:,} have retained values. No model was trained.\n\n"
        "## Data and provenance\n\n"
        f"Dataset: `data/processed/{VERSION}/release/` from the repository root.\n"
        "`food_provenance.csv` contains source food categories and IDs. "
        "`composition_source_aliases.csv` records native names/categories and canonical IDs. "
        "`unique_composition_registry.csv` has the retained axes. "
        "`selected_value_provenance.csv.gz` traces selected cells; "
        "`orphan_compound_review.csv` reports recovered Compound IDs and their actual final identities.\n\n"
        "## Reproduce in Colab\n\n```bash\n"
        "pip install -r requirements-colab.txt\n"
        "python scripts/review_foodb_centered_v5.py\n"
        "python scripts/build_scientific_food_composition_v5.py\n"
        f"PYTHONPATH=src python -m foodcomp.audit --root . --release-dir data/processed/{VERSION}/release --audit-dir data/audits/{VERSION}/dataset\n"
        "python scripts/report_v5_foodb_centered.py\n"
        "PYTHONPATH=src python -m unittest discover -s tests\n```\n\n"
        "The build requires the archived raw sources, ontology snapshots and v3/v4 source staging already "
        "documented in the repository. It refuses to overwrite a completed candidate. "
        "Compile the report with XeLaTeX twice; this is separate from dataset construction.\n"
        "\n## Explicit composition classification\n\n"
        "Every retained axis has a macro/micro/other category, subtype, nutritional parent, "
        "bilingual rationale, authoritative references and limitations in "
        "composition_classification.csv and composition_classification.md. "
        "The full report includes the same rationale for all 293 axes. "
        "The source dictionary's category remains separate; earlier nutritional-role fields "
        "are named legacy_* in the enriched registry.\n\n"
        "This documentation-only update does not alter numerical values, splits or maskability. "
        "To refresh only the report, run python scripts/report_v5_foodb_centered.py; "
        "a full dataset rebuild is not needed. Classification is a project crosswalk grounded "
        "in official definitions, not individual expert certification.\n\n"
        "**Semantic release issues:** Energy is still incorrectly retained in the mass-labelled "
        "matrix, while nitrogen-derived total protein is excluded under label_expression. "
        "The below-LOD aggregate and several source-expression identities also need review. "
        "These are recorded in classification_data_issues.csv; the inherited automatic pass "
        "is not sufficient to call the candidate nutritionally complete or training-ready.\n"
    )
    print(target)


if __name__=='__main__':
    main()
