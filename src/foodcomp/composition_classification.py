"""Evidence-linked reporting roles, independent of frozen numerical eligibility."""

import json
from pathlib import Path

import pandas as pd


def load_classification(path: Path) -> dict:
    specification = json.loads(path.read_text(encoding="utf-8"))
    lookup = {}
    rule_ids = set()
    for rule in specification["rules"]:
        if rule["id"] in rule_ids:
            raise ValueError(f"Duplicate classification rule: {rule['id']}")
        rule_ids.add(rule["id"])
        if rule["category"] not in specification["categories"]:
            raise ValueError(f"Unknown category in rule {rule['id']}")
        for ref in rule["refs"]:
            if ref not in specification["references"]:
                raise ValueError(f"Missing classification reference: {ref}")
        for field in ("why_en", "why_zh", "essentiality", "chemical_family"):
            if not rule.get(field):
                raise ValueError(f"Empty {field} in rule {rule['id']}")
        for namespace, identifiers in rule["members"].items():
            for identifier in identifiers.split():
                key = (namespace, identifier)
                if key in lookup:
                    raise ValueError(f"Ambiguous classification: {key}")
                lookup[key] = rule
    for key, references in specification.get("identity_references", {}).items():
        for reference in references:
            if reference not in specification["references"]:
                raise ValueError(f"Missing reference for {key}: {reference}")
    specification["lookup"] = lookup
    return specification


def classify_components(axes: pd.DataFrame, specification: dict) -> pd.DataFrame:
    """Annotate every retained ID; never infer a role by a fuzzy display name."""
    if not axes["component_concept_id"].is_unique:
        raise ValueError("Classification requires unique component_concept_id.")
    rows = []
    for axis in axes.to_dict("records"):
        key = (str(axis["authority_namespace"]), str(axis["authority_id"]))
        if key not in specification["lookup"]:
            raise ValueError(f"Unreviewed classification identity: {key}")
        rule = specification["lookup"][key]
        label_en, label_zh, macro_micro = specification["categories"][rule["category"]]
        reference_keys = list(dict.fromkeys(
            rule["refs"] + specification.get("identity_references", {}).get(":".join(key), [])
        ))
        sources = [specification["references"][ref][1] for ref in reference_keys]
        if key[0] == "CHEBI":
            sources.append("https://www.ebi.ac.uk/chebi/" + key[1])
        status = "evidence_supported_project_assignment"
        review = "Family annotation does not certify every source measurement."
        action = "Reporting annotation only; frozen numerical eligibility unchanged."
        if rule["category"] == "element_boundary":
            status = "qualified_nutritional_status"
            review = rule["essentiality"]
        if rule["category"] == "unresolved":
            status = "identity_unresolved"
            review = "Do not claim a resolved nutritional identity or pool by name."
            action = "Resolve source identity before future numerical use."
        if rule["category"] == "non_mass_or_censored":
            status = "numerical_use_requires_correction"
            review = "Current inclusion is not approved by this classification."
            action = "Hold numerical use; correct modality/censoring in a new dataset version."
        if rule["id"] == "B1_salt":
            status = "measurement_expression_review"
            review = "Verify whether the published mass denotes salt or active thiamin."
        if rule["id"] == "cryptoxanthin":
            status = "isomer_activity_review"
            review = "Provitamin A assignment requires beta-isomer evidence."
        if rule["id"] == "salt":
            status = "measurement_expression_review"
            review = "Verify NaCl mass versus sodium-based salt equivalent."
        if key[0] == "SOURCE" and rule["category"] not in {
            "non_mass_or_censored", "unresolved", "proximate"
        }:
            status = "source_label_family_assignment"
            review = "Broad family is supported; exact source expression still needs review."
        if key[0] == "INCHIKEY" and rule["category"] in {
            "micro_vitamin", "carbohydrate_fraction"
        }:
            status = "source_expression_and_structure_review"
            review = "Check source measurement versus dictionary structure; no new identity merge."
        parent = rule["parent"]
        if parent == "element-specific mineral":
            parent = "iodine" if key[1] == "ID" else axis["canonical_name"].split(",")[0]
        rows.append({
            "component_concept_id": axis["component_concept_id"],
            "classification_version": specification["version"],
            "classification_category": rule["category"],
            "classification_category_en": label_en,
            "classification_category_zh": label_zh,
            "macro_micro_class": macro_micro,
            "classification_subtype": rule["subtype"],
            "parent_nutrient_or_family": parent,
            "reporting_chemical_family": rule["chemical_family"],
            "classification_rule_id": rule["id"],
            "classification_reason_en": axis["canonical_name"] + ": " + rule["why_en"],
            "classification_reason_zh": axis["canonical_name"] + "：" + rule["why_zh"],
            "essentiality_interpretation": rule["essentiality"],
            "classification_reference_keys": ";".join(reference_keys),
            "classification_evidence_urls": " | ".join(dict.fromkeys(sources)),
            "classification_review_status": status,
            "classification_limitation": review,
            "classification_recommended_action": action,
            "classification_is_project_mapping": True,
            "classification_expert_signed_off": False,
        })
    annotations = pd.DataFrame(rows)
    # Old labels remain available as historical metadata, not current role claims.
    result = axes.rename(columns={
        column: "legacy_" + column for column in axes.columns
        if column.startswith("nutritional_role")
    }).merge(annotations, on="component_concept_id", validate="one_to_one")
    if len(result) != len(axes):
        raise AssertionError("Classification lost one or more retained axes.")
    return result


def classification_counts(axes: pd.DataFrame, specification: dict) -> pd.DataFrame:
    counts = axes.groupby(["classification_category", "training_role"]).size().unstack(fill_value=0)
    rows = []
    for category, (english, chinese, macro_micro) in specification["categories"].items():
        if category not in counts.index:
            continue
        row = counts.loc[category]
        rows.append({
            "category": category, "category_en": english, "category_zh": chinese,
            "macro_micro_class": macro_micro, "axes": int(row.sum()),
            "inherited_maskable_targets": int(row.get("maskable_target", 0)),
            "inherited_context_only": int(row.get("context_only", 0)),
        })
    return pd.DataFrame(rows)


def classification_findings(all_axes: pd.DataFrame, retained: pd.DataFrame) -> pd.DataFrame:
    rows = []
    retained_ids = set(retained["component_concept_id"])
    for axis in all_axes.to_dict("records"):
        if axis["authority_id"] == "PROCNT" and axis["component_concept_id"] not in retained_ids:
            rows.append({
                "component_concept_id": axis["component_concept_id"],
                "canonical_name": axis["canonical_name"],
                "issue": "Established macronutrient absent from retained matrix",
                "evidence": (
                    f"Registry expression_variant={axis['expression_variant']}; "
                    f"exclusion={axis['training_exclusion_reason']}; "
                    f"train/validation support={axis['train_count']}/{axis['validation_count']}."
                ),
                "recommendation": (
                    "Review the inherited label-expression exclusion at source-record level. "
                    "Nitrogen-derived protein is still a mass-based macronutrient; total nitrogen "
                    "cannot substitute for protein. Do not invent a missing protein axis in the report."
                ),
            })
    for axis in retained.to_dict("records"):
        if axis["classification_review_status"] in {
            "identity_unresolved", "numerical_use_requires_correction",
            "measurement_expression_review", "isomer_activity_review",
        }:
            rows.append({
                "component_concept_id": axis["component_concept_id"],
                "canonical_name": axis["canonical_name"],
                "issue": axis["classification_review_status"],
                "evidence": axis["classification_reason_en"],
                "recommendation": axis["classification_recommended_action"] + " "
                + axis["classification_limitation"],
            })
    return pd.DataFrame(rows)


def classification_markdown(axes: pd.DataFrame, counts: pd.DataFrame, spec: dict) -> str:
    tick = chr(96)
    lines = [
        "# 当前 composition 分类与依据",
        "",
        f"分类版本：{tick}{spec['version']}{tick}。逐项覆盖 {len(axes)} 个保留轴。",
        "",
        "本文件是文档分类层，不修改现有数值、成分 ID、split 或 target/context 资格。"
        "旧 nutritional_role 仅保留为 legacy 字段，不代表本次认可的分类。",
        "",
        "## 分类原则",
        "",
        "- macro：总脂肪、总碳水、膳食纤维；水按 USDA 广义宏量框架纳入，但标明不供能。",
        "- micro：公认维生素及供给形式、公认膳食矿物质。胆碱作为其他必需营养素单列。",
        "- 营养子成分：脂肪酸、氨基酸、糖和纤维分级，不因剂量小就叫 micro。",
        "- 维生素相关物：前体、部分衍生物和代谢物，不能都计作额外的必需维生素。",
        "- 其他类按结构或分析目的细分；有争议、身份不明、非质量表达均单列。",
        "- 这些互斥的报告分组是依据权威定义形成的项目映射，不声称是某机构原封不动的分类表。",
        "- 同一维生素的多个化学形式仍对应同一营养家族；轴数不是必需营养素的种类数。",
        "",
        "## 数量",
        "",
        "| 类别 | 轴数 | 继承的预测资格 | 继承的仅上下文资格 |",
        "|---|---:|---:|---:|",
    ]
    for row in counts.to_dict("records"):
        lines.append(
            f"| {row['category_zh']} | {row['axes']} | "
            f"{row['inherited_maskable_targets']} | {row['inherited_context_only']} |"
        )
    lines += [
        "", "## 数据问题必须与分类分开", "",
        "Energy 仍在旧的质量标签矩阵中，不能解释为 g/100g；"
        "总蛋白质 PROCNT 有观测支持却被 label_expression 规则排除。"
        "另外检测限汇总、盐表达和部分结构/来源名不一致仍需处理。"
        "本次明确记录这些问题，不静默修改冻结矩阵；因此不能把分类覆盖完整称为数据集已经营养完整或可以直接训练。",
        "", "## 逐项分类", "",
    ]
    for category, labels in spec["categories"].items():
        subset = axes[axes.classification_category.eq(category)].sort_values("canonical_name")
        if subset.empty:
            continue
        lines += ["### " + labels[1], ""]
        for row in subset.to_dict("records"):
            links = [
                f"[{ref}]({spec['references'][ref][1]})"
                for ref in row["classification_reference_keys"].split(";")
            ]
            lines += [
                f"#### {row['canonical_name']}",
                "",
                f"- ID：{tick}{row['component_concept_id']}{tick}；分类：{row['classification_category_zh']}；"
                f"细类：{row['classification_subtype']}。",
                f"- 理由：{row['classification_reason_zh']}",
                f"- 对应营养/化学家族：{row['parent_nutrient_or_family']}。",
                f"- 原来源：{row.get('original_source_categories', '')}。",
                f"- 原名称：{row.get('original_names_by_source', '')}。",
                f"- 依据：{'，'.join(links)}。",
                f"- 证据边界：{row['classification_review_status']}；"
                f"{row['classification_limitation']}。",
                "",
            ]
    return "\n".join(lines) + "\n"
