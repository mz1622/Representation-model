"""Generate bilingual scientific reports, data cards and release manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .constants import AS_OF_DATE, DATASET_VERSION, MAIN_BASIS
from .util import read_component_csv, sha256_file, write_csv, write_json


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def _latex(value: Any) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in str(value))


def _source_overview(release_dir: Path, scoping_dir: Path, dataset_audit_dir: Path) -> pd.DataFrame:
    foods = pd.read_csv(release_dir / "food_observation.csv.gz", low_memory=False)
    components = read_component_csv(release_dir / "component_observation.csv.gz")
    disposition = pd.read_csv(dataset_audit_dir / "source_measurement_disposition.csv", low_memory=False)
    registry = pd.read_csv(scoping_dir / "source_registry.csv", low_memory=False)
    food_count = foods.groupby("source_key")["food_observation_id"].nunique()
    component_count = components.groupby("source_key")["component_observation_id"].nunique()
    total = disposition.groupby("source_key")["measurement_count"].sum()
    eligible = disposition[disposition["main_value_eligible"].astype(str).str.casefold().eq("true")].groupby("source_key")["measurement_count"].sum()
    available = registry[registry["source_key"].fillna("").ne("")].drop_duplicates("source_key").set_index("source_key")
    keys = sorted(set(food_count.index) | set(component_count.index) | set(total.index))
    rows = []
    for key in keys:
        source = available.loc[key] if key in available.index else pd.Series(dtype=object)
        rows.append({
            "source_key": key,
            "database_name": source.get("database_name", key),
            "version": source.get("version", ""),
            "layer": source.get("layer", ""),
            "food_observations": int(food_count.get(key, 0)),
            "component_observations": int(component_count.get(key, 0)),
            "staged_measurements": int(total.get(key, 0)),
            "main_eligible_measurements": int(eligible.get(key, 0)),
            "license": source.get("license", ""),
            "official_url": source.get("official_url", ""),
            "full_fao_evaluation": source.get("official_full_evaluation_status", "pending_manual_expert_assessment"),
        })
    return pd.DataFrame(rows)


def _role_overview(components: pd.DataFrame) -> pd.DataFrame:
    retained = components[components["training_role"].ne("excluded")]
    return retained.groupby(["nutritional_role", "training_role"], dropna=False).size().reset_index(name="component_count")


def _write_descriptive_tables(release_dir: Path, dataset_audit_dir: Path, report_dir: Path, components: pd.DataFrame) -> None:
    food_observations = pd.read_csv(release_dir / "food_observation.csv.gz", low_memory=False)
    food_concepts = pd.read_csv(release_dir / "food_concept.csv.gz", low_memory=False)
    profiles = pd.read_csv(release_dir / "canonical_profile.csv.gz", low_memory=False)
    partitions = pd.read_csv(release_dir / "ml_partition.csv", low_memory=False)
    disposition = pd.read_csv(dataset_audit_dir / "source_measurement_disposition.csv", low_memory=False)

    group_distribution = food_observations.assign(
        food_group=food_observations["food_group"].fillna("unreported").replace("", "unreported")
    ).groupby(["source_key", "food_group"])["food_observation_id"].nunique().reset_index(name="food_observation_count")
    write_csv(group_distribution, report_dir / "food_group_distribution_by_source.csv.gz")

    authority = components.groupby(
        ["authority_namespace", "identity_status", "training_role"], dropna=False
    ).size().reset_index(name="component_count")
    write_csv(authority, report_dir / "component_authority_overview.csv")

    chemical = components[components["training_role"].ne("excluded")].groupby(
        ["chemical_class", "nutritional_role", "training_role"], dropna=False
    ).size().reset_index(name="component_count")
    write_csv(chemical, report_dir / "retained_chemical_class_overview.csv.gz")

    component_exclusions = components[components["training_role"].eq("excluded")].groupby(
        "training_exclusion_reason", dropna=False
    ).size().reset_index(name="component_count")
    write_csv(component_exclusions, report_dir / "component_exclusion_summary.csv")

    measurement_summary = disposition.groupby(
        ["source_key", "conversion_status", "quality_tier", "main_value_eligible", "exclusion_reason"], dropna=False
    )["measurement_count"].sum().reset_index()
    write_csv(measurement_summary, report_dir / "measurement_disposition_summary.csv.gz")

    accepted = profiles[profiles["aggregation_status"].eq("accepted")]
    observed_by_partition = accepted.groupby("partition").agg(
        accepted_profiles=("component_concept_id", "size"),
        represented_foods=("food_concept_id", "nunique"),
        represented_components=("component_concept_id", "nunique"),
    ).reset_index()
    write_csv(observed_by_partition, report_dir / "canonical_profile_partition_overview.csv")

    partition_overview = partitions.groupby(["partition", "validation_panel"], dropna=False).agg(
        food_concepts=("food_concept_id", "nunique"),
        family_blocks=("family_cluster_id", "nunique"),
        reconstruction_eligible=("reconstruction_task_eligible", lambda x: x.astype(str).str.casefold().eq("true").sum()),
        text_eligible=("text_task_eligible", lambda x: x.astype(str).str.casefold().eq("true").sum()),
    ).reset_index()
    write_csv(partition_overview, report_dir / "partition_and_task_overview.csv")

    concept_sources = food_concepts.assign(source_combination=food_concepts["source_keys"].fillna("unknown")).groupby(
        "source_combination"
    ).size().reset_index(name="food_concept_count")
    write_csv(concept_sources, report_dir / "food_concept_source_overlap.csv")


def _table_rows(frame: pd.DataFrame, columns: list[str]) -> str:
    return "\n".join(" & ".join(_latex(row[column]) for column in columns) + r" \\" for _, row in frame.iterrows())


def _english_tex(summary: dict[str, Any], audit: dict[str, Any], sources: pd.DataFrame, roles: pd.DataFrame, prisma: dict[str, Any]) -> str:
    source_rows = _table_rows(sources, ["source_key", "food_observations", "main_eligible_measurements", "layer"])
    role_rows = _table_rows(roles, ["nutritional_role", "training_role", "component_count"])
    return rf"""\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,longtable,array,hyperref,graphicx}}
\title{{Scientific Food Composition Dataset and Benchmark: Candidate Build}}
\date{{Build date: {AS_OF_DATE}}}
\begin{{document}}
\maketitle

\section{{Release Status}}
This document describes \texttt{{{_latex(DATASET_VERSION)}}}. It is a candidate scientific build, not a signed final public release. The automated release gate passed {audit.get('critical_checks', 0)} critical checks with {audit.get('critical_failures', 0)} failures. Domain-expert identity review, two-reviewer literature screening, complete FAO/INFOODS source evaluation, and licence confirmation remain release gates. No model result is used to change this dataset after the locked validation is defined.

\section{{Evidence and Standards}}
The protocol follows PRISMA-ScR and PRISMA-S for a reproducible scoping review. Searches of PubMed, Crossref and OpenAlex retrieved {prisma.get('retrieved_records', 0):,} records and produced {prisma.get('unique_candidates', 0):,} deduplicated candidates. These records remain candidates until dual human screening. The source registry begins with the 101 FCDBs reported in the 2025 global landscape review and records local acquisition decisions separately.

There is no claim that one universal food-composition standard exists. Compatible authoritative systems are assigned distinct responsibilities: FAO/INFOODS for database evaluation, food matching, component identifiers, expressions and checks; EuroFIR QE-SCIREP for value-level evidence; FoodOn, LanguaL and FoodEx2 for food identity and hierarchy; INFOODS, CDNO, ChEBI and LIPID MAPS for component identity and classification; and FAIR for provenance and reproducible release.

\section{{Source Layers}}
\begin{{longtable}}{{p{{0.20\linewidth}}rrp{{0.31\linewidth}}}}
\toprule Source & Foods & Main values & Build layer \\ \midrule
{source_rows}
\bottomrule
\end{{longtable}}
Primary-reference values are admitted only after value-level unit, independence and quality gates. USDA Foundation Foods is reserved as a locked source-holdout panel. FNDDS is an auxiliary calculated-dish layer. FooDB is retained as a specialist and provenance archive; values copied from USDA, Frida/DTU or specialist databases are not counted as independent evidence.

\section{{Canonical Data Model}}
The release contains source registry, food observation, food concept, component observation, component concept, measurement, and canonical profile tables. Original names, identifiers, values, ranges, units, citations and source lineage are permanent fields. Canonical concepts never replace source records. Every canonical value records the contributing measurement IDs and source keys.

\section{{Food Identity}}
A food concept is defined by its name together with scientific taxon, anatomical part, maturity, processing, cooking, preservation, physical state, packing medium, geography, cultivar and recipe status when available. Exact or similar names create candidates only. Explicitly conflicting facets prevent a merge. Shared official lineage can merge records only when their explicit facets are compatible. Exact concepts and shared formal lineages cannot cross a split. Parent-child relations are stored in a DAG and never authorize numerical inheritance.

Supplements, inedible/refuse-only records, and identified label-only branded records are excluded from the main ML corpus. Foods need at least one accepted target for the text task and at least two non-equivalent component families for reconstruction.

\section{{Component Identity and Values}}
The sole primary numerical modality is \texttt{{{_latex(MAIN_BASIS)}}}. Grams, milligrams and micrograms are converted exactly to grams. Energy, IU, RAE, DFE, NE, alpha-TE, molar concentration, relative fatty-acid percentages, dry-weight values without moisture, volume values without density, and serving values without portion mass remain traceable non-main records and are not regression targets. Missing, explicit zero, trace, below-LOD/LOQ, and range-only states are distinct; censored and range-only records are never ordinary point labels.

INFOODS identifiers are accepted only when found in the archived official FAO lists. ChEBI, InChIKey, EuroFIR/EFSA and stable USDA/CNF nutrient numbers provide additional identity keys. Fuzzy or embedding matches never merge components automatically. Independent compatible measurements use the best available value-level quality tier. Log-scale random effects are used when standard errors permit; otherwise a sample-size-weighted log median is used. Prespecified high heterogeneity is withheld for expert adjudication.

\section{{Classification and Filtering}}
Chemical class, nutritional role, measurement modality and training role are separate fields. Macronutrient, vitamin and mineral roles are linked to FAO/INFOODS and National Academies/NIH terminology; chemical classes are inherited from ChEBI, CDNO or LIPID MAPS where a formal link exists. Name-rule classifications are visibly marked for expert review rather than presented as authority assertions.

\begin{{longtable}}{{p{{0.44\linewidth}}p{{0.28\linewidth}}r}}
\toprule Nutritional role & Training role & Axes \\ \midrule
{role_rows}
\bottomrule
\end{{longtable}}
Capability-adjusted coverage below 2\% is excluded, with 1\%, 2\% and 5\% sensitivity tables supplied. A maskable target additionally requires at least 100 training concepts, at least 30 locked-validation concepts, a stable defined identity, a compatible mass expression, and a non-degenerate train-only distribution. Other sufficiently supported identities are context-only.

\section{{Partitions and Benchmark}}
The ML release contains {summary.get('train_food_concepts', 0):,} train concepts and {summary.get('validation_food_concepts', 0):,} locked validation concepts; no test partition is produced. Train contains five deterministic grouped-CV folds. Validation contains {summary.get('family_holdout_food_concepts', 0):,} unseen-family concepts and {summary.get('source_holdout_food_concepts', 0):,} USDA Foundation source-holdout concepts. The family panel is fully family-disjoint. The source panel blocks exact concepts and shared formal lineages, while allowing {summary.get('source_family_overlap_train_food_concepts', 0):,} non-duplicate foods from the same broad families to remain in train; it therefore measures cross-source transfer within represented families rather than unseen-family generalization.

The co-primary tasks are (1) unseen-family component reconstruction with 50\% observed target axes visible, supplemented by 10\%, 30\% and 70\% curves, and (2) composition prediction from canonical text only. Entire component families are masked together to prevent algebraic leakage. Every method receives the same fixed masks and information. Preregistered baselines are per-axis train mean, per-axis train median, food-group mean, text-embedding kNN, random forest and XGBoost.

The primary metric is an axis-equal MAE after train-only median/IQR-or-MAD robust standardization of \(\log(1+x)\), where \(x\) is g/100 g. MSE, RMSE, raw-unit MAE/RMSE, \(R^2\), Spearman correlation, range, SD, IQR, sample size and 95\% food-family cluster-bootstrap confidence intervals are also reported. The two primary tasks use Holm correction. Locked validation requires explicit version confirmation and may not be used to revise this version.

\section{{Quality Assurance and Open Gates}}
The machine audit checks units and physical bounds, zero/missing/censoring semantics, concept-key uniqueness, facet compatibility, family and lineage isolation, Foundation-only source-holdout labels, target support and train-only scales. FAO/INFOODS plausibility checks flag records for review rather than silently deleting them. Current open queues include {audit.get('unresolved_measurement_conflicts', 0):,} heterogeneous food-component conflicts and {audit.get('expert_identity_review_items', 0):,} component identity items, plus the separate food-name review queue.

The release remains a candidate until a food-composition expert adjudicates all high-risk items and a deterministic 10\% sample of high-confidence mappings, two reviewers complete the scoping review and FAO/INFOODS evaluations, and licences are confirmed. Redistributable values may then be published directly; restricted sources must be reconstructed from download scripts, hashes and mappings.

\section{{Reproducibility}}
All split assignments, IDs and masks are deterministic. Raw files are immutable and hashed. Train-only statistics are stored beside the component registry. Machine-readable ledgers document source decisions, name evidence, merge candidates, exclusions, conflicts, coverage sensitivity, masks and audit checks.

\end{{document}}
"""


def _chinese_tex(summary: dict[str, Any], audit: dict[str, Any], sources: pd.DataFrame, roles: pd.DataFrame, prisma: dict[str, Any]) -> str:
    source_rows = _table_rows(sources, ["source_key", "food_observations", "main_eligible_measurements", "layer"])
    role_rows = _table_rows(roles, ["nutritional_role", "training_role", "component_count"])
    return rf"""\documentclass[11pt]{{ctexart}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs,longtable,array,hyperref,graphicx}}
\title{{科研级食品成分数据集与 Benchmark：候选版本}}
\date{{构建日期：{AS_OF_DATE}}}
\begin{{document}}
\maketitle

\section{{版本状态}}
本文记录 \texttt{{{_latex(DATASET_VERSION)}}}。它是科研候选版本，不是已经签字确认的公开最终版。机器发布门共运行 {audit.get('critical_checks', 0)} 项关键检查，失败 {audit.get('critical_failures', 0)} 项。食品营养专家审核、双人文献筛选、完整 FAO/INFOODS 数据库评分和许可复核仍是正式发布前置条件。锁定 validation 后，不能根据模型结果反向修改本版本数据。

\section{{证据与规范}}
Scoping review 按 PRISMA-ScR 和 PRISMA-S 记录。PubMed、Crossref 和 OpenAlex 共返回 {prisma.get('retrieved_records', 0):,} 条记录，标识符和标题去重后有 {prisma.get('unique_candidates', 0):,} 条候选；它们在完成两名审核者筛选前不算正式纳入证据。Source registry 以 2025 年全球综述中的 101 个 FCDB 为起点，另行记录本项目的下载和纳入决定。

本项目不声称存在一个能够解决全部问题的唯一全球标准，而是组合目前最权威且职责相容的体系：FAO/INFOODS 负责数据库评价、食品匹配、成分标识、表达换算和数据检查；EuroFIR QE-SCIREP 负责 value-level 质量维度；FoodOn、LanguaL、FoodEx2 负责食品身份和层级；INFOODS、CDNO、ChEBI、LIPID MAPS 负责成分身份与化学分类；FAIR 负责 provenance 和可复现发布。

\section{{数据源分层}}
\begin{{longtable}}{{p{{0.20\linewidth}}rrp{{0.31\linewidth}}}}
\toprule 来源 & 食品记录 & 主模态数值 & 构建层 \\ \midrule
{source_rows}
\bottomrule
\end{{longtable}}
Primary-reference 数据只有通过单位、独立性和 value-level 质量门才进入主语料。USDA Foundation Foods 全部保留为锁定 source holdout。FNDDS 属于计算菜肴辅助层。FooDB 属于 specialist/provenance 档案层；其转载自 USDA、Frida/DTU 或专业数据库的值不算独立证据。

\section{{统一数据模型}}
公开 schema 包括 source registry、food observation、food concept、component observation、component concept、measurement 和 canonical profile。原始名称、ID、数值、范围、单位、引用与来源谱系永久保留；规范概念不会覆盖原始记录。每个规范值均能回溯至 measurement ID 和来源。

\section{{食品身份}}
食品身份由名称以及物种、部位、成熟度、加工、烹饪、保存、物理状态、包装介质、地域、品种和配方状态共同定义。名称相同或相似只产生候选；显式 facet 冲突禁止合并。共享正式 lineage 也只有在 facet 相容时才允许合并。相同 concept 与共享正式 lineage 的记录不得跨 split。父子食品只通过 DAG 组织，数值绝不沿父子边继承。

补充剂、不可食/废弃部位和已识别的纯标签品牌记录不进入主 ML 语料。文本任务要求至少一个合格目标；遮蔽重建还要求至少两个互不等价的 component family。

\section{{成分身份与数值}}
唯一主数值模态为 \texttt{{{_latex(MAIN_BASIS)}}}。g、mg、微克按精确质量比例换成 g。能量、IU、RAE、DFE、NE、alpha-TE、摩尔浓度、脂肪酸相对百分比、无水分值支持的干重、无密度支持的体积值和无份量质量的每份值保留 provenance，但不作主回归目标。Missing、显式零、trace、低于 LOD/LOQ 和只有范围的值分别保存；censored 和 range-only 不作为普通点标签。

INFOODS ID 必须在保存的 FAO 官方清单中出现才能使用。ChEBI、InChIKey、EuroFIR/EFSA 和稳定 USDA/CNF nutrient number 提供其他身份键。Fuzzy/embedding 只能提出候选，不能自动合并。相容的独立测量先按 value-level 质量层筛选；有标准误时在 log scale 做 random-effects 汇总，否则使用按样本数加权的 log median。达到预设高异质性门槛的值不发布 canonical label，进入专家队列。

\section{{分类与筛选}}
数据分别保存 chemical class、nutritional role、measurement modality 和 training role。宏量营养、维生素和矿物质角色连接 FAO/INFOODS 与 National Academies/NIH 术语；有正式链接时，化学分类继承 ChEBI、CDNO 或 LIPID MAPS。仅由名称规则得到的分类明确标记为待专家审核，不冒充权威结论。

\begin{{longtable}}{{p{{0.44\linewidth}}p{{0.28\linewidth}}r}}
\toprule 营养角色 & 训练角色 & 轴数 \\ \midrule
{role_rows}
\bottomrule
\end{{longtable}}
按具备报告能力的数据源计算覆盖率，低于 2\% 的轴排除，同时提供 1\%、2\%、5\% 敏感性分析。Maskable target 还必须至少覆盖 100 个 train concept 和 30 个锁定 validation concept，具有稳定且有定义的身份、相容质量表达和非退化的 train-only 分布。其余有支持的成分只能作 context-only。

\section{{切分与 Benchmark}}
ML 候选版包含 {summary.get('train_food_concepts', 0):,} 个 train concept 和 {summary.get('validation_food_concepts', 0):,} 个锁定 validation concept，不生成 test。Train 内保存 5 个确定性 grouped-CV folds。Validation 由 {summary.get('family_holdout_food_concepts', 0):,} 个未见家族 concept 和 {summary.get('source_holdout_food_concepts', 0):,} 个 USDA Foundation source-holdout concept 组成。Family panel 与 train 完全按家族隔离；source panel 阻断相同 concept 和共享正式 lineage，但允许 {summary.get('source_family_overlap_train_food_concepts', 0):,} 个同属宽泛家族、却不是重复记录的食品进入 train。因此前者评价未见家族泛化，后者评价已见家族中的跨来源迁移。

两个 co-primary tasks 为：（1）未见食品家族上保留 50\% 已观测 target 的重建，并补充 10\%、30\%、70\% 可见率曲线；（2）只给规范文本的 cold start。遮蔽以 component family 联动，避免总量和子项之间的代数泄漏。所有方法使用同一固定 mask 和相同可见信息。Baseline 为逐轴 train mean、train median、food-group mean、text embedding kNN、RF 和 XGBoost。

首要指标是对每个轴等权计算的 MAE：先对 g/100 g 做 \(\log(1+x)\)，再使用仅由 train 拟合的 median 与 IQR/MAD robust scale 标准化。同时报告 MSE、RMSE、原始 g/100 g MAE/RMSE、\(R^2\)、Spearman、范围、SD、IQR、样本量和 95\% food-family cluster bootstrap CI；两个主任务使用 Holm 校正。打开锁定 validation 必须显式确认数据版本，开封后任何数据修复都必须升版本并重跑全部方法。

\section{{质量检查与未完成门槛}}
机器检查覆盖单位和物理范围、zero/missing/censored 语义、concept key 唯一性、facet 相容性、family/lineage 隔离、Foundation-only holdout label、target 支持度及 train-only scale。FAO/INFOODS 组成合理性检查只生成审核 flag，不静默删除。当前尚有 {audit.get('unresolved_measurement_conflicts', 0):,} 个高异质性 food-component 冲突、{audit.get('expert_identity_review_items', 0):,} 个 component identity 审核项，以及独立的食品名称审核队列。

正式发布前，食品成分专家必须审核全部高风险项和确定性抽取的 10\% 高置信度样本，两名审核者必须完成文献筛选和 FAO/INFOODS 评价，并完成许可确认。可再分发数值可直接发布；受限来源只发布版本、下载步骤、哈希、映射和构建代码。

\section{{可复现性}}
所有 ID、split 和 mask 均为确定性生成。Raw files 保持不可变并记录哈希。Train-only 统计量随 component registry 保存。机器可读 ledger 覆盖来源决定、名称证据、合并候选、排除、冲突、覆盖率敏感性、固定 mask 和审计检查。

\end{{document}}
"""


def _data_card(summary: dict[str, Any], audit: dict[str, Any], sources: pd.DataFrame, language: str) -> str:
    if language == "en":
        title = "Scientific Food Composition Dataset Candidate - Data Card"
        status = "Candidate only. Domain-expert, dual-reviewer, full source-quality and licence gates remain open."
        body = f"""# {title}

**Version:** `{DATASET_VERSION}`
**Build date:** {AS_OF_DATE}
**Status:** {status}

## Intended use

Pretraining and method development for food-composition representation, family-aware component reconstruction, and text-only cold-start prediction. It is not a clinical, dietary-prescription, safety, diagnostic, or regulatory dataset.

## Composition

- Source-level food observations: {summary.get('food_observations', 0):,}
- Canonical food concepts: {summary.get('food_concepts', 0):,}
- ML food concepts: {summary.get('ml_food_concepts', 0):,}
- Train / locked validation: {summary.get('train_food_concepts', 0):,} / {summary.get('validation_food_concepts', 0):,}
- Validation panels, unseen-family / USDA Foundation: {summary.get('family_holdout_food_concepts', 0):,} / {summary.get('source_holdout_food_concepts', 0):,}
- Retained maskable / context-only components: {summary.get('maskable_targets', 0):,} / {summary.get('context_only_components', 0):,}
- Main-eligible source measurements: {summary.get('main_eligible_measurements', 0):,}
- Primary modality: `{MAIN_BASIS}`

## Identity and value policy

Original records are immutable. Canonical food identity requires compatible explicit facets. Parent-child foods are never assigned inherited values. Component merges require a formal identity key or expert review. Missing, zero, censored, trace and ranges are distinct. Only compatible exact mass-per-100-g fresh-weight values enter the primary target modality.

## Partitions and evaluation

Only `train` and locked `validation` exist. Five grouped folds are supplied within train. The unseen-family panel is family-disjoint. The USDA Foundation source panel is exact-concept and formal-lineage disjoint but deliberately permits non-duplicate training foods from the same broad family. Fixed family masks support 10%, 30%, 50% and 70% visible-composition reconstruction plus text-only cold start. The primary metric is per-axis-equal robust-standardized log MAE.

## Known limitations and release gates

- Automated audit: {audit.get('automated_release_gate', 'not run')} with {audit.get('critical_failures', 'unknown')} critical failures.
- High-heterogeneity unresolved profiles: {audit.get('unresolved_measurement_conflicts', 0):,}.
- Automated literature search is not a completed dual-reviewer scoping review.
- FoodOn coverage and scientific display-name review are incomplete.
- Source quality scores are preliminary and cannot rank databases until the official full evaluation is completed by experts.
- Some source licences require reconstruction rather than redistribution.
- Geographic and food-category coverage is uneven and must be reported in downstream work.

## Governance

Opening locked validation freezes this data version. Any subsequent data repair requires a new version and rerunning every baseline. A food-composition expert must adjudicate all high-risk rows and at least 10% of high-confidence identity mappings before scientific release.
"""
    else:
        title = "科研级食品成分数据集候选版 - Data Card"
        status = "仅为候选版；领域专家、双人文献审核、完整来源质量评价和许可门槛尚未关闭。"
        body = f"""# {title}

**版本：** `{DATASET_VERSION}`
**构建日期：** {AS_OF_DATE}
**状态：** {status}

## 预期用途

用于食品成分表征预训练、按食品家族隔离的成分重建和纯文本 cold-start 方法研究。不可用于临床决策、膳食处方、食品安全判断、诊断或监管用途。

## 数据组成

- 来源级食品记录：{summary.get('food_observations', 0):,}
- 规范食品 concept：{summary.get('food_concepts', 0):,}
- ML 食品 concept：{summary.get('ml_food_concepts', 0):,}
- Train / 锁定 validation：{summary.get('train_food_concepts', 0):,} / {summary.get('validation_food_concepts', 0):,}
- Validation panels（未见家族 / USDA Foundation）：{summary.get('family_holdout_food_concepts', 0):,} / {summary.get('source_holdout_food_concepts', 0):,}
- Maskable / context-only 成分：{summary.get('maskable_targets', 0):,} / {summary.get('context_only_components', 0):,}
- 主模态合格 source measurements：{summary.get('main_eligible_measurements', 0):,}
- 主数值模态：`{MAIN_BASIS}`

## 身份和数值规则

原始记录不可变。食品只有在显式 facets 相容时才能形成同一 canonical concept，父子食品不继承数值。成分合并需要正式身份键或专家决定。Missing、显式零、censored、trace 和范围分别保存。只有能够精确换算为 fresh-weight g/100 g 的质量值进入主目标。

## 切分与评价

只提供 `train` 和锁定 `validation`。Train 内提供五折 grouped CV。未见家族 panel 与 train 完全按家族隔离；USDA Foundation source panel 与 train 按相同 concept 和正式 lineage 隔离，但有意允许同一宽泛家族中的非重复食品训练。固定 family mask 支持 10%、30%、50%、70% 可见成分重建及纯文本 cold start。首要指标为逐轴等权的 robust-standardized log MAE。

## 已知限制和发布门槛

- 机器审计状态：{audit.get('automated_release_gate', '未运行')}；关键失败 {audit.get('critical_failures', '未知')} 项。
- 未解决高异质性 profile：{audit.get('unresolved_measurement_conflicts', 0):,}。
- 自动检索结果尚不是完成双人筛选的 scoping review。
- FoodOn 覆盖和科学化 display-name 审核尚未完成。
- 来源质量分数只是预筛，完成官方 full evaluation 前不能用于数据库排名。
- 部分来源受许可约束，只能提供重建步骤而不能直接再分发数值。
- 地域和食品类别覆盖不均，任何下游论文必须报告。

## 治理

锁定 validation 一旦开封，本数据版本即冻结。之后如修复数据，必须提升版本并重跑全部 baseline。正式科研发布前，食品成分专家必须裁决全部高风险项，并复核至少 10% 高置信度身份映射。
"""
    return body


def _preregistration() -> str:
    return f"""# Methods and Benchmark Preregistration

Version: `{DATASET_VERSION}`. Locked validation is not opened by the default command.

## Co-primary endpoints

1. Family-holdout reconstruction at 50% visible observed maskable components.
2. Text-only cold-start prediction on the locked validation panels.

Visibility curves at 10%, 30% and 70% are secondary. Component families are hidden jointly. The fixed mask file is shared by all methods.

## Models

Per-axis train mean, per-axis train median, food-group mean, frozen MiniLM text kNN, random forest and XGBoost. Hyperparameter selection occurs only in grouped train folds. A future FoodNutriGPT evaluation must use the unchanged release and masks.

## Primary metric and inference

For component `j`, transform `x` in g/100 g using `l=log(1+x)`, center by the train median and divide by `max(IQR/1.349, 1.4826*MAD)`. Compute MAE within each axis, then average axes equally. Report MSE, RMSE, raw-unit MAE/RMSE, R2, Spearman, observed range, SD, IQR, sample count and a 95% food-family cluster-bootstrap interval. Co-primary tests use family-cluster paired errors and Holm adjustment.

## Freeze rule

Validation access requires the explicit version string `{DATASET_VERSION}`. Once accessed, any identity, unit, exclusion, split, target or mask repair creates a new data version and all methods are rerun. Validation performance cannot select or redefine data.
"""


def _schema_contract() -> dict[str, Any]:
    return {
        "dataset_version": DATASET_VERSION,
        "tables": {
            "source_registry": {"primary_key": "registry_id", "purpose": "Database version, licence, acquisition, quality-review and dependency registry."},
            "food_observation": {"primary_key": "food_observation_id", "purpose": "Immutable source food identity, original names and all available facets."},
            "food_concept": {"primary_key": "food_concept_id", "purpose": "Conservative canonical identity and split-family block; source records remain linked."},
            "component_observation": {"primary_key": "component_observation_id", "purpose": "Immutable source component identifier, name, definition, unit and authority links."},
            "component_concept": {"primary_key": "component_concept_id", "purpose": "Canonical analyte/expression identity with chemical, nutritional, modality and training roles."},
            "measurement": {"primary_key": "measurement_id", "purpose": "Value-level raw and converted values, censor/range/zero state, method, sample size, quality and lineage."},
            "canonical_profile": {"primary_key": ["food_concept_id", "component_concept_id"], "purpose": "Quality-selected compatible aggregate used by the model interface."},
            "ml_partition": {"primary_key": "food_concept_id", "purpose": "Train/grouped-CV or locked-validation assignment; no test partition."},
        },
        "value_semantics": {
            "missing": "Unknown; no numeric label.",
            "explicit_zero": "Observed numeric zero.",
            "trace_or_below_limit": "Censored; no ordinary point label.",
            "range_only": "Interval retained, no ordinary point label without a reported center.",
            "main_modality": MAIN_BASIS,
        },
    }


def _build_policy() -> dict[str, Any]:
    return {
        "dataset_version": DATASET_VERSION,
        "as_of_date": AS_OF_DATE,
        "coverage_threshold": 0.02,
        "coverage_sensitivity": [0.01, 0.02, 0.05],
        "maskable_min_train_concepts": 100,
        "maskable_min_locked_validation_concepts": 30,
        "primary_modality": MAIN_BASIS,
        "accepted_value_quality_tiers": ["A", "B"],
        "aggregation": {
            "with_standard_error": "DerSimonian-Laird random effects on log1p scale",
            "otherwise": "independent-sample-count-weighted median on log1p scale",
            "heterogeneity_review": "I2 > 0.75, or >3-fold log span across at least two sources",
        },
        "food_merge": "formal lineage or exact normalized name, only with no conflicting explicit facets",
        "fuzzy_merge": "candidate and split-block evidence only; never automatic identity merge",
        "partitions": ["train", "validation"],
        "validation_panels": ["family_holdout", "source_holdout_usda_foundation"],
        "locked_validation_opened": False,
    }


def _release_manifest(root: Path, report_dir: Path, release_dir: Path, audit_root: Path, split_dir: Path) -> pd.DataFrame:
    rows = []
    roots = [release_dir, audit_root, split_dir, report_dir]
    for base in roots:
        if not base.exists():
            continue
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            relative = str(path.relative_to(root))
            if base == report_dir:
                redistribution = "project_authored_artifact"
            elif base == release_dir:
                redistribution = "candidate_mixed_source_artifact_requires_source_licence_filter"
            elif base == split_dir:
                redistribution = "derived_benchmark_artifact_requires_source_attribution_review"
            else:
                redistribution = "audit_artifact_may_contain_source_names_or_metadata_review_before_release"
            rows.append({
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "redistribution_status": redistribution,
                "dataset_version": DATASET_VERSION,
            })
    return pd.DataFrame(rows)


def run(root: Path) -> None:
    release_dir = root / "data/processed/scientific_food_composition_v1/release"
    audit_root = root / "data/audits/scientific_food_composition_v1"
    dataset_audit_dir = audit_root / "dataset"
    scoping_dir = audit_root / "scoping_review"
    split_dir = root / "data/splits/scientific_food_composition_v1"
    report_dir = root / "reports/scientific_food_composition_v1"
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = _read_json(release_dir / "dataset_summary.json")
    audit = _read_json(dataset_audit_dir / "audit_summary.json")
    prisma = _read_json(scoping_dir / "prisma_counts.json")
    components = read_component_csv(release_dir / "component_concept.csv.gz")
    sources = _source_overview(release_dir, scoping_dir, dataset_audit_dir)
    roles = _role_overview(components)
    write_csv(sources, report_dir / "source_build_overview.csv")
    write_csv(roles, report_dir / "retained_component_role_overview.csv")
    _write_descriptive_tables(release_dir, dataset_audit_dir, report_dir, components)
    write_json(_schema_contract(), release_dir / "schema_contract.json")
    write_json(_build_policy(), release_dir / "build_policy.json")

    (report_dir / "BUILD_REPORT_EN.tex").write_text(_english_tex(summary, audit, sources, roles, prisma), encoding="utf-8")
    (report_dir / "BUILD_REPORT_ZH.tex").write_text(_chinese_tex(summary, audit, sources, roles, prisma), encoding="utf-8")
    (report_dir / "DATA_CARD_EN.md").write_text(_data_card(summary, audit, sources, "en"), encoding="utf-8")
    (report_dir / "DATA_CARD_ZH.md").write_text(_data_card(summary, audit, sources, "zh"), encoding="utf-8")
    (report_dir / "METHODS_AND_BENCHMARK_PREREGISTRATION.md").write_text(_preregistration(), encoding="utf-8")
    write_json({
        "dataset_version": DATASET_VERSION,
        "scientific_release_status": audit.get("scientific_release_status", "candidate_pending_review"),
        "required_signoffs": [
            "food-composition domain expert: all high-risk plus deterministic 10% high-confidence sample",
            "two independent scoping-review screeners and adjudicator",
            "two independent FAO/INFOODS full-evaluation reviewers",
            "licence and redistribution review",
        ],
        "locked_validation_opened": False,
    }, report_dir / "OPEN_RELEASE_GATES.json")

    manifest = _release_manifest(root, report_dir, release_dir, audit_root, split_dir)
    write_csv(manifest, report_dir / "release_manifest.csv")
    print(f"Wrote bilingual reports and data cards to {report_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
