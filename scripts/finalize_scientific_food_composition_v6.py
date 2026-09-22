"""Annotate the rebuilt v6 release, verify its interface and write its report.

Colab: python scripts/build_scientific_food_composition_v6.py
       python scripts/finalize_scientific_food_composition_v6.py
No model fitting or locked-validation outcome evaluation is performed.
"""

import copy
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.benchmark import load_release, make_fixed_masks, fit_train_normalizer
from foodcomp.composition_classification import load_classification, classify_components, classification_counts
from foodcomp.composition_revision import ROLE_POLICY, MERGES, identity_key
from foodcomp.dataset import FoodCompositionDataset
from foodcomp.util import read_component_csv, write_csv, write_json, sha256_file
from report_v4_provenance import tex, identifier, table

VERSION = "scientific_food_composition_v6"
CORE = {"macro", "micro_vitamin", "micro_mineral", "choline"}

# These are reviewed distinctions, not rejected same-analyte matches.
DISTINCTIONS = [
    ("retinol / all-trans-retinol", "Frida parameter 225 explicitly carries RETOLAT (all-trans), whereas SR/CNF use RETOL. A shared display name Retinol is insufficient to erase that analytical distinction."),
    ("glucose / D-glucopyranose; fructose / beta-D-fructofuranose; lactose / beta-lactose; maltose / alpha-maltose", "A named structural form is not automatically evidence that the source measured total sugar. Preserve the form until source-expression evidence establishes the same analytical target."),
    ("vitamin B12 / cobalamin / cyanocobalamin; vitamin C / ascorbic acid / dehydroascorbic acid", "Total nutrient activity/form sum and individual chemical forms are different targets; family masks hide related forms jointly."),
    ("vitamin D total / vitamin D3 / unspecified Vitamin D structure", "D2+D3 sum is not D3 alone; a different stereochemical identifier is not an exact synonym."),
    ("folate total / food folate / free folate / folic acid", "The included molecular species and analytical definitions differ."),
    ("20:4 / arachidonic acid; 22:4 / adrenic acid; 18:4 / stearidonic acid", "Generic chain-length/double-bond totals do not assert the same omega position and stereochemistry as a named isomer."),
    ("18:1 total / cis-18:1 / oleic acid / trans-18:1", "Total, geometric family and positional isomer must remain distinct."),
    ("SR nutrient 853 versus Foundation nutrient 855", "An SR dictionary row displays PUFA 20:4 n-6 but carries formal nutrient 853 (20:3 n-6); 855 denotes 20:4 n-6. Keep distinct by formal code and flag the source name inconsistency, not merge by this display name."),
    ("Sugars with sucrose-like InChIKey; epsilon-Polylysine with lysine-like InChIKey; ALPHA-LINOLEIC-ACID", "Source-label/structure ambiguities remain flagged; no automatic reassignment to sucrose, lysine or alpha-linolenic acid."),
    ("starch / pregelatinized starch; dietary fibre / insoluble fibre / NSP fractions", "Physical form or constituent fraction is more specific than the whole aggregate; source method metadata are preserved."),
]


def read(path):
    return pd.read_csv(path, low_memory=False, keep_default_na=False, na_values=[""])


def verify_previous_conflict_winners(release, audit):
    """Independently check all old conflicts against their original candidates."""
    from foodcomp.cell_selection import SOURCE_PRIORITY
    from foodcomp.source_policy import validation_reference_mask
    previous = ROOT / 'data/processed/scientific_food_composition_v6/release'
    resolved = read(audit / 'resolved_previous_heterogeneity_cells.csv')
    keys = ['food_concept_id', 'component_concept_id']
    food_map = read(previous / 'food_observation_to_concept.csv.gz')[['food_observation_id', 'food_concept_id']]
    component_map = read(previous / 'component_observation_to_concept.csv.gz')[['component_observation_id', 'component_concept_id']]
    observation_ids = set(food_map.loc[food_map.food_concept_id.isin(resolved.food_concept_id), 'food_observation_id'])
    columns = ['measurement_id', 'food_observation_id', 'component_observation_id', 'source_key',
               'quality_tier', 'sample_count', 'normalized_value_g_per_100g',
               'main_value_eligible', 'validation_reference_eligible']
    chunks = []
    for chunk in pd.read_csv(previous / 'measurement_main_eligible.csv.gz', usecols=columns, chunksize=100000, low_memory=False):
        chunks.append(chunk[chunk.food_observation_id.isin(observation_ids)])
    candidates = pd.concat(chunks, ignore_index=True).merge(food_map, on='food_observation_id', validate='many_to_one')
    candidates = candidates.merge(component_map, on='component_observation_id', validate='many_to_one')
    candidates = candidates.merge(resolved[keys + ['partition']], on=keys, validate='many_to_one')
    candidates = candidates[candidates.partition.eq('train') | validation_reference_mask(candidates)]
    expected = resolved.set_index(keys)
    source_ranks = {s: i for i, s in enumerate(SOURCE_PRIORITY)}
    checked = 0
    for key, group in candidates.groupby(keys):
        def rank(row):
            n = row['sample_count']
            n = max(0.0, float(n)) if pd.notna(n) else 0.0
            return (source_ranks[row['source_key']], {'A': 0, 'B': 1, 'C': 2, 'D': 3}.get(row['quality_tier'], 99), -n, row['measurement_id'])
        winner = min(group.to_dict('records'), key=rank)
        actual = expected.loc[key]
        if winner['measurement_id'] != actual.selected_measurement_id or winner['normalized_value_g_per_100g'] != actual.canonical_value_g_per_100g:
            raise AssertionError(f'Previous conflict winner violates precedence: {key}')
        checked += 1
    if checked != len(resolved):
        raise AssertionError('Not all previous conflicts were independently checked')
    return checked


def classification_specification(axes):
    original = load_classification(ROOT / "data/reference/composition_classification_v5_1.json")
    specification = {key: copy.deepcopy(value) for key, value in original.items() if key != "lookup"}
    specification["version"] = VERSION + "_classification"
    specification["rules"] = []
    protein = copy.deepcopy(original["lookup"][("INFOODS", "FAT")])
    protein.update(id="macro_protein", parent="protein", subtype="total protein, nitrogen-derived",
                   chemical_family="aggregate protein", essentiality="Established macronutrient; nitrogen conversion method is retained.",
                   why_en="Total protein is a macronutrient. Nitrogen-derived protein remains protein mass; total nitrogen is a separate analytical indicator.",
                   why_zh="总蛋白质是宏量营养素。以总氮乘换算系数得到的蛋白质仍为蛋白质质量；总氮本身另列分析指标。")
    protein["refs"] = list(dict.fromkeys(protein["refs"] + ["usda_protein_method"]))
    specification["references"]["usda_protein_method"] = ["USDA FoodData Central: Foundation Foods documentation, protein calculation", "https://fdc.nal.usda.gov/Foundation_Foods_Documentation/", "Protein is commonly calculated from nitrogen using a documented factor."]
    additions = {("INFOODS", "PROCNT"): protein}
    fatty = original["lookup"][("INFOODS", "F20D3")]
    additions.update({("INFOODS", key): fatty for key in ("F20D4", "F22D4")})
    aggregate_fatty = copy.deepcopy(original["lookup"][("INFOODS", "FASAT")])
    aggregate_fatty.update(id="source_fatty_acid_aggregate", parent="fatty acids, aggregate definition unspecified",
                         why_en="The FooDB Nutrient table calls this aggregate Fatty acids. It is a lipid-constituent expression, not total fat and not one fatty acid; the aggregation scope remains source-defined.",
                         why_zh="FooDB Nutrient 表将其命名为 Fatty acids，属于脂质子成分汇总表达，不等于总脂肪，也不是某种单一脂肪酸；聚合范围仍按来源原定义保留。")
    additions[("SOURCE", "foodb:Nutrient:4")] = aggregate_fatty
    unknown = copy.deepcopy(original["lookup"][("SOURCE", "foodb:Compound:6288")])
    unknown.update(id="unresolved_foodb_4858", parent="unresolved source compound", subtype="unresolved source identity", chemical_family="unresolved", refs=["infoods"], essentiality="Unknown; no nutritional essentiality claim.",
                   why_en="Source identifier 4858 is retained, but its original dictionary identity is unresolved. It cannot be scientifically assigned to a nutrient family; it is explicitly an unresolved source composition.",
                   why_zh="来源 ID 4858 被保留，但现有原始字典不能确定具体物质。不能将其归为某种营养素，单列来源身份待确认项，不声称具有营养必需性。")
    additions[("SOURCE", "foodb:Compound:4858")] = unknown
    missing, rules = [], {}
    for axis in axes.to_dict("records"):
        rep = str(axis["classification_representative_key"])
        namespace, identity = rep.split(":", 1)
        if namespace == "CHEBI":
            identity = "CHEBI:" + identity
        source_key = (namespace, identity)
        rule = original["lookup"].get(source_key, additions.get(source_key))
        if rule is None:
            missing.append({"identity": source_key, "name": axis["canonical_name"], "train": axis["train_count"], "validation": axis["validation_count"]})
            continue
        if rule["id"] not in rules:
            rules[rule["id"]] = copy.deepcopy(rule)
            rules[rule["id"]]["members"] = {}
        members = rules[rule["id"]]["members"]
        namespace, identity = axis["authority_namespace"], axis["authority_id"]
        members[namespace] = (members.get(namespace, "") + " " + identity).strip()
    if missing:
        raise ValueError("Additional exact classification decisions required: " + json.dumps(missing, ensure_ascii=False))
    specification["rules"] = list(rules.values())
    return specification


def write_report(axes, counts, summary, decisions, aliases, output, checks, spec):
    n = len(axes)
    distribution_dir = output / 'distributions'
    distribution_manifest = distribution_dir / 'manifest.json'
    has_distributions = distribution_manifest.exists()
    if has_distributions:
        distribution_check = json.loads(distribution_manifest.read_text())
        if distribution_check['dataset_version'] != summary['dataset_version'] or distribution_check['complete_axes'] != n:
            raise ValueError('Distribution module does not match the current release; regenerate it')
        for path, digest in distribution_check['input_hashes'].items():
            if sha256_file(ROOT / path) != digest:
                raise ValueError(f'Stale distribution input: {path}; regenerate distribution report')
        for name in ['overview.tex', 'atlas.tex', 'overview.png', 'summary.json', 'README_zh.md']:
            if not (distribution_dir / name).exists():
                raise FileNotFoundError(f'Incomplete distribution module: {name}')
        for display_id in axes.display_id:
            if not (distribution_dir / 'figures' / f'{display_id}.png').exists():
                raise FileNotFoundError(f'Missing distribution figure for {display_id}')
    version_label = summary['dataset_version'].removeprefix('scientific_food_composition_').replace('_', '.')
    single_record = 'cell_selection_policy' in summary
    source_labels = {'foodb': 'FooDB', 'usda_foundation': 'USDA Foundation', 'usda_sr_legacy': 'USDA SR Legacy',
                     'cnf': 'CNF', 'frida': 'Frida', 'ciqual': 'CIQUAL'}
    if single_record:
        resolution = summary['previous_heterogeneity_resolution']
        winner_counts = ', '.join(f"{source_labels[source]}: {count}" for source, count in resolution['winner_sources'].items())
        dedup_en = [
            "Food deduplication retains the frozen food-concept mapping: formal source-lineage links or normalized exact-name matches can join records only when their explicit food facets are compatible. Different parts, preparation states and other conflicting facets remain distinct. Similar names alone do not trigger an automatic merge. Parent--child relations do not transfer composition values. This revision does not regroup foods or move the locked validation identities.",
            "Composition aliases are merged using reviewed analyte identity, scope and source definitions, not spelling alone. Different totals, fractions and isomers remain separate unless the reviewed definition explicitly combines them. A cell is identified by the unique food-concept ID and composition-concept ID, after conversion to g/100 g.",
            r"For each eligible cell, keep exactly one source record in the order \textbf{FooDB $>$ USDA Foundation $>$ USDA SR Legacy $>$ CNF $>$ Frida $>$ CIQUAL}. The first three follow the user-confirmed primary precedence; the last three retain the supplementary fallback order. Only when a preferred source has no eligible value may a lower-priority source fill the cell. This is a deterministic construction policy, not a measured universal ranking of database accuracy.",
            "All existing numerical and validation-reference gates are applied before source ranking. The Foundation source-holdout panel remains Foundation-only; no FooDB record substitutes for its validation reference. Source-published internal citations are not re-adjudicated.",
            "When the selected source itself has multiple records, retain the best existing quality tier (A, B, C, D, then ungraded), followed by the largest reported positive sample count. Missing sample counts rank below known positive counts. Remaining ties use the lexicographically smallest stable measurement ID, independent of row order and prediction error. This last tie-break ensures reproducibility, not demonstrated analytical superiority. No cross-source mean, median or random-effects estimate is released; the retained number is one original source value after unit conversion.",
            f"The preceding aggregation left {resolution['previous_groups']} within-cell heterogeneity groups without a label. Its best-quality-tier subsets contained both CNF and SR Legacy in 57 groups and multiple CNF records in 213 groups. These subsets did not include every candidate source: some lower-quality-tier SR Legacy records were previously discarded. Under source-first precedence, all {resolution['resolved_groups']} groups now have one selected value; {resolution['retained_axis_groups']} belong to retained axes. Their final source counts are {tex(winner_counts)}. No unresolved heterogeneity cell remains under the new selection policy. Selection resolves which label is used, not the underlying biological or analytical disagreement.",
            "The same rule is applied to all eligible cells, not only the previously unresolved groups. Final numerical output contains one value and one supplying source per cell. Rejected alternatives are not printed in the report; immutable originals remain available for provenance. In the per-axis catalogue, only the highest-priority source that actually contributes retained values is displayed. Other foods on that axis may use supplementary sources when the preferred source has no value; their supplying source remains recorded at cell level.",
        ]
        dedup_zh = [
            "食品沿用冻结的 food concept 映射：来源谱系关联或规范化后完全同名，且已知食品属性不冲突时才可合并。不同部位、处理和烹饪状态保持独立；相似名称只提供候选，父子食品不继承数值。本次不改变食品分组或固定验证身份。",
            "成分按审核后的分析物身份、总量/子组分范围和来源定义合并，不能只因名称相近就合并。统一为 g/100 g 后，以 food concept ID + composition concept ID 确定唯一单元格。",
            "每个单元格只保留一个原始来源值，顺序为 FooDB > USDA Foundation > USDA SR Legacy > CNF > Frida > CIQUAL。后三者是补充顺序，仅在前面来源没有合格值时使用。这是指定的构建优先级，不代表已经证明其分析准确度存在统一高低顺序。",
            "先执行数值、验证标签资格和冻结观测限制，再排序。Foundation 专用验证面板仍只用 Foundation，不由 FooDB 替换；内部转载保持原记录，不额外处理。",
            "同一来源仍有多条记录时，依次选择现有质量等级最好者（A/B/C/D/未评级）、已知有效样本数最多者；仍并列则按稳定 measurement ID 排序取首条。缺失样本数不虚构为实测样本。最后一级仅用于可复现，不意味着该记录更准确。不再对候选来源取均值、中位数或随机效应汇总。",
            f"此前 {resolution['previous_groups']} 组异质性冲突，在旧规则筛选出的最高质量等级子集中，57 组涉及 CNF 与 SR Legacy，213 组只包含 CNF 重复记录。该子集不是全部候选：部分组还存在此前因质量等级较低而未进入聚合的 SR Legacy 值。改为来源优先后，现已选出 {resolution['resolved_groups']} 个单值，{resolution['retained_axis_groups']} 个属于保留轴。最终来源：{winner_counts}。新规则下无未选定值的异质性冲突，但不声称原始测量差异本身已经消失。",
            "该规则统一用于全部合格单元格。最终数值表每格只输出一个值及一个来源，不展开未入选来源。逐轴报告只展示实际供值来源中优先级最高的一家；同轴的其他食品仍可在缺值时由补充来源供值，具体来源在单元格记录中保留。原始档案不删除。",
        ]
    else:
        dedup_en = ["FooDB-supplied cells take precedence and supplementary sources fill gaps. Existing quality and heterogeneity rules aggregate compatible selected records; unresolved conflicts stay outside ordinary labels. No cross-axis values are averaged merely because their names look similar."]
        dedup_zh = ["FooDB 为主，补充来源填充没有 FooDB 值的单元格；按要求保留内部转载，不额外回查或替换。"]
    stages = axes[axes.training_role.eq("maskable_target")].prediction_stage.value_counts()
    markdown = [f"# Composition {version_label}：资格、去重、分类与两阶段训练", "",
                f"保留 {n} 个轴，{summary['maskable_targets']} 个预测目标，{summary['context_only_components']} 个仅上下文轴。",
                f"非空食品 {summary['nonempty_food_concepts']:,} 个；固定 validation 身份 {summary['validation_identities_preserved']:,} 个。", "",
                "## 资格规则", "",
                "目标：train >= 50、validation >= 5，沿用类别可报告数据源覆盖率 >= 2% 与 train-only 稳健尺度 > 1e-8。",
                "不再要求 authority_verified/stable-source 标签，也不因 biological_equivalent/label_expression 标签直接排除。",
                "真正的能量、活性当量、无法换算的单位和检测限汇总不是 g/100g 的普通回归值，仍不混入质量模态。",
                "没有真实成分 ID 的记录不作为单一可预测物质；既有成分范围决策保留在排除档案中。",
                "未达到 50/5 或稳健尺度退化的轴，仅在满足覆盖率且 train 中有观测时作为 context；上下文缺失仍是 missing，不变成零。",
                "阈值是本研究的实验设计，不是权威机构规定的充分统计功效；5 个 validation 样本的单轴结论不稳定。", "",
                "## 去重", "",
                f"{len(decisions)} 个旧索引经 {decisions.new_component_concept_id.nunique()} 组审核合并。C001/C002 与其他总膳食纤维表达共同归入 Dietary fibre, total。",
                "合并发生在源数据值选择和 food–composition 聚合之前；重新计算独立 food concept 支持数。保留原名、原 ID、来源类别和方法。",
                *dedup_zh,
                "膳食纤维、胆固醇、B6 总量的跨方法归轴是一种更宽的研究分析物定义，不证明不同检测方法数值完全等价。", "",
                "## 为什么这样分类", "",
                "营养角色、化学结构、测量表达、训练资格分别保存。macro/micro 不能仅按含量大小划分。",
                "总蛋白质、总脂肪、总碳水、全膳食纤维属于宏量框架；水按 USDA 广义宏量定义纳入，标明不供能。",
                "公认维生素与膳食矿物质归为 micro；不同维生素形式不是新的独立必需维生素。胆碱单列为其他必需营养素。",
                "脂肪酸、氨基酸、糖及淀粉属于营养子成分，不因为质量小就成为 micronutrient；其中必需脂肪酸和必需氨基酸仍保留必需性标记。",
                "其他成分按化学/分析类型分类；污染物不是营养素，未知身份不强行贴上 macro/micro。下方逐轴给出引用和理由。", "",
                "## 两阶段训练", "",
                f"Stage 1：{int(stages.get(1, 0))} 个合格 macro、micro 和胆碱目标。学习基础营养组成。",
                f"Stage 2：{int(stages.get(2, 0))} 个其余合格成分目标，继续 Stage 1 权重，学习营养子成分和更详细的组成。",
                "两阶段均可见所有未遮蔽的观测轴，context-only 永远不产生监督损失。两阶段共用联动遮蔽，Stage 2 待预测的值在 Stage 1 也被隐藏。分类只决定监督与评估分组，不要求重新加入 axis-type embedding。",
                "总量、组分和相关形式按 mask family 联动遮蔽。Stage 1 loss 仅计核心目标，Stage 2 loss 仅计其余目标；无该阶段目标的样本跳过 loss。",
                "这个顺序是待检验的课程训练设计，不是营养学定律，尚未证明比联合训练更好。Stage 1 保存核心 checkpoint；Stage 2 后应检查核心能力是否遗忘。", "",
                "| 分类 | 总轴 | 可预测 | 仅上下文 | 预测阶段 |", "|---|---:|---:|---:|---|"]
    for row in counts.to_dict("records"):
        markdown.append(f"| {row['category_zh']} | {row['axes']} | {row['maskable_targets']} | {row['context_only']} | {1 if row['category'] in CORE else 2} |")
    if has_distributions:
        distribution_summary = json.loads((distribution_dir / 'summary.json').read_text())
        d = distribution_summary['axis_counts']
        markdown.extend(['', '## Composition mean 与 median', '',
            f"全部 {n} 个轴分别报告原始值与log1p的mean、median及差值。观测值中位数为0的有 {d['zero_median']['all']} 个。",
            '同时展示包含明确零值的全部观测、仅正值两种口径，以及逐轴ECDF。详见 PDF 附录及 distributions/README_zh.md、composition_distribution.csv、normalization_review.csv。本次不修改归一化。',
            '统计只使用train观测；未知值不当成0，不用validation分布来选择参数。'])
    markdown.extend(["", "## 暂不合并的近似名称", ""])
    markdown.extend(f"- **{name}**: {reason}" for name, reason in DISTINCTIONS)
    markdown.extend(["", "## 逐轴理由与来源", ""])
    for row in axes.to_dict("records"):
        source = aliases[aliases.component_concept_id.eq(row["component_concept_id"])]
        names = source.groupby("source_key").original_name.agg(lambda x: "; ".join(sorted(set(x))))
        markdown.extend([f"### {row['display_id']} {row['canonical_name']}", "",
                         f"- 分类：{row['classification_category_zh']}；资格：{row['training_role']}；预测阶段：{row['prediction_stage']}。",
                         f"- 样本：train {row['train_count']}，validation {row['validation_count']}。",
                         "- 理由：" + row["classification_reason_zh"],
                         "- 证据：" + row["classification_evidence_urls"],
                         "- 来源原名：" + " | ".join(f"{source_labels.get(k, k)}: {v}" for k, v in names.items()), ""])
    (output / "composition_training_report_zh.md").write_text("\n".join(markdown), encoding="utf-8")

    preamble = (ROOT / "reports/scientific_food_composition_v4/provenance_report_template.tex").read_text().split(r"\begin{document}")[0]
    preamble = preamble.replace("v4 candidate", version_label + " candidate").replace("8 September 2026", "9 September 2026").replace("scientific\\_food\\_composition\\_v4", summary['dataset_version'].replace('_', r'\_'))
    preamble = preamble.replace("Sources, Original Categories, and the Basis of Our Classification", "Composition Eligibility, Deduplication, Classification and Two-Stage Training")
    if has_distributions:
        preamble += '\n' + r'\usepackage{graphicx}' + '\n'
    preamble += "\n" + r"\widowpenalty=10000\clubpenalty=10000" + "\n"
    sections = [r"\maketitle", r"\section{Current Dataset}",
                f"The FooDB-centred release contains {summary['nonempty_food_concepts']:,} non-empty food concepts and {n} retained axes: {summary['maskable_targets']} prediction targets and {summary['context_only_components']} context-only axes. The training partition contains {summary['train_food_concepts']:,} foods. All {summary['validation_identities_preserved']:,} locked validation identities are preserved, of which {summary['validation_nonempty_foods']:,} have values. No test partition or model-performance results were generated.",
                f"The matrix contains {summary['matrix']['observed']:,} observed food--composition cells. Explicit zeros are observed values; missing cells remain unknown. Each retained value is expressed in g/100 g edible food.",
                "Direct supplementary sources are USDA SR Legacy, CNF, Frida and CIQUAL; USDA Foundation remains a source holdout. Direct AFCD, CoFID and Norway are excluded. Source-published internal citations and values are preserved without renewed origin adjudication. Food concepts and family assignments retain the existing release semantics.",
                r"\section{Eligibility}",
                r"A maskable target requires at least \textbf{50 training food concepts and 5 validation food concepts}, capability-adjusted coverage of at least 2\%, and a non-degenerate train-only robust log scale. Formal authority/stable-identity status and the labels biological-equivalent/label-expression are no longer eligibility gates. They remain descriptive metadata. Counts refer to unique food concepts, not repeated measurements, and include explicit observed zeros.",
                r"True activity equivalents, energy and a below-detection-limit fatty-acid summary are not mass regression labels. Removing expression-category gates does not convert kcal, RAE or censored summaries into grams.",
                r"Context-only axes meet the 2\% coverage rule and have at least one observed training concept, but fail support or robust-scale criteria, or retain the calculated-expression context rule. The scale is $\max(\mathrm{IQR}(\log(1+x))/1.349,1.4826\,\mathrm{MAD}(\log(1+x)))$. A zero robust scale can occur in a zero-heavy, nonconstant axis; it does not prove biological invariance. This normalization is unchanged.",
                "The 50/5 thresholds are study-design choices, not an authority-endorsed demonstration of statistical power. Per-axis estimates based on five validation foods require prominent sample counts and uncertainty. Support counts can be inspected for dataset construction, but no model error or validation outcome comparison has been opened.",
                r"\section{Deduplication}",
                *([r"\subsection{Food and Composition Identity}", *dedup_en[:2],
                   r"\subsection{Single-Value Conflict Resolution}", *dedup_en[2:],
                   r"\subsection{Reviewed Composition Merges}"] if single_record else []),
                f"{len(decisions)} existing indices are consolidated into {decisions.new_component_concept_id.nunique()} reviewed analyte groups. Each original alias, source category, definition and method remains in the source observation registry. Merges precede cell-value selection; support is recalculated on unique food--composition pairs.",
                "Old C001 (Dietary fibre) and C002 (Fiber (dietary)), together with other total-fibre expressions, form one broad Dietary fibre, total target. Whole dietary fibre is not merged with insoluble fibre or NSP fractions. Grouping method-specific fibre, cholesterol and total B6 expressions defines a broader research target; it does not establish analytical interchangeability. Method-level error analysis remains necessary. FooDB Chromium and Nickel observations denote elemental content; their ion dictionary structures do not create separate assay targets. The invalid missing-ID bucket Compound:0 is removed rather than presented as a single analyte.",
                table(["Final analyte", "Old indices"], [[tex(name), str(len(group))] for name, group in decisions.groupby("canonical_name")], [0.78, 0.15]),
                *(dedup_en if not single_record else []),
                table(["Kept distinct", "Reason"], [[tex(a), tex(b)] for a, b in DISTINCTIONS], [0.39, 0.54]),
                r"\section{Classification Rationale}",
                r"Nutritional role, chemical class, measurement expression and training role are separate fields. The mutually exclusive reporting categories are evidence-linked project assignments, not a verbatim universal ontology. The scientific basis is USDA/National Academies nutrition terminology and NIH fact sheets, together with FAO/INFOODS analytical definitions and ChEBI chemical identities \cite{nal_macro,nasem,nih,infoods,chebi}.",
                "Macronutrients include total protein, fat, total carbohydrate and whole dietary fibre; water follows the USDA broad macronutrient framework and is explicitly non-energy-yielding. Vitamins and established dietary minerals form the micronutrient group. Choline is an additional essential nutrient, listed separately. Vitamin forms are not counted as additional essential vitamins.",
                "Fatty acids, amino acids, sugars and starch are nutritional constituents, not micronutrients merely because an amount is small. Essential fatty acids and essential amino acids retain their essentiality annotation even though their prediction belongs to the constituent stage. Vitamin-related metabolites, sterols, phytochemicals, other organics, analytical indicators and contaminants are separately identified. Chemical membership alone does not establish a dietary requirement or a health benefit.",
                table(["Category", "Axes", "Targets", "Context", "Stage"], [[tex(x['category_en']), str(x['axes']), str(x['maskable_targets']), str(x['context_only']), str(1 if x['category'] in CORE else 2)] for x in counts.to_dict('records')], [0.55, 0.07, 0.08, 0.09, 0.06]),
                r"\section{Two-Stage Training Specification}",
                f"Stage 1 supervises {int(stages.get(1, 0))} eligible core axes: macronutrients, vitamins, established dietary minerals and choline. Stage 2 initializes from Stage 1 and supervises {int(stages.get(2, 0))} remaining eligible composition axes. Context-only axes receive no direct prediction loss in either stage.",
                "Both stages see the same observed, unmasked context. Joint family masking also hides Stage 2 labels during Stage 1. Related totals and constituent forms are hidden together; the stage-specific loss selects only its own target subset. Samples with no eligible targets for a stage contribute no loss. Missing observations are never converted into biological zeros. Classification selects supervision and evaluation groups; it does not require an axis-type input embedding.",
                "This curriculum is an explicit modeling hypothesis: first learn broad nutrient composition, then extend the shared representation to detailed constituents. It is not a scientific requirement for transformers. Preserve the Stage 1 checkpoint for core-nutrient evaluation and measure core retention after Stage 2. No claim of superiority over joint training or XGBoost is made by this dataset revision.",
                r"\section{Audit and Limitations}",
                f"Automated checks verify unique composition IDs and display names, unique food--composition cells, traceable original measurements, mass values in [0,100], matrix/missing-mask agreement, unchanged locked validation IDs, and train-only normalization. {checks['joint_mask_smoke_samples']} staged dataset samples were checked for identical hidden context, disjoint stage loss and no context-only loss. Fixed benchmark masks have been regenerated for the new axis universe.",
                "The release is a candidate pending domain review. Formal identity status is no longer a training gate, so remaining label/structure ambiguities are visible in the catalogue rather than silently treated as resolved. Five validation samples are weak evidence for individual axes. Broad cross-method analyte definitions and inherited food/source grouping remain limitations; this task does not certify absence of every source-copy or food-family dependence.",
                r"\section{Per-Axis Catalogue}"]
    if has_distributions:
        insert_at = sections.index(r'\section{Two-Stage Training Specification}')
        sections.insert(insert_at, r'\input{' + (distribution_dir / 'overview.tex').relative_to(ROOT).as_posix() + '}')
    for row in axes.to_dict("records"):
        source = aliases[aliases.component_concept_id.eq(row["component_concept_id"])]
        sections.extend([r"\needspace{9\baselineskip}", r"\subsection*{" + tex(row['display_id'] + ' ' + row['canonical_name']) + "}",
                         r"\textbf{Stable ID:} " + identifier(row['component_concept_id']) + ".",
                         r"\textbf{Category:} " + tex(row['classification_category_en']) + ".",
                         r"\textbf{Why:} " + tex(row['classification_reason_en']),
                         r"\textbf{Use:} " + tex(row['training_role']) + f"; prediction stage {row['prediction_stage']}; train/validation concepts {row['train_count']}/{row['validation_count']}.",
                         r"\textbf{Essentiality:} " + tex(row['essentiality_interpretation']),
                         r"\textbf{Limitation:} " + tex(row['classification_limitation']),
                         r"\textbf{Classification evidence:} " + "; ".join(r"\url{" + u + "}" for u in row['classification_evidence_urls'].split(" | ")),
                         table(["Displayed source" if single_record else "Source", "Original category", "Original name(s)"], [[tex(source_labels.get(s, s)), tex('; '.join(sorted(set(g.source_component_group.fillna('unspecified'))))), tex('; '.join(sorted(set(g.original_name))))] for s, g in source.groupby('source_key')], [0.17, 0.22, 0.54])])
    if has_distributions:
        sections.append(r'\input{' + (distribution_dir / 'atlas.tex').relative_to(ROOT).as_posix() + '}')
    refs = {
        "nal_macro": ("USDA National Agricultural Library. Macronutrients.", "https://www.nal.usda.gov/human-nutrition-and-food-safety/food-composition/macronutrients"),
        "nasem": ("National Academies. Dietary Reference Intakes.", "https://nap.nationalacademies.org/collection/57/dietary-reference-intakes"),
        "nih": ("NIH Office of Dietary Supplements. Nutrient fact sheets.", "https://ods.od.nih.gov/factsheets/list-all/"),
        "infoods": ("FAO/INFOODS. Food component identifiers and analytical definitions.", "https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/"),
        "chebi": ("EMBL-EBI. Chemical Entities of Biological Interest.", "https://www.ebi.ac.uk/chebi/"),
    }
    sections.append(r"\begin{thebibliography}{9}")
    for key, (title, url) in refs.items():
        sections.append(r"\bibitem{" + key + "}" + tex(title) + r" \url{" + url + "}.")
    sections.extend([r"\end{thebibliography}", r"\end{document}"])
    (output / "composition_training_report.tex").write_text(preamble + r"\begin{document}" + "\n\n" + "\n\n".join(sections), encoding="utf-8")


def main(*, report_only=False, version=VERSION):
    VERSION = version
    release = ROOT / "data/processed" / VERSION / "release"
    audit = ROOT / "data/audits" / VERSION / "dataset"
    output = ROOT / "reports" / VERSION
    output.mkdir(parents=True, exist_ok=True)
    if report_only:
        checks = json.loads((output / "audit_checks.json").read_text())
        if not checks["checks_passed"]:
            raise ValueError("Cannot publish a report without a passed dataset audit.")
        write_report(read(output / "composition_classification.csv"), read(output / "classification_counts.csv"),
                     json.loads((release / "dataset_summary.json").read_text()),
                     read(output / "composition_deduplication_ledger.csv"), read(output / "composition_source_aliases.csv"),
                     output, checks, load_classification(output / "composition_classification_rules.json"))
        return
    all_axes = read_component_csv(release / "component_concept.csv.gz")
    broad_masks = {
        "SOURCE:foodb:Compound:6288": "lipid_and_fatty_acid_family",
        "INCHIKEY:KDXKERNSBIXSRK-YFKPBYRVSA-N": "protein_and_amino_acid_family",
    }
    if "pre_v6_component_family" not in all_axes:
        all_axes["pre_v6_component_family"] = all_axes.component_family
    assigned = all_axes.apply(identity_key, axis=1).map(broad_masks)
    all_axes.loc[assigned.notna(), "component_family"] = assigned[assigned.notna()]
    write_csv(all_axes.loc[assigned.notna(), ["component_concept_id", "canonical_name", "pre_v6_component_family", "component_family"]], output / "conservative_mask_family_review.csv")
    retained = all_axes[all_axes.training_role.ne("excluded")].copy()
    # Finalization is rerunnable, but never annotates over a different axis matrix.
    annotations = [c for c in retained if c.startswith(("classification_", "legacy_nutritional_role")) or c in {
        "macro_micro_class", "parent_nutrient_or_family", "reporting_chemical_family", "essentiality_interpretation", "display_id", "prediction_stage"}]
    retained = retained.drop(columns=[c for c in annotations if c != "classification_representative_key"])
    spec = classification_specification(retained)
    spec_path = output / "composition_classification_rules.json"
    write_json(spec, spec_path)
    spec = load_classification(spec_path)
    axes = classify_components(retained, spec)
    axes["classification_evidence_urls"] = axes.apply(lambda row: " | ".join(dict.fromkeys(
        [url for field in (row.classification_evidence_urls, row.get("deduplication_evidence_urls", ""))
         if pd.notna(field) for url in str(field).split(" | ") if url])), axis=1)
    category_order = {c: i for i, c in enumerate(spec["categories"])}
    axes = axes.assign(category_order=axes.classification_category.map(category_order)).sort_values(["category_order", "canonical_name"]).drop(columns="category_order")
    axes["display_id"] = [f"V6-C{i:03d}" for i in range(1, len(axes) + 1)]
    axes["prediction_stage"] = np.where(axes.training_role.eq("maskable_target"), np.where(axes.classification_category.isin(CORE), 1, 2), 0)
    axes["classification_recommended_action"] = "Use explicit training_role and prediction_stage; retain source and classification limitations."
    annotation_columns = [c for c in axes if c != "classification_representative_key" and (
        c not in all_axes or c.startswith(("classification_", "legacy_nutritional_role")) or c in {
            "prediction_stage", "display_id", "macro_micro_class", "parent_nutrient_or_family",
            "reporting_chemical_family", "essentiality_interpretation"})]
    annotation_columns = list(dict.fromkeys(["component_concept_id", *annotation_columns]))
    registry = all_axes.drop(columns=[c for c in annotation_columns if c != "component_concept_id" and c in all_axes]).merge(axes[annotation_columns], on="component_concept_id", how="left", validate="one_to_one")
    registry["prediction_stage"] = registry.prediction_stage.fillna(0).astype(int)
    write_csv(registry, release / "component_concept.csv.gz")
    counts = classification_counts(axes, spec).rename(columns={"inherited_maskable_targets": "maskable_targets", "inherited_context_only": "context_only"})
    sensitivity = read(audit / "coverage_threshold_sensitivity.csv")
    sensitivity["authority_verified_count"] = [int((registry.capability_adjusted_coverage.ge(t) & registry.identity_status.eq("authority_verified")).sum()) for t in sensitivity.coverage_threshold]
    write_csv(sensitivity, audit / "coverage_threshold_sensitivity.csv")
    decisions = read(audit / "composition_deduplication_ledger.csv")
    mappings = read(release / "component_observation_to_concept.csv.gz")
    observations = read_component_csv(release / "component_observation.csv.gz")
    aliases = mappings.merge(observations, on="component_observation_id", validate="one_to_one")
    aliases = aliases[aliases.component_concept_id.isin(axes.component_concept_id)]
    if (release / "cell_selection_policy.json").exists():
        from foodcomp.cell_selection import SOURCE_PRIORITY
        selected_profiles = read(release / "canonical_profile.csv.gz")
        selected_profiles = selected_profiles[selected_profiles.component_concept_id.isin(axes.component_concept_id)]
        measurements = read(release / "measurement_main_eligible.csv.gz")
        used_observations = set(measurements.loc[measurements.measurement_id.isin(selected_profiles.measurement_ids), "component_observation_id"])
        contributing = aliases[aliases.component_observation_id.isin(used_observations)].copy()
        ranks = contributing.source_key.map({s: i for i, s in enumerate(SOURCE_PRIORITY)})
        preferred = ranks.groupby(contributing.component_concept_id).transform("min")
        aliases = contributing[ranks.eq(preferred)].copy()
        if set(aliases.component_concept_id) != set(axes.component_concept_id):
            raise AssertionError("A retained axis has no actually contributing source for the catalogue")
        if not aliases.groupby('component_concept_id').source_key.nunique().eq(1).all():
            raise AssertionError("Per-axis catalogue must display one preferred source")
        resolved = read(audit / "resolved_previous_heterogeneity_cells.csv")
        food_names = read(release / "food_concept.csv.gz")[["food_concept_id", "canonical_name"]].rename(columns={"canonical_name": "food_name"})
        axis_names = axes[["component_concept_id", "canonical_name", "display_id"]].rename(columns={"canonical_name": "composition_name"})
        resolved = resolved.merge(food_names, on="food_concept_id", validate="many_to_one").merge(axis_names, on="component_concept_id", how="left", validate="many_to_one")
        write_csv(resolved[["food_name", "display_id", "composition_name", "canonical_value_g_per_100g", "selected_source_key", "selected_measurement_id", "axis_retained", "food_concept_id", "component_concept_id"]], output / "resolved_conflict_values.csv")
    write_csv(aliases, output / "composition_source_aliases.csv")
    write_csv(axes, output / "composition_classification.csv")
    write_csv(counts, output / "classification_counts.csv")
    write_csv(decisions, output / "composition_deduplication_ledger.csv")
    previous_axes = read(ROOT / "reports/scientific_food_composition_v5_1/unique_composition_registry.csv")
    previous_axes = previous_axes.assign(category_order=previous_axes.classification_category.map(category_order)).sort_values(["category_order", "canonical_name"])
    previous_axes["old_display_id"] = [f"C{i:03d}" for i in range(1, len(previous_axes) + 1)]
    replacements = decisions.set_index("old_component_concept_id").new_component_concept_id.to_dict()
    crosswalk = previous_axes[["old_display_id", "component_concept_id", "canonical_name"]].rename(columns={"component_concept_id": "old_component_concept_id", "canonical_name": "old_name"})
    crosswalk["component_concept_id"] = crosswalk.old_component_concept_id.map(lambda x: replacements.get(x, x))
    crosswalk = crosswalk.merge(axes[["component_concept_id", "display_id", "canonical_name", "training_role", "prediction_stage"]], on="component_concept_id", how="left", validate="many_to_one")
    crosswalk["new_status"] = np.where(crosswalk.display_id.notna(), "retained", "excluded_from_new_matrix")
    write_csv(crosswalk, output / "old_report_index_to_v6.csv")
    fibre = crosswalk[crosswalk.old_display_id.isin(["C001", "C002"])]
    if len(fibre) != 2 or fibre.component_concept_id.nunique() != 1 or fibre.display_id.isna().any():
        raise AssertionError("Requested old C001/C002 merge was not implemented.")
    write_csv(pd.DataFrame(DISTINCTIONS, columns=["expressions", "why_not_merged"]), output / "preserved_distinctions.csv")
    write_csv(registry, audit / "component_training_decision_ledger.csv.gz")
    policy = {"version": VERSION, "eligibility": ROLE_POLICY,
              "stage1_categories": sorted(CORE), "stage2_categories": sorted(set(axes.classification_category) - CORE),
              "stage1_targets": axes.loc[axes.prediction_stage.eq(1), "component_concept_id"].tolist(),
              "stage2_targets": axes.loc[axes.prediction_stage.eq(2), "component_concept_id"].tolist(),
              "context_only": axes.loc[axes.prediction_stage.eq(0), "component_concept_id"].tolist(),
              "initialization": "stage2_from_stage1_internal_training_CV_selected_checkpoint",
              "loss": "masked regression on stage-specific targets only",
              "masking": "same jointly hidden component families across both stages; no hidden labels exposed as context",
              "context": "all observed, non-hidden retained axes in both stages",
              "skip_empty_stage_targets": True, "model_trained": False}
    write_json(policy, release / "two_stage_training.json")
    values, foods, components, _, _ = load_release(release)
    target_columns = np.flatnonzero(components.training_role.eq("maskable_target"))
    families = components.component_family.to_numpy()
    foods["mask_family_count"] = [len(set(families[target_columns[np.isfinite(row[target_columns])]])) for row in values]
    foods["text_task_eligible"] = np.isfinite(values[:, target_columns]).any(axis=1)
    foods["reconstruction_task_eligible"] = foods.mask_family_count.ge(2)
    write_csv(foods, release / "ml_partition.csv")
    train_rows = np.flatnonzero(foods.partition.eq("train"))
    center, scale, stats = fit_train_normalizer(values, train_rows)
    stats.insert(0, "component_concept_id", components.component_concept_id)
    write_csv(stats, release / "train_normalization.csv")
    if not np.allclose(center, components.train_log_median):
        raise AssertionError("Stored normalizer is not train-only.")
    target = components.training_role.eq("maskable_target")
    if not (components.loc[target, "train_count"].ge(50).all() and components.loc[target, "validation_count"].ge(5).all()):
        raise AssertionError("New target support threshold violated.")
    if not components.component_concept_id.is_unique or components.canonical_name.str.casefold().duplicated().any():
        raise AssertionError("Duplicate composition ID or canonical name.")
    if not (np.nanmin(values) >= 0 and np.nanmax(values) <= 100):
        raise AssertionError("Invalid chemical mass fraction.")
    archive = np.load(release / "canonical_profile_matrix.npz", allow_pickle=False)
    np.testing.assert_array_equal(np.isfinite(values), archive["observed"])
    profiles = read(release / "canonical_profile.csv.gz")
    accepted = profiles[profiles.aggregation_status.eq("accepted") & profiles.component_concept_id.isin(components.component_concept_id) & profiles.food_concept_id.isin(foods.food_concept_id)]
    if accepted.duplicated(["food_concept_id", "component_concept_id"]).any():
        raise AssertionError("Duplicate food--composition labels.")
    food_index = pd.Series(np.arange(len(foods)), index=foods.food_concept_id)
    component_index = pd.Series(np.arange(len(components)), index=components.component_concept_id)
    actual = values[accepted.food_concept_id.map(food_index).to_numpy(), accepted.component_concept_id.map(component_index).to_numpy()]
    np.testing.assert_allclose(actual, accepted.canonical_value_g_per_100g.to_numpy(), rtol=1e-6, atol=1e-10)
    if len(accepted) != int(np.isfinite(values).sum()):
        raise AssertionError("A matrix value has no canonical profile provenance.")
    used_measurements = set(accepted.measurement_ids.str.split(";").explode())
    measurements = read(release / "measurement_main_eligible.csv.gz")
    used = measurements[measurements.measurement_id.isin(used_measurements)]
    if set(used.measurement_id) != used_measurements or not used.measurement_id.is_unique:
        raise AssertionError("Broken or duplicate measurement provenance.")
    if (release / "cell_selection_policy.json").exists():
        if not profiles.selected_measurement_count.eq(1).all() or not profiles.contributing_source_count.eq(1).all():
            raise AssertionError("Final cells must contain one record from one source")
        source_values = used.set_index('measurement_id')
        np.testing.assert_allclose(accepted.canonical_value_g_per_100g,
                                   accepted.measurement_ids.map(source_values.normalized_value_g_per_100g), rtol=0, atol=0)
        if not accepted.source_keys.eq(accepted.measurement_ids.map(source_values.source_key)).all():
            raise AssertionError("Canonical source attribution differs from selected record")
        source_panel = profiles.validation_panel.fillna('').str.contains('source_holdout')
        if not profiles.loc[source_panel, 'source_keys'].eq('usda_foundation').all():
            raise AssertionError("Non-Foundation value entered source-holdout validation")
        verified_conflicts = verify_previous_conflict_winners(release, audit)
    if used.is_censored.any() or used.is_range_only.any():
        raise AssertionError("Censored or range-only record used as an exact label.")
    if not set(used.component_observation_id).issubset(set(observations.component_observation_id)):
        raise AssertionError("Source component provenance missing.")
    converted = pd.to_numeric(used.numeric_value) * pd.to_numeric(used.conversion_factor)
    np.testing.assert_allclose(converted, used.normalized_value_g_per_100g, rtol=1e-6, atol=1e-10)
    frozen = read(ROOT / "data/processed" / VERSION / "frozen_validation_identity_manifest.csv")
    if set(frozen.food_concept_id) != set(foods.loc[foods.partition.eq("validation"), "food_concept_id"]):
        raise AssertionError("Locked validation identities changed.")
    if set(components.authority_id) & {"CHOCDF", "foodb:Nutrient:38", "320", "RF-00000252-NTR", "foodb:Compound:0"}:
        raise AssertionError("Explicitly removed non-mass or carbohydrate axis retained.")
    target_ids = set(axes.loc[axes.training_role.eq("maskable_target"), "component_concept_id"])
    if set(policy["stage1_targets"]) & set(policy["stage2_targets"]) or set(policy["stage1_targets"] + policy["stage2_targets"]) != target_ids:
        raise AssertionError("Two stages do not partition maskable targets.")
    sample_count = 0
    for partition in ("train", "validation"):
        first, second = [FoodCompositionDataset(release, partition, stage=s) for s in (1, 2)]
        for i in np.linspace(0, len(first) - 1, min(250, len(first)), dtype=int):
            a, b = first[i], second[i]
            np.testing.assert_array_equal(a['context_mask'], b['context_mask'])
            if np.any(a['target_mask'] * b['target_mask']) or np.any(a['context_mask'] * a['jointly_hidden_target_mask']):
                raise AssertionError("Stage mask leakage.")
            if np.any((a['target_mask'] + b['target_mask'])[~target.to_numpy()]):
                raise AssertionError("Context-only label enters the loss.")
            sample_count += 1
    benchmark_foods = foods.copy()
    benchmark_foods["benchmark_eligible"] &= benchmark_foods.partition.eq("validation")
    masks = make_fixed_masks(values, benchmark_foods, components, release / "benchmark", dataset_version=VERSION)
    if not masks.partition.eq("validation").all():
        raise AssertionError("Locked benchmark contains training foods.")
    summary = json.loads((release / "dataset_summary.json").read_text())
    checks = {"dataset_version": VERSION, "retained_axes": len(axes), "classified_axes": len(axes),
              "stage1_targets": len(policy["stage1_targets"]), "stage2_targets": len(policy["stage2_targets"]),
              "mass_min": float(np.nanmin(values)), "mass_max": float(np.nanmax(values)),
              "joint_mask_smoke_samples": sample_count, "benchmark_mask_records": len(masks),
              "fixed_benchmark_scope": "locked_validation_only; internal-CV masks generated separately",
              "audited_observed_cells": len(accepted), "audited_source_measurements": len(used),
              "normalization_fit": "train_only", "target_thresholds": ROLE_POLICY,
              "locked_validation_preserved": True, "original_sources_retained": True,
              "model_trained": False, "expert_certified": False, "checks_passed": True}
    if (release / "cell_selection_policy.json").exists():
        checks.update(single_record_per_cell=True, exact_selected_source_value=True,
                      foundation_only_reference_panel=True, independently_verified_conflict_winners=verified_conflicts,
                      per_axis_report_source='highest_priority_actual_contributor_only')
    write_json(checks, output / "audit_checks.json")
    write_report(axes, counts, summary, decisions, aliases, output, checks, spec)
    suffix = VERSION.removeprefix('scientific_food_composition_')
    (output / "README.md").write_text(f"# Composition {suffix}\n\nSee composition_training_report_zh.md and composition_training_report.tex.\n\nBuild in Colab:\n\n```bash\npip install -r requirements-colab.txt\npython scripts/build_scientific_food_composition_{suffix}.py\npython scripts/finalize_scientific_food_composition_{suffix}.py\n```\n\nThe release is data/interface-ready, pending remaining identity and domain review. No training or validation model outcomes were run.\n", encoding="utf-8")
    write_json({"version": VERSION, "checks": checks, "release_hashes": {str(p.relative_to(release)): sha256_file(p) for p in release.rglob("*") if p.is_file()}}, release.parent / "finalized_manifest.json")
    print(json.dumps(checks, indent=2))
    print(counts.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true", help="Regenerate documentation from an already audited v6 release.")
    main(report_only=parser.parse_args().report_only)
