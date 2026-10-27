#!/usr/bin/env python3
"""Build the revised bilingual dependency-mask evidence report."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.research_r0 import digest, write_json  # noqa: E402


NAVY, ORANGE, GREEN, GRAY = "#17365D", "#F28E2B", "#59A14F", "#7F7F7F"


def md_table(frame, columns, labels=None, digits=4):
    labels = labels or columns
    lines = ["| " + " | ".join(labels) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(f"{value:.{digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=190, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--main-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    figures, tables = args.output_dir / "figures", args.output_dir / "tables"
    figures.mkdir(parents=True)
    tables.mkdir()

    rep = args.analysis_dir / "representation_r2"
    rep_v3 = args.analysis_dir / "representation_v3"
    zero = args.analysis_dir / "taiwan_zero_shot"
    few = args.analysis_dir / "taiwan_fewshot_baselines_v2"
    sparse = args.analysis_dir / "sparse_observation_v2"
    general_sens = args.analysis_dir / "sensitivity"
    group_sens = args.analysis_dir / "infoods_group_sensitivity_v1"
    ablation = args.analysis_dir / "dependency_family_ablation_v1"
    required = [rep, rep_v3, zero, few, sparse, general_sens, group_sens, ablation]
    if not all(path.exists() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.exists()])

    policy = json.loads((args.main_results_dir / "manifest.json").read_text(encoding="utf-8"))
    benchmark = pd.read_csv(args.main_results_dir / "benchmark_metrics.csv")
    category = pd.read_csv(rep / "category_metrics.csv")
    robustness = pd.read_csv(rep / "robustness.csv")
    stability = pd.read_csv(rep / "umap_seed_stability.csv")
    sources = pd.read_csv(rep_v3 / "database_sources.csv")
    zero_metrics = pd.read_csv(zero / "metrics.csv")
    zero_boot = pd.read_csv(zero / "strict_all_paired_bootstrap.csv")
    few_summary = pd.read_csv(few / "fewshot_summary.csv")
    few_runs = pd.read_csv(few / "fewshot_runs.csv")
    sparse_summary = pd.read_csv(sparse / "summary.csv")
    general_summary = pd.read_csv(general_sens / "summary.csv")
    general_boot = pd.read_csv(general_sens / "paired_bootstrap.csv")
    group_summary = pd.read_csv(group_sens / "group_sensitivity.csv")
    within_between = pd.read_csv(group_sens / "within_between_summary.csv")
    group_axes = pd.read_csv(group_sens / "infoods_axis_groups.csv")
    exposure = pd.read_csv(ablation / "exposure_summary.csv")
    perturb = pd.read_csv(ablation / "perturbation_summary.csv")
    perturb_boot = pd.read_csv(ablation / "paired_bootstrap.csv")
    reveal = pd.read_csv(ablation / "reveal_comparison.csv")
    reveal_boot = pd.read_csv(ablation / "reveal_paired_bootstrap.csv")

    copies = {
        "complete_test_model_metrics.csv": benchmark[benchmark.partition.eq("complete_test")],
        "umap_category_metrics.csv": category,
        "umap_robustness.csv": robustness,
        "umap_seed_stability.csv": stability,
        "database_sources.csv": sources,
        "taiwan_zero_shot_metrics.csv": zero_metrics,
        "taiwan_zero_shot_paired_bootstrap.csv": zero_boot,
        "taiwan_fewshot_summary.csv": few_summary,
        "taiwan_fewshot_runs.csv": few_runs,
        "sparse_observation_summary.csv": sparse_summary,
        "general_sensitivity_summary.csv": general_summary,
        "general_sensitivity_paired_bootstrap.csv": general_boot,
        "infoods_axis_groups.csv": group_axes,
        "infoods_group_sensitivity.csv": group_summary,
        "infoods_within_between_sensitivity.csv": within_between,
        "dependency_exposure_ablation.csv": exposure,
        "dependency_perturbation_ablation.csv": perturb,
        "dependency_perturbation_bootstrap.csv": perturb_boot,
        "dependency_reveal_comparison.csv": reveal,
        "dependency_reveal_bootstrap.csv": reveal_boot,
    }
    for name, frame in copies.items():
        frame.to_csv(tables / name, index=False)

    shutil.copy2(rep_v3 / "umap_food_category_database.png", figures / "umap_food_category_database.png")
    for name in ("group_sensitivity_plus10.png", "group_sensitivity_deletion.png"):
        shutil.copy2(group_sens / name, figures / name)
    for name in ("dependency_exposure_ablation.png", "dependency_perturbation_ablation.png"):
        shutil.copy2(ablation / name, figures / name)

    strict = zero_metrics[zero_metrics.panel.eq("strict_all")].copy()
    order = ["V5_CCE", "XGBoost", "RF", "KNN"]
    strict["order"] = strict.method.map({name: i for i, name in enumerate(order)})
    strict = strict.sort_values("order")
    bars = plt.bar(
        strict.method.replace({"V5_CCE": "V5+CCE"}), strict.macro_scaled_log_mae,
        color=[NAVY, ORANGE, GREEN, GRAY],
    )
    plt.bar_label(bars, fmt="%.3f", padding=3)
    plt.ylabel("Macro scaled-log MAE")
    plt.title("Taiwan TFDA zero-shot completion")
    plt.ylim(0, strict.macro_scaled_log_mae.max() * 1.18)
    savefig(figures / "taiwan_zero_shot.png")

    palette = {"V5+CCE frozen": NAVY, "RF": GREEN, "XGBoost": ORANGE}
    marker = {"V5+CCE frozen": "o", "RF": "s", "XGBoost": "^"}
    for method in ["V5+CCE frozen", "RF", "XGBoost"]:
        current = few_summary[few_summary.method.eq(method)].sort_values("shots")
        plt.errorbar(
            current.shots, current.macro_scaled_log_mae_mean,
            yerr=current.macro_scaled_log_mae_sd, color=palette[method],
            marker=marker[method], linewidth=2, capsize=4, label=method,
        )
    plt.xscale("log", base=2)
    plt.xticks([8, 32, 128], ["8", "32", "128"])
    plt.xlabel("Taiwan adaptation profiles")
    plt.ylabel("Macro scaled-log MAE")
    plt.title("Frozen foundation model vs few-shot tree fitting")
    plt.legend(frameon=False)
    savefig(figures / "taiwan_fewshot_curve.png")

    budget_order = ["5", "10", "20", "40", "all"]
    sparse_summary["order"] = sparse_summary.visible_budget.map({x: i for i, x in enumerate(budget_order)})
    sparse_palette = {"V5_CCE": NAVY, "RF": GREEN, "XGBoost": ORANGE, "KNN": GRAY}
    for method in ["V5_CCE", "RF", "XGBoost", "KNN"]:
        current = sparse_summary[sparse_summary.method.eq(method)].sort_values("order")
        plt.plot(
            current.order, current.macro_scaled_log_mae_mean, marker="o",
            color=sparse_palette[method], label=method.replace("_", "+"),
        )
    plt.xticks(range(5), ["5", "10", "20", "40", "All"])
    plt.xlabel("Visible component budget")
    plt.ylabel("Macro scaled-log MAE")
    plt.title("Completion with thinned context")
    plt.legend(frameon=False)
    savefig(figures / "sparse_observation_curve.png")

    complete = benchmark[benchmark.partition.eq("complete_test")].copy()
    complete = complete[complete.method.isin(["V5+CCE", "XGBoost", "RF", "KNN"])]
    complete = complete.sort_values("macro_scaled_log_mae")
    complete_table = md_table(
        complete, ["method", "macro_scaled_log_mae", "macro_scaled_log_rmse", "macro_raw_mae"],
        ["Method", "SLMAE", "SLRMSE", "Raw MAE"],
    )
    cat = category[category.method.isin(["Nutrients_mask", "V4", "V5", "V5_CCE"])].copy()
    cat["method"] = cat.method.replace({"Nutrients_mask": "Values+mask", "V5_CCE": "V5+CCE"})
    category_table = md_table(cat, ["method", "macro_purity_10", "macro_f1", "balanced_accuracy"],
                              ["Representation", "Macro purity@10", "Macro-F1", "Balanced acc."])
    source_table = md_table(sources, ["source_code", "source_key", "source_name", "umap_profiles"],
                            ["Code", "Source key", "Database", "UMAP n"], digits=0)
    zero_display = strict.copy()
    zero_display["method"] = zero_display.method.replace({"V5_CCE": "V5+CCE"})
    zero_table = md_table(zero_display, ["method", "macro_scaled_log_mae", "macro_scaled_log_rmse", "macro_raw_mae"],
                          ["Method", "SLMAE", "SLRMSE", "Raw MAE"])
    few_table = md_table(few_summary.sort_values(["shots", "method"]),
                         ["method", "shots", "macro_scaled_log_mae_mean", "macro_scaled_log_mae_sd"],
                         ["Method", "Profiles", "SLMAE mean", "SD"])
    sparse_table = md_table(sparse_summary.sort_values(["order", "method"]),
                            ["method", "visible_budget", "macro_scaled_log_mae_mean", "macro_scaled_log_mae_sd"],
                            ["Method", "Visible", "SLMAE mean", "SD"])
    wb = within_between.pivot(index="condition", columns="relation", values="prediction_shift")
    wb_table = md_table(within_between, ["condition", "relation", "prediction_shift", "error_delta", "pairs"],
                        ["Condition", "Relation", "Prediction shift", "Error delta", "Pairs"])
    exp_all = exposure[exposure.component.eq("ALL")]
    exp_table = md_table(exp_all, ["condition", "absolute_error", "prediction_shift_vs_masked", "target_axes", "jobs"],
                         ["Condition", "SLMAE", "Shift vs masked", "Axes", "Jobs"])
    pert_all = perturb[perturb.component.eq("ALL")]
    pert_table = md_table(pert_all, ["factor", "family_shift", "control_shift", "family_error_delta", "control_error_delta"],
                          ["Factor", "Dependency shift", "Control shift", "Dependency Δerror", "Control Δerror"])
    reveal_all = reveal[reveal.component.eq("ALL")].iloc[0]
    rb = reveal_boot.iloc[0]
    group_manifest = json.loads((group_sens / "manifest.json").read_text(encoding="utf-8"))
    ablation_manifest = json.loads((ablation / "manifest.json").read_text(encoding="utf-8"))
    few_frozen = few_summary[few_summary.method.eq("V5+CCE frozen")].iloc[0]
    few_rf = few_summary[few_summary.method.eq("RF")].set_index("shots")
    few_xgb = few_summary[few_summary.method.eq("XGBoost")].set_index("shots")
    sparse_pivot = sparse_summary.pivot(index="visible_budget", columns="method", values="macro_scaled_log_mae_mean")
    v5zero = strict[strict.method.eq("V5_CCE")].iloc[0]

    zh = f"""# 显式依赖 mask 下的食物组成基础模型证据：修订分析

**版本：** 2026-10-28 r2

**主模型：** V5+CCE（dependency-mask v1）

**范围：** mask 训练审计、food/database UMAP、台湾 zero/few-shot、少观测补全、INFOODS 组件组 sensitivity、dependency 暴露消融

## 摘要

当前 dependency mask **同时影响训练和预测**。训练中的每个 profile-target 任务先隐藏目标轴及其有向祖先和后代；validation、complete test、台湾外部验证和以下分析复用同一 policy。V4、V5、V5+CCE 均在这个 mask 下从头训练。

本次按要求修订四部分：UMAP 只保留 food category 与 database source；台湾 few-shot 只比较冻结 V5+CCE 与台湾少样本 RF/XGBoost；少 observation 只比较 V5+CCE、RF、XGBoost、KNN；sensitivity 增加 FAO/INFOODS 组件组矩阵和 dependency-family 推理暴露消融。

主要结果：

- 台湾固定测试集上，冻结 V5+CCE 为 **{few_frozen.macro_scaled_log_mae_mean:.4f}**。RF 的 8/32/128-shot 为 **{few_rf.loc[8].macro_scaled_log_mae_mean:.4f}/{few_rf.loc[32].macro_scaled_log_mae_mean:.4f}/{few_rf.loc[128].macro_scaled_log_mae_mean:.4f}**；XGBoost 为 **{few_xgb.loc[8].macro_scaled_log_mae_mean:.4f}/{few_xgb.loc[32].macro_scaled_log_mae_mean:.4f}/{few_xgb.loc[128].macro_scaled_log_mae_mean:.4f}**。冻结模型在三个样本量都更好。
- +10% 单输入扰动中，INFOODS 同组 input-target 的平均 shift 是 **{wb.loc['plus_10pct','within_group']:.5f}**，跨组是 **{wb.loc['plus_10pct','between_group']:.5f}**；删除输入时分别是 **{wb.loc['deletion','within_group']:.5f}** 与 **{wb.loc['deletion','between_group']:.5f}**。
- dependency 暴露消融没有证明严格父子/包含轴在当前模型中比普通轴影响更强。暴露全部通常隐藏值后，SLMAE 从 **{exp_all.set_index('condition').loc['standard_mask'].absolute_error:.4f}** 变为 **{exp_all.set_index('condition').loc['expose_all'].absolute_error:.4f}**；依赖轴扰动的 shift 也整体低于 matched control。这是重要的负结果。

## 1. 共同设置与 mask

mask 图只登记 FAO/INFOODS tagname 定义能够明确支持的 totals、components、subtotals 和 structural members。37 条有向边覆盖 44 个轴；187 个监督轴中 144 个是 singleton。目标 mask 为 `目标 + 全部有向祖先 + 全部有向后代`，siblings 仍可见，旧 `mask_family` 不参与训练或评估。

{complete_table}

## 2. UMAP：food category 与 database

UMAP 使用 4,000 个 validation profile 的 V5+CCE 第一遍 `[FOOD]` 表征；类别和 database 只用于着色，不进入模型。图中不再展示 observation count 或单个 nutrient 梯度。

![Food category 与 database UMAP](figures/umap_food_category_database.png)

{category_table}

database code 对应如下：

{source_table}

缺失模式本身对 source 的 balanced accuracy 仍达到约 0.971，因此 database 分离包含明显的 coverage/missingness 混杂，不能单独作为语义表征证据。

## 3. 台湾 zero-shot

台湾数据不参与训练、尺度估计、调参或 checkpoint 选择。strict panel 含 2,212 个 food、128,470 个 food-axis job 和 80 个轴。

![台湾 zero-shot](figures/taiwan_zero_shot.png)

{zero_table}

V5+CCE 的主要 scaled-log MAE 为 **{v5zero.macro_scaled_log_mae:.4f}**，仍优于三个 source-trained baseline；raw MAE 最优仍可能是 XGBoost。

## 4. 台湾 few-shot：冻结模型与树模型少样本拟合

固定 400 个 name-novel food 为测试集，1,389 个为 adaptation pool。8-shot 子集先按缺失模式做 axis coverage 集合覆盖，再随机填满；不查看数值或测试误差。子集保持 `8 ⊂ 32 ⊂ 128`，三个 seed 使用相同协议。V5+CCE 完全冻结；RF/XGBoost 对每个轴只用相应台湾 adaptation profiles 拟合正式配置。这里没有 scratch Transformer，也没有 Transformer fine-tuning。

![台湾 few-shot](figures/taiwan_fewshot_curve.png)

{few_table}

128 个台湾 food 后，XGBoost 已到 {few_xgb.loc[128].macro_scaled_log_mae_mean:.4f}、RF 到 {few_rf.loc[128].macro_scaled_log_mae_mean:.4f}，但仍未达到冻结 V5+CCE 的 {few_frozen.macro_scaled_log_mae_mean:.4f}。这比 pretrained-vs-scratch Transformer 更直接地显示跨数据库预训练的样本效率。

## 5. 少量可见 observation

固定 12 个高支持目标，每轴 200 个 validation job。先应用 dependency mask，再随机保留 5、10、20、40 或全部 context；有限预算重复三次。只比较四个指定方法。

![少观测补全](figures/sparse_observation_curve.png)

{sparse_table}

V5+CCE 在 5/10/20 个可见值时分别为 {sparse_pivot.loc['5','V5_CCE']:.3f}/{sparse_pivot.loc['10','V5_CCE']:.3f}/{sparse_pivot.loc['20','V5_CCE']:.3f}，均优于三个 baseline。40 个和 all 时 RF 最强，XGBoost 次之。这说明 foundation advantage 集中在真正稀疏的 context，而不是充分观测条件。

## 6. FAO/INFOODS 组件组 sensitivity

### 6.1 分类依据

只采用一个来源：[FAO/INFOODS 与 EuroFIR 组件命名 workshop report](https://www.fao.org/fileadmin/templates/food_composition/documents/upload/9IFDCReportfinal.pdf) Annex 1 的组件层级，并用 [INFOODS tagnames](https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/)核对名称。Annex 的层级标为 draft proposition，因此只登记氨基酸、脂肪酸、糖、膳食纤维、矿物质、维生素、有机酸七个清楚的大组；其余轴保持未分类。本分析共分类 {group_manifest['classified_axes']}/252 个轴，其中 {group_manifest['classified_target_axes']} 个为监督目标。这个分类只用于汇总 sensitivity，不参与 mask 或模型输入。

### 6.2 跨组矩阵

每个 target job 先应用 dependency mask，再在每个可用 input group 中固定抽一个正值轴。共 {group_manifest['selected_input_target_pairs']:,} 个配对组合。矩阵数值是 target-axis macro 的 normalized-log prediction shift。

![+10% 组件组 sensitivity](figures/group_sensitivity_plus10.png)

![删除一个输入的组件组 sensitivity](figures/group_sensitivity_deletion.png)

{wb_table}

对角线总体高于非对角线：+10% 时约为跨组的 {wb.loc['plus_10pct','within_group']/wb.loc['plus_10pct','between_group']:.1f} 倍，deletion 时约为 {wb.loc['deletion','within_group']/wb.loc['deletion','between_group']:.1f} 倍。糖与膳食纤维的组内响应最强，但膳食纤维部分 cell 支持量很低，必须结合 `tables/infoods_group_sensitivity.csv` 的 jobs/axes 阅读。这些是预测关联，不是营养干预因果效应。

## 7. Dependency-family 暴露与扰动消融

本实验只使用当前 policy 本来会隐藏的 target-specific ancestors/descendants；siblings 本来就可见。共 {ablation_manifest['jobs']:,} 个 job、{ablation_manifest['targets']} 个目标和 {ablation_manifest['dependency_components']} 个 dependency components。标准输入依次改为暴露一个、一半、全部已观测依赖亲属。

![Dependency 暴露消融](figures/dependency_exposure_ablation.png)

{exp_table}

暴露越多并没有改善准确率，反而使整体误差上升。这符合一个关键事实：模型在训练时从未见过“预测目标同时看到其祖先/后代”的任务，推理时解开 mask 是分布外输入。

随后分别扰动一个已暴露依赖轴和一个按 INFOODS 组/normalized value 匹配的非依赖可见 control：

![Dependency 与 matched control 扰动](figures/dependency_perturbation_ablation.png)

{pert_table}

单纯 reveal 一个值时，dependency shift 为 {reveal_all.family_reveal_shift:.5f}，control reveal shift 为 {reveal_all.control_reveal_shift:.5f}；profile bootstrap 的 dependency-minus-control 95% 区间为 [{rb.ci_low:.5f}, {rb.ci_high:.5f}]。三种数值扰动下 dependency shift 也均较小，bootstrap 区间见附件。

因此本消融**不能证明**当前模型在推理时强烈依赖这些父子轴。它反而支持继续 mask：一旦只在推理时开放，误差变差。父子/包含关系的科学依据来自 INFOODS 定义；若要检验“不 mask 会带来多大预测泄漏”，需要重新训练一个 matched no-dependency-mask 模型，不能用这次 inference-only OOD 消融替代。

## 8. 结论与限制

1. 新 mask 确实进入训练与所有预测路径。
2. 冻结 V5+CCE 在台湾 fixed test 上优于仅用 8–128 个台湾 food 拟合的 RF/XGBoost，支持预训练的外部样本效率。
3. 少 observation 下 V5+CCE 在 5–20 个可见值最强；40/all 时树模型更强。
4. FAO/INFOODS 大组矩阵显示组内响应整体大于组间，但这是关联结构。
5. inference-only dependency 暴露是负结果，不能作为依赖关系已被模型学会的证据。
6. UMAP 的 database 分离受 missingness 强烈混杂；台湾只覆盖 80/187 个目标；所有神经结果仍来自单一预训练 seed。

## 9. 复现与附件

报告只包含聚合表、图和哈希，不包含 checkpoint、原始数据或逐条预测。完整脚本、manifest、表和图随报告提交。
"""

    en = f"""# Food-Composition Foundation Evidence under Explicit Dependency Masking: Revised Analyses

**Version:** 2026-10-28 r2

**Primary model:** V5+CCE (dependency-mask v1)

**Scope:** mask audit, food/database UMAP, Taiwan zero/few-shot, sparse completion, INFOODS group sensitivity, dependency exposure ablation

## Executive summary

The dependency mask affects **both training and inference**. Every training profile-target task hides the target plus its directed ancestors and descendants; validation, complete test, Taiwan transfer, and all analyses use the same policy. V4, V5, and V5+CCE were trained from scratch under this mask.

This revision keeps only food-category and database-source UMAP views; compares frozen V5+CCE with Taiwan few-shot RF/XGBoost without a scratch Transformer; restricts sparse completion to V5+CCE, RF, XGBoost, and KNN; and adds an FAO/INFOODS component-group sensitivity matrix plus a dependency exposure ablation.

Key results:

- Frozen V5+CCE scores **{few_frozen.macro_scaled_log_mae_mean:.4f}** on the fixed Taiwan test. RF scores **{few_rf.loc[8].macro_scaled_log_mae_mean:.4f}/{few_rf.loc[32].macro_scaled_log_mae_mean:.4f}/{few_rf.loc[128].macro_scaled_log_mae_mean:.4f}** at 8/32/128 shots; XGBoost scores **{few_xgb.loc[8].macro_scaled_log_mae_mean:.4f}/{few_xgb.loc[32].macro_scaled_log_mae_mean:.4f}/{few_xgb.loc[128].macro_scaled_log_mae_mean:.4f}**. The frozen model is better at every budget.
- For +10% input perturbations, within-group prediction shift is **{wb.loc['plus_10pct','within_group']:.5f}** versus **{wb.loc['plus_10pct','between_group']:.5f}** between groups; deletion gives **{wb.loc['deletion','within_group']:.5f}** versus **{wb.loc['deletion','between_group']:.5f}**.
- The inference-only dependency exposure ablation does **not** show stronger use of strict dependency relatives than ordinary controls. Exposing all normally hidden relatives changes SLMAE from **{exp_all.set_index('condition').loc['standard_mask'].absolute_error:.4f}** to **{exp_all.set_index('condition').loc['expose_all'].absolute_error:.4f}** and therefore worsens accuracy.

## 1. Shared setup and mask

The graph contains only totals, components, subtotals, and structural members explicitly supported by FAO/INFOODS tagname definitions. It has 37 directed edges over 44 axes; 144 of 187 supervised targets are singletons. A target mask is `target + all directed ancestors + all directed descendants`; siblings remain visible and the legacy `mask_family` column is unused.

{complete_table}

## 2. UMAP: food category and database

UMAP uses the first-pass V5+CCE `[FOOD]` representation for 4,000 validation profiles. Category and database labels are used only for colouring. Observation-count and single-nutrient gradient panels have been removed.

![Food-category and database UMAP](figures/umap_food_category_database.png)

{category_table}

Database codes are listed below.

{source_table}

Missingness alone reaches about 0.971 source balanced accuracy, so database separation is materially confounded by coverage patterns.

## 3. Taiwan zero-shot

Taiwan is excluded from training, scale estimation, tuning, and checkpoint selection. The strict panel contains 2,212 foods, 128,470 food-axis jobs, and 80 axes.

![Taiwan zero-shot](figures/taiwan_zero_shot.png)

{zero_table}

V5+CCE has the best primary scaled-log MAE, while raw MAE can still favour XGBoost.

## 4. Taiwan few-shot: frozen model versus tree fitting

A fixed 400 name-novel food test is held out from 1,389 adaptation foods. The 8-shot subset first uses missingness-only set coverage so every test axis has at least one label, then fills randomly; subsets are nested through 32 and 128. V5+CCE remains frozen. RF and XGBoost use their formal configurations and are fit per axis only on the selected Taiwan profiles. There is no scratch Transformer and no Transformer fine-tuning.

![Taiwan few-shot](figures/taiwan_fewshot_curve.png)

{few_table}

Even at 128 Taiwan foods, XGBoost reaches {few_xgb.loc[128].macro_scaled_log_mae_mean:.4f} and RF {few_rf.loc[128].macro_scaled_log_mae_mean:.4f}, compared with {few_frozen.macro_scaled_log_mae_mean:.4f} for frozen V5+CCE. This directly demonstrates external sample efficiency from pretraining.

## 5. Completion with few visible observations

The panel fixes 12 high-support targets and 200 validation jobs per target. Dependency masking is applied first, followed by random retention of 5, 10, 20, 40, or all context values. Only the four requested methods are retained.

![Sparse completion](figures/sparse_observation_curve.png)

{sparse_table}

V5+CCE leads at 5, 10, and 20 visible values. RF leads at 40 and all, with XGBoost second. The foundation-model advantage is concentrated in genuinely sparse context.

## 6. FAO/INFOODS component-group sensitivity

### 6.1 Classification authority

One source is used: the [FAO/INFOODS and EuroFIR component nomenclature workshop report](https://www.fao.org/fileadmin/templates/food_composition/documents/upload/9IFDCReportfinal.pdf), Annex 1, with names checked against [INFOODS tagnames](https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/). The Annex hierarchy is labelled a draft proposition, so only seven clear groups are registered: amino acids, fatty acids, sugars, dietary fibre, minerals, vitamins, and organic acids. Unlisted axes remain unclassified. The mapping covers {group_manifest['classified_axes']}/252 axes and {group_manifest['classified_target_axes']} supervised targets. It is used only for sensitivity aggregation and never for masking or model input.

### 6.2 Cross-group matrix

After dependency masking, one positive visible input is selected from every available input group for each target job. The audit contains {group_manifest['selected_input_target_pairs']:,} input-target pairs.

![+10% group sensitivity](figures/group_sensitivity_plus10.png)

![Single-input deletion group sensitivity](figures/group_sensitivity_deletion.png)

{wb_table}

The diagonal is about {wb.loc['plus_10pct','within_group']/wb.loc['plus_10pct','between_group']:.1f} times the off-diagonal mean for +10% and {wb.loc['deletion','within_group']/wb.loc['deletion','between_group']:.1f} times for deletion. Sugars and dietary fibre have the strongest diagonal responses, although fibre cells have low support. These are predictive associations, not causal nutritional interventions.

## 7. Dependency-family exposure and perturbation ablation

This experiment uses only target-specific ancestors/descendants that the current policy normally hides; siblings are already visible. It covers {ablation_manifest['jobs']:,} jobs, {ablation_manifest['targets']} targets, and {ablation_manifest['dependency_components']} dependency components.

![Dependency exposure ablation](figures/dependency_exposure_ablation.png)

{exp_table}

Exposing more dependency values does not improve accuracy. It is an out-of-distribution inference condition because training never allowed a target to see these relatives.

One exposed dependency value and a matched unrelated visible control are then perturbed separately.

![Dependency versus matched-control perturbation](figures/dependency_perturbation_ablation.png)

{pert_table}

Revealing one dependency value shifts prediction by {reveal_all.family_reveal_shift:.5f}, versus {reveal_all.control_reveal_shift:.5f} for the control. The profile-bootstrap 95% interval for dependency minus control is [{rb.ci_low:.5f}, {rb.ci_high:.5f}]. Dependency perturbation shifts are also lower overall at all three factors.

This ablation therefore does **not** establish that the trained model strongly uses these strict parent-child relations when they are exposed only at inference. It supports retaining the mask because inference-only unmasking worsens error. The scientific relation itself comes from INFOODS definitions. Quantifying leakage from omitting the mask would require a separately trained matched no-dependency-mask model.

## 8. Conclusions and limitations

1. The new mask is part of training and every evaluation path.
2. Frozen V5+CCE beats RF/XGBoost fit on 8–128 labelled Taiwan foods, supporting external sample efficiency.
3. V5+CCE leads sparse completion at 5–20 visible values; trees lead with richer context.
4. Broad INFOODS groups show stronger within-group than between-group response, but this is associative.
5. Inference-only strict-dependency exposure is a negative result and cannot prove learned dependency use.
6. UMAP database separation is strongly confounded by missingness; Taiwan covers 80/187 targets; neural evidence uses one pretrained seed.

## 9. Reproducibility and attachments

The package contains aggregate tables, figures, hashes, and scripts. It excludes checkpoints, raw data, and row-level predictions.
"""

    (args.output_dir / "report_zh.md").write_text(zh, encoding="utf-8")
    (args.output_dir / "report_en.md").write_text(en, encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Revised dependency-mask foundation evidence report\n\n"
        "Bilingual aggregate report. No checkpoints, raw data, or row-level predictions are included.\n",
        encoding="utf-8",
    )
    manifest = {
        "status": "aggregate_report_sources_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "dependency-mask r2 analyses requested 2026-10-28",
        "raw_data_included": False, "model_weights_included": False,
        "row_level_predictions_included": False,
        "source_hashes": {
            "policy_manifest": digest(args.main_results_dir / "manifest.json"),
            "representation_manifest": digest(rep / "manifest.json"),
            "taiwan_zero_shot_manifest": digest(zero / "manifest.json"),
            "taiwan_fewshot_manifest": digest(few / "manifest.json"),
            "sparse_manifest": digest(sparse / "manifest.json"),
            "general_sensitivity_manifest": digest(general_sens / "manifest.json"),
            "group_sensitivity_manifest": digest(group_sens / "manifest.json"),
            "dependency_ablation_manifest": digest(ablation / "manifest.json"),
        },
    }
    write_json(args.output_dir / "report_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
