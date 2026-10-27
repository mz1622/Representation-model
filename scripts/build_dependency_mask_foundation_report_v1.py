#!/usr/bin/env python3
"""Build bilingual aggregate reports for dependency-mask v1 analyses."""
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


NAVY = "#17365D"
BLUE = "#4F81BD"
ORANGE = "#F28E2B"
GREEN = "#59A14F"
RED = "#E15759"
GRAY = "#7F7F7F"


def table(frame, columns, labels=None, digits=4):
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


def pct_reduction(new, old):
    return 100 * (old - new) / old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--main-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    figures = args.output_dir / "figures"
    tables = args.output_dir / "tables"
    figures.mkdir(parents=True)
    tables.mkdir()

    representation = args.analysis_dir / "representation_r2"
    zero = args.analysis_dir / "taiwan_zero_shot"
    fewshot = args.analysis_dir / "taiwan_fewshot"
    sparse = args.analysis_dir / "sparse_observation"
    sensitivity = args.analysis_dir / "sensitivity"
    required = [representation, zero, fewshot, sparse, sensitivity]
    if not all(path.exists() for path in required):
        raise FileNotFoundError([str(path) for path in required if not path.exists()])

    policy = json.loads((args.main_results_dir / "manifest.json").read_text(encoding="utf-8"))
    benchmark = pd.read_csv(args.main_results_dir / "benchmark_metrics.csv")
    category = pd.read_csv(representation / "category_metrics.csv")
    robustness = pd.read_csv(representation / "robustness.csv")
    umap_stability = pd.read_csv(representation / "umap_seed_stability.csv")
    source_metrics = json.loads((representation / "missingness_source_metrics.json").read_text())
    zero_metrics = pd.read_csv(zero / "metrics.csv")
    zero_boot = pd.read_csv(zero / "strict_all_paired_bootstrap.csv")
    few_summary = pd.read_csv(fewshot / "fewshot_summary.csv")
    few_runs = pd.read_csv(fewshot / "fewshot_runs.csv")
    few_zero = json.loads((fewshot / "zero_shot_metrics.json").read_text())
    sparse_summary = pd.read_csv(sparse / "summary.csv")
    sensitivity_summary = pd.read_csv(sensitivity / "summary.csv")
    sensitivity_boot = pd.read_csv(sensitivity / "paired_bootstrap.csv")

    copies = {
        "complete_test_model_metrics.csv": benchmark[benchmark.partition.eq("complete_test")],
        "umap_category_metrics.csv": category,
        "umap_robustness.csv": robustness,
        "umap_seed_stability.csv": umap_stability,
        "taiwan_zero_shot_metrics.csv": zero_metrics,
        "taiwan_zero_shot_paired_bootstrap.csv": zero_boot,
        "taiwan_fewshot_summary.csv": few_summary,
        "taiwan_fewshot_runs.csv": few_runs,
        "sparse_observation_summary.csv": sparse_summary,
        "sensitivity_summary.csv": sensitivity_summary,
        "sensitivity_paired_bootstrap.csv": sensitivity_boot,
    }
    for name, frame in copies.items():
        frame.to_csv(tables / name, index=False)
    shutil.copy2(representation / "umap_v5_cce.png", figures / "umap_v5_cce.png")

    # Category representation comparison.
    display_methods = ["Nutrients_mask", "PCA50", "V4", "V5", "V5_CCE"]
    plot = category.set_index("method").loc[display_methods]
    x = np.arange(len(plot))
    width = 0.36
    plt.figure(figsize=(8.6, 4.2))
    plt.bar(x - width / 2, plot.macro_purity_10, width, label="Macro purity@10", color=BLUE)
    plt.bar(x + width / 2, plot.macro_f1, width, label="Linear macro-F1", color=ORANGE)
    plt.xticks(x, ["Nutrients\n+ mask", "PCA-50", "V4", "V5", "V5+CCE"])
    plt.ylim(0, 1)
    plt.ylabel("Validation score")
    plt.title("Food-category structure in representations")
    plt.legend(frameon=False)
    savefig(figures / "representation_category_metrics.png")

    # Taiwan zero-shot.
    strict = zero_metrics[zero_metrics.panel.eq("strict_all")].copy()
    order = ["V5_CCE", "XGBoost", "RF", "KNN"]
    strict["method"] = pd.Categorical(strict.method, order, ordered=True)
    strict = strict.sort_values("method")
    plt.figure(figsize=(7.6, 4.2))
    colors = [NAVY, ORANGE, GREEN, GRAY]
    zero_labels = strict.method.astype(str).replace({"V5_CCE": "V5+CCE"})
    bars = plt.bar(zero_labels, strict.macro_scaled_log_mae, color=colors)
    plt.bar_label(bars, fmt="%.3f", padding=3)
    plt.ylabel("Macro scaled-log MAE")
    plt.ylim(0, strict.macro_scaled_log_mae.max() * 1.18)
    plt.title("Taiwan TFDA zero-shot completion")
    savefig(figures / "taiwan_zero_shot.png")

    # Few-shot learning curve.
    plt.figure(figsize=(7.6, 4.4))
    for initialization, color, marker in (
        ("pretrained", NAVY, "o"), ("scratch", ORANGE, "s"),
    ):
        current = few_summary[few_summary.initialization.eq(initialization)].sort_values("shots")
        plt.errorbar(
            current.shots, current.macro_scaled_log_mae_mean,
            yerr=current.macro_scaled_log_mae_sd, marker=marker, linewidth=2,
            capsize=4, color=color, label=initialization.capitalize(),
        )
    plt.axhline(few_zero["macro_scaled_log_mae"], color=GRAY, linestyle="--", label="Frozen zero-shot")
    plt.xscale("log", base=2)
    plt.xticks([8, 32, 128], ["8", "32", "128"])
    plt.xlabel("Taiwan adaptation profiles")
    plt.ylabel("Macro scaled-log MAE")
    plt.title("Taiwan few-shot adaptation: same architecture")
    plt.legend(frameon=False)
    savefig(figures / "taiwan_fewshot_curve.png")

    # Sparse observation curve.
    budget_order = ["5", "10", "20", "40", "all"]
    sparse_summary["budget_order"] = sparse_summary.visible_budget.map(
        {value: index for index, value in enumerate(budget_order)}
    )
    plt.figure(figsize=(8.2, 4.5))
    palette = {"V4": "#9C755F", "V5": "#76B7B2", "V5_CCE": NAVY, "RF": GREEN, "KNN": ORANGE}
    for method in ["V4", "V5", "V5_CCE", "RF", "KNN"]:
        current = sparse_summary[sparse_summary.method.eq(method)].sort_values("budget_order")
        plt.plot(
            current.budget_order, current.macro_scaled_log_mae_mean,
            marker="o", label=method.replace("_", "+"), color=palette[method],
        )
    plt.xticks(range(len(budget_order)), ["5", "10", "20", "40", "All"])
    plt.xlabel("Visible component budget")
    plt.ylabel("Macro scaled-log MAE")
    plt.title("Completion with thinned dependency-safe context")
    plt.legend(frameon=False, ncol=3)
    savefig(figures / "sparse_observation_curve.png")

    # Sensitivity plots.
    local_factors = [0.8, 0.9, 0.95, 1.05, 1.1, 1.2]
    local = sensitivity_summary[
        sensitivity_summary.condition.isin([f"factor_{value:g}" for value in local_factors])
        & sensitivity_summary.method.isin(["V5_CCE", "RF"])
    ].copy()
    local["percent"] = local.condition.str.replace("factor_", "", regex=False).astype(float).sub(1).mul(100)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for method, color in (("V5_CCE", NAVY), ("RF", GREEN)):
        current = local[local.method.eq(method)].sort_values("percent")
        axes[0].plot(current.percent, current.prediction_shift, marker="o", color=color, label=method.replace("_", "+"))
        axes[1].plot(current.percent, current.error_delta, marker="o", color=color, label=method.replace("_", "+"))
    axes[0].set_title("Prediction shift")
    axes[1].set_title("Error increase")
    for axis in axes:
        axis.axvline(0, color="#BBBBBB", linewidth=0.8)
        axis.set_xlabel("Selected input perturbation (%)")
        axis.legend(frameon=False)
    axes[0].set_ylabel("Target-macro scaled-log value")
    fig.suptitle("Full-validation local sensitivity")
    savefig(figures / "sensitivity_dose_response.png")

    stress = sensitivity_summary[
        sensitivity_summary.condition.str.startswith("factor_")
        & sensitivity_summary.method.isin(["V5_CCE", "RF"])
    ].copy()
    stress["factor"] = stress.condition.str.replace("factor_", "", regex=False).astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    for method, color in (("V5_CCE", NAVY), ("RF", GREEN)):
        current = stress[stress.method.eq(method)].sort_values("factor")
        axes[0].plot(current.factor, current.prediction_shift, marker="o", color=color, label=method.replace("_", "+"))
        axes[1].plot(current.factor, current.perturbed_error, marker="o", color=color, label=method.replace("_", "+"))
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("Multiplicative factor on selected positive input")
        axis.legend(frameon=False)
    axes[0].set_title("Prediction shift")
    axes[1].set_title("Perturbed error")
    axes[0].set_ylabel("Target-macro scaled-log value")
    fig.suptitle("Numerical stress response")
    savefig(figures / "sensitivity_stress.png")

    # Compact tables used in both language versions.
    complete = benchmark[benchmark.partition.eq("complete_test")].copy()
    complete["method"] = pd.Categorical(
        complete.method, ["V5+CCE", "XGBoost", "RF", "KNN"], ordered=True
    )
    complete = complete.sort_values("method")
    complete_table = table(
        complete,
        ["method", "macro_scaled_log_mae", "macro_scaled_log_rmse", "macro_raw_mae"],
        ["Method", "SLMAE", "SLRMSE", "Raw MAE"],
    )
    category_display = category[category.method.isin(display_methods)].set_index("method").loc[display_methods].reset_index()
    category_display["method"] = category_display.method.replace({"V5_CCE": "V5+CCE"})
    category_table = table(
        category_display,
        ["method", "macro_purity_10", "macro_f1", "balanced_accuracy", "silhouette_cosine"],
        ["Representation", "Macro purity@10", "Macro-F1", "Balanced acc.", "Cosine silhouette"],
        3,
    )
    strict_display = strict.copy()
    strict_display["method"] = strict_display.method.astype(str).replace({"V5_CCE": "V5+CCE"})
    zero_table = table(
        strict_display,
        ["method", "macro_scaled_log_mae", "macro_scaled_log_rmse", "macro_raw_mae"],
        ["Method", "SLMAE", "SLRMSE", "Raw MAE"],
    )
    few_table = table(
        few_summary.sort_values(["shots", "initialization"]),
        ["initialization", "shots", "macro_scaled_log_mae_mean", "macro_scaled_log_mae_sd", "macro_raw_mae_mean"],
        ["Init.", "Shots", "SLMAE mean", "SLMAE SD", "Raw MAE mean"],
    )
    sparse_focus = sparse_summary[
        sparse_summary.method.isin(["V5_CCE", "RF", "KNN"])
    ].copy().sort_values(["budget_order", "method"])
    sparse_focus["method"] = sparse_focus.method.replace({"V5_CCE": "V5+CCE"})
    sparse_table = table(
        sparse_focus,
        ["method", "visible_budget", "mean_visible", "macro_scaled_log_mae_mean"],
        ["Method", "Budget", "Mean visible", "SLMAE"],
        3,
    )
    key_conditions = ["factor_0.9", "factor_1.1", "deletion", "factor_1000"]
    sensitivity_focus = sensitivity_summary[
        sensitivity_summary.method.isin(["V5_CCE", "RF"])
        & sensitivity_summary.condition.isin(key_conditions)
    ].copy()
    sensitivity_focus["method"] = sensitivity_focus.method.str.replace("_", "+")
    sensitivity_table = table(
        sensitivity_focus,
        ["method", "condition", "prediction_shift", "error_delta", "perturbed_error"],
        ["Method", "Condition", "Prediction shift", "Error delta", "Perturbed error"],
        5,
    )

    v5cce = strict[strict.method.astype(str).eq("V5_CCE")].iloc[0]
    xgb = strict[strict.method.astype(str).eq("XGBoost")].iloc[0]
    rf = strict[strict.method.astype(str).eq("RF")].iloc[0]
    knn = strict[strict.method.astype(str).eq("KNN")].iloc[0]
    few_pre = few_summary[few_summary.initialization.eq("pretrained")].set_index("shots")
    few_scratch = few_summary[few_summary.initialization.eq("scratch")].set_index("shots")
    sparse_pivot = sparse_summary.pivot(index="visible_budget", columns="method", values="macro_scaled_log_mae_mean")
    sens = sensitivity_summary.set_index(["method", "condition"])
    robustness_mean = robustness.groupby("fraction_hidden", as_index=False).mean(numeric_only=True)
    cce_category = category.set_index("method").loc["V5_CCE"]
    v4_category = category.set_index("method").loc["V4"]
    raw_category = category.set_index("method").loc["Nutrients_mask"]
    umap_overlap = umap_stability.loc[umap_stability.seed.ne(umap_stability.seed.min()), "neighbor_overlap10_vs_first"].mean()
    best_complete = complete.sort_values("macro_scaled_log_mae").iloc[0]

    zh = f"""# 显式依赖 mask 下的食物组成基础模型证据

**版本：** 2026-10-28

**主模型：** V5+CCE（dependency-mask v1）

**范围：** 新 mask 训练审计、UMAP 表征、台湾 TFDA 零样本与少样本、少观测补全、全验证集敏感性

## 摘要

这份报告重做了 commit `5eb7aeb` 的证据链，但所有补全任务都使用新的显式依赖 mask。这个 mask **同时影响训练和预测**：训练集中每一个 profile-target 样本先隐藏目标轴，以及依赖图中与它相连的祖先和后代；验证、complete test、台湾 zero-shot/few-shot、少观测与 sensitivity 使用同一个规则。新 V4、V5、V5+CCE 均从头按此规则训练，绝不是只在推理时替换 mask。

主要结果：

- 新 mask 只登记 FAO/INFOODS 定义可支持的 37 条有向关系，涉及 44 个轴。187 个监督轴中有 144 个是 singleton；平均每个目标隐藏 {policy['mean_mask_size']:.3f} 个轴，而旧语义 family 平均会隐藏 {policy['legacy_mean_family_size']:.3f} 个轴。兄弟成分仍可见。
- complete test 上，V5+CCE 的 macro scaled-log MAE 为 **{best_complete.macro_scaled_log_mae:.4f}**，在本轮四个正式方法中最低。
- 台湾 strict zero-shot 上，V5+CCE 为 **{v5cce.macro_scaled_log_mae:.4f}**，相对 XGBoost、RF、KNN 分别降低 **{pct_reduction(v5cce.macro_scaled_log_mae, xgb.macro_scaled_log_mae):.1f}%、{pct_reduction(v5cce.macro_scaled_log_mae, rf.macro_scaled_log_mae):.1f}%、{pct_reduction(v5cce.macro_scaled_log_mae, knn.macro_scaled_log_mae):.1f}%**。
- 在相同架构、目标和优化器下，预训练初始化在 8/32/128-shot 的台湾适配中分别比随机初始化降低 **{pct_reduction(few_pre.loc[8].macro_scaled_log_mae_mean, few_scratch.loc[8].macro_scaled_log_mae_mean):.1f}%、{pct_reduction(few_pre.loc[32].macro_scaled_log_mae_mean, few_scratch.loc[32].macro_scaled_log_mae_mean):.1f}%、{pct_reduction(few_pre.loc[128].macro_scaled_log_mae_mean, few_scratch.loc[128].macro_scaled_log_mae_mean):.1f}%**。
- UMAP 和定量检查显示，V5+CCE 的食品类别结构强于同一训练链中的 V4/V5，但原始数值加缺失标记仍更强；来源和观测覆盖是明显混杂因素。
- sensitivity 显示模型对 ±10% 单输入扰动平滑；删除一个可见正值后的退化需要与 RF 的干净误差一起解释。prediction shift、error delta、perturbed error 不能混为同一结论。

旧 family-mask 与新 dependency-mask 定义了不同难度的任务，因此本报告只在新 mask 内比较模型；数值不能直接用来宣称模型本身比旧报告提高了多少。

## 1. 共同设定与新 mask

### 1.1 mask 的依据和作用位置

唯一分类依据是 [FAO/INFOODS component identifiers (tagnames)](https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/) 对显式总量、分量、子总量和结构成员的定义。规则是：对目标轴隐藏 `target + all directed ancestors + all directed descendants`。没有被登记的轴独立预测；兄弟轴不因为同属一个宽泛营养类别而被隐藏。`mask_family` 列不再参与训练或评价。

实现审计：

- 37 条有向边，44 个登记轴；
- 144 个 singleton target，43 个 dependency-connected target；
- mask size 最小 1、最大 {policy['maximum_mask_size']}、平均 {policy['mean_mask_size']:.3f}；
- 234 个 sibling-visibility 检查通过；
- food name、source、food family label、profile ID 均不是模型输入。

训练时共有 337,048 个 profile-target 任务/epoch。每个任务先由目标轴调用同一个 `DependencyMaskPolicy` 构造可见 context，再进入 V4/V5/CCE 的训练 batch。推理也调用同一对象，因此不存在“训练仍用旧 family、预测才换新 mask”的情况。

### 1.2 模型与指标

V5+CCE 保持 3 层、hidden 192、6 heads、ReGLU 256、有序 axis-pair bias。V5 使用两次共享参数前向：第一遍产生 `[FOOD]`，第二遍以 `detach(h_food)` 作为 prompt；CCE 在第一遍 `[FOOD]` 的两个 dropout view 上增加对比损失。正式预测来自第二遍目标 token。

每个轴使用原训练集正值中位数 `s_a`：`z = log(1 + y/s_a)`。先在轴内计算 MAE/RMSE，再对评价轴等权平均。scaled-log MAE 是主要指标，raw MAE 保留原含量尺度。

### 1.3 新 mask 的 complete-test 基准

{complete_table}

V5+CCE 的主要指标最好；XGBoost 的 raw MAE 和 scaled-log RMSE 更低。这说明 Transformer 更均衡地改善不同量级轴上的典型误差，而树模型对原单位误差或较大对数误差仍有优势。

## 2. UMAP 与表征结构

对 75,875 个 train+validation profile 输入全部已观测数值，并使用空 target query，提取第一遍 `[FOOD]`。空 query 本身不触发某个目标的 mask；它检验的是**由新 mask 训练得到的 encoder** 的全 profile 表征。UMAP 只用于二维展示，不参与训练、选 checkpoint 或性能评价。

![新版 V5+CCE UMAP](figures/umap_v5_cce.png)

{category_table}

V5+CCE 的 macro purity@10 为 {cce_category.macro_purity_10:.3f}，高于 V4 的 {v4_category.macro_purity_10:.3f}；linear macro-F1 为 {cce_category.macro_f1:.3f}。但 raw nutrients+mask 仍达到 {raw_category.macro_f1:.3f} macro-F1，说明 FOOD embedding 没有普遍超过原始组成表示。

missingness-only 的类别 macro-F1 为 {category.set_index('method').loc['Missingness_only'].macro_f1:.3f}，来源 balanced accuracy 为 {source_metrics['balanced_accuracy']:.3f}。图中的来源簇和观测轴数量梯度因此是实质混杂，而不能全部解释为食品生物学类别。

三个 UMAP seed 相对首个 seed 的 10 邻域平均重合为 {umap_overlap:.3f}。随机隐藏 10%、30%、50% 已观测输入后，V5+CCE 与完整输入的平均 cosine similarity 分别为 {robustness_mean.set_index('fraction_hidden').loc[0.1].cosine_to_full:.3f}、{robustness_mean.set_index('fraction_hidden').loc[0.3].cosine_to_full:.3f}、{robustness_mean.set_index('fraction_hidden').loc[0.5].cosine_to_full:.3f}。UMAP 适合做表征审计，不单独构成 foundation-model 优势证据。

![表征类别定量](figures/representation_category_metrics.png)

## 3. 台湾 TFDA zero-shot

台湾数据没有参与训练、调参、checkpoint 选择或 `s_a` 估计。strict panel 含 2,212 个食品、128,470 个 food-axis 任务和 80 个轴；name-novel panel 含 1,789 个食品。所有方法都先应用新 dependency mask。RF/XGBoost/KNN 只在原训练集拟合。

![台湾 zero-shot](figures/taiwan_zero_shot.png)

{zero_table}

V5+CCE 在 strict panel 的 scaled-log MAE 最低。2,000 次台湾 profile 配对 bootstrap 中，V5+CCE 相对每个 baseline 的差值区间均由 `tables/taiwan_zero_shot_paired_bootstrap.csv` 给出。name-novel panel 得到相同排序。raw MAE 与 scaled-log 指标仍可能给出不同最优模型，应按各自定义解释。

## 4. 台湾 TFDA few-shot

只使用 name-novel 食品。固定 400 个食品作为测试集，其余 1,389 个组成适配池；每个 seed 内 8 subset 32 subset 128。比较预训练 V5+CCE 与完全相同架构的随机初始化，均训练 20 epochs、AdamW、lr `1e-4`、batch 128；固定最后一轮，不使用测试集选模型。

![台湾 few-shot](figures/taiwan_fewshot_curve.png)

{few_table}

固定测试集的冻结 zero-shot SLMAE 是 {few_zero['macro_scaled_log_mae']:.4f}。8-shot 平均为 {few_pre.loc[8].macro_scaled_log_mae_mean:.4f}，略差于 zero-shot；32-shot 为 {few_pre.loc[32].macro_scaled_log_mae_mean:.4f}，小幅改善；128-shot 为 {few_pre.loc[128].macro_scaled_log_mae_mean:.4f}，改善 {pct_reduction(few_pre.loc[128].macro_scaled_log_mae_mean, few_zero['macro_scaled_log_mae']):.1f}%。预训练相对 scratch 的优势在每个样本量和每个 seed 都成立，这是当前最直接的样本效率证据。

## 5. 少量可见 observation

从验证集选择支持量最高的 12 个目标，每个目标固定 200 个任务，共 2,400 个任务。每个任务先隐藏目标及其依赖祖先/后代，再随机保留 5、10、20、40 或全部 context；有限预算重复三次。RF/KNN 使用正式 dependency-mask 训练特征，没有额外 mask-mix augmentation。

![少观测补全](figures/sparse_observation_curve.png)

{sparse_table}

V5+CCE 在 10 和 20 个可见成分时分别得到 {sparse_pivot.loc['10','V5_CCE']:.3f} 和 {sparse_pivot.loc['20','V5_CCE']:.3f}，优于 RF/KNN；5 个成分时 V5 与 V5+CCE 接近。到 40 个和全部 context 时，RF 分别达到 {sparse_pivot.loc['40','RF']:.3f} 和 {sparse_pivot.loc['all','RF']:.3f}，优于 Transformer。这表明新任务下的优势集中在 context 真正稀疏的区间，而充分观测时 RF 很强。由于不再隐藏宽泛 nutrition family，“all” 条件可见信息通常多于旧报告，因此不能把两份 sparse 数值直接相减解释为架构提升。

## 6. 全验证集 sensitivity

分析覆盖 11,175 个 validation profile、323,809 个真实 target job 和 187 个目标。每个任务先应用 dependency mask，再从可见正值中按固定哈希选择一个输入；所有模型对同一输入作 ×0.001 至 ×1000、±5%/10%/20% 以及 deletion。另对可见零值作 `+0.1*s_a` 探针。这个设计保留全部 target job，但每个 job 只抽一个输入，因此是**全目标覆盖的配对 sampled-input sensitivity**，不是对全部 input-target pair 的穷举。

三个量分别是：prediction shift（预测移动）、error delta（扰动后误差减干净误差）、perturbed error（扰动后的绝对误差）。后两者必须结合 clean error 解读。

![局部敏感性](figures/sensitivity_dose_response.png)

{sensitivity_table}

在 ±10% 下，V5+CCE 的 prediction shift 和 error delta 都很小。deletion 条件下，V5+CCE 的 error delta 为 {sens.loc[('V5_CCE','deletion')].error_delta:.5f}，RF 为 {sens.loc[('RF','deletion')].error_delta:.5f}；扰动后误差分别为 {sens.loc[('V5_CCE','deletion')].perturbed_error:.5f} 和 {sens.loc[('RF','deletion')].perturbed_error:.5f}。这些结果描述模型响应，不能视为真实营养干预或因果效应。

![极端 stress](figures/sensitivity_stress.png)

×1000 用于发现数值失稳，不代表合理食品。所有结果均为有限值。500 次 profile 配对 bootstrap 的 CCE-RF 区间在 `tables/sensitivity_paired_bootstrap.csv` 中。

## 7. 综合判断

1. **新 mask 确实进入训练。** 任务生成、validation、complete test、台湾和扰动分析共享同一 target-specific policy；checkpoint 也记录 policy hash 和 `legacy_mask_family_used_for_masking=false`。
2. **外部迁移优势仍成立。** 台湾 zero-shot 中 V5+CCE 的主要指标优于三种 baseline，且 name-novel 子集排序一致。
3. **预训练提高样本效率。** 相同架构 scratch 对照显著更差；128-shot 还明显优于冻结 zero-shot。
4. **UMAP 是描述性证据。** CCE 表征比 V4/V5 更有类别结构，但 raw 数值+mask 更强，来源缺失模式明显。
5. **稳健性取决于条件和指标。** 小幅数值扰动总体平滑；删除与极端倍数下需要分别看预测移动、误差增量和最终误差。

最稳妥的陈述是：

> 在只屏蔽 FAO/INFOODS 可明确支持的依赖关系、并让其余轴独立预测后，重新训练的 V5+CCE 仍在台湾外部数据库上表现出清楚的 zero-shot 优势；同架构 few-shot 消融进一步表明预训练显著提高样本效率。表征和 sensitivity 分析支持模型利用了跨轴结构，同时显示数据库缺失模式和输入覆盖仍是重要混杂。

## 8. 局限

1. 新旧 mask 定义不同任务；新 mask 更窄，绝对误差不可直接用于比较架构进步。
2. 台湾只覆盖 80/187 个监督轴；名称未精确匹配不等于语义上完全未见。
3. few-shot 只有一个固定测试 split 和三个 seed；scratch 与 pretrained 使用统一超参数，而非分别调到最优。
4. UMAP 使用空 target query，展示 encoder 的全 profile 状态，不对应某一个具体补全 query。
5. sensitivity 每个 job 抽一个输入，并非穷举所有 input-target pair；扰动可能破坏真实食品的质量守恒。
6. 所有神经结果来自一条单 seed 训练链，未估计预训练 seed 间不确定性。

## 9. 复现与附件

报告只包含聚合表、图和哈希；不包含 checkpoint、原始台湾数据或逐条预测。主要附件位于 `tables/` 与 `figures/`。完整本地分析 manifest 记录数据、policy、checkpoint 和代码哈希。
"""

    en = f"""# Food-Composition Foundation Evidence under Explicit Dependency Masking

**Version:** 2026-10-28

**Primary model:** V5+CCE (dependency-mask v1)

**Scope:** mask training audit, UMAP, Taiwan TFDA zero/few-shot transfer, sparse completion, and full-validation sensitivity

## Executive summary

This report reruns the evidence chain from commit `5eb7aeb` under the explicit dependency mask. The mask affects **both training and inference**. Every training profile-target example hides the target plus its registered ancestors and descendants. Validation, complete test, Taiwan zero/few-shot, sparse completion, and sensitivity use the same rule. V4, V5, and V5+CCE were retrained from scratch under this policy; this is not an inference-only mask substitution.

Main findings:

- The registry contains 37 FAO/INFOODS-supported directed relations over 44 axes. Of 187 supervised axes, 144 are singletons. Mean mask size is {policy['mean_mask_size']:.3f}, versus {policy['legacy_mean_family_size']:.3f} for the former semantic-family mask. Siblings remain visible.
- V5+CCE reaches **{best_complete.macro_scaled_log_mae:.4f}** macro scaled-log MAE on complete test, the best primary score among the four formal methods in this round.
- On strict Taiwan zero-shot evaluation, V5+CCE scores **{v5cce.macro_scaled_log_mae:.4f}**, reducing scaled-log MAE by **{pct_reduction(v5cce.macro_scaled_log_mae, xgb.macro_scaled_log_mae):.1f}%**, **{pct_reduction(v5cce.macro_scaled_log_mae, rf.macro_scaled_log_mae):.1f}%**, and **{pct_reduction(v5cce.macro_scaled_log_mae, knn.macro_scaled_log_mae):.1f}%** relative to XGBoost, RF, and KNN.
- With the same architecture, objective, and optimizer, pretrained initialization reduces Taiwan few-shot error versus scratch by **{pct_reduction(few_pre.loc[8].macro_scaled_log_mae_mean, few_scratch.loc[8].macro_scaled_log_mae_mean):.1f}%**, **{pct_reduction(few_pre.loc[32].macro_scaled_log_mae_mean, few_scratch.loc[32].macro_scaled_log_mae_mean):.1f}%**, and **{pct_reduction(few_pre.loc[128].macro_scaled_log_mae_mean, few_scratch.loc[128].macro_scaled_log_mae_mean):.1f}%** at 8, 32, and 128 shots.
- UMAP and quantitative probes show stronger food-category structure in V5+CCE than in V4/V5, while raw numerical values plus missingness remain stronger. Source-specific coverage is a material confounder.
- Responses to single-input ±10% perturbations are smooth. Deletion and stress conclusions depend on clean error, prediction shift, error increase, and final perturbed error; no one quantity is sufficient.

The old family mask and the new dependency mask define tasks of different difficulty. All comparisons in this report are therefore within the new policy; absolute values should not be used to claim architectural improvement over the old report.

## 1. Shared setup and the new mask

### 1.1 Authority and where masking is applied

The sole authority is the [FAO/INFOODS component-identifier (tagname) standard](https://www.fao.org/infoods/infoods/standards-guidelines/food-component-identifiers-tagnames/en/) for explicit totals, components, subtotals, and structural members. For target `a`, the hidden set is `a + all directed ancestors + all directed descendants`. Unregistered axes are predicted independently. Sibling components stay visible, and the legacy `mask_family` column is unused.

Audit facts:

- 37 directed edges over 44 registered axes;
- 144 singleton and 43 dependency-connected targets;
- mask size 1 to {policy['maximum_mask_size']}, mean {policy['mean_mask_size']:.3f};
- 234 sibling-visibility checks passed;
- no food name, source, family label, or profile identifier is a model input.

Training has 337,048 profile-target tasks per epoch. Every task calls the same `DependencyMaskPolicy` before batching. Evaluation uses the identical object. The checkpoint records the policy hash and `legacy_mask_family_used_for_masking=false`.

### 1.2 Model and metrics

V5+CCE uses three layers, hidden size 192, six heads, ReGLU width 256, and ordered axis-pair bias. V5 executes the shared model twice: pass one creates `[FOOD]`, and pass two uses `detach(h_food)` as its prompt. CCE adds a contrastive objective between two dropout views of the first-pass `[FOOD]`. Official predictions come from second-pass target tokens.

For each axis, `z = log(1 + y/s_a)`, where `s_a` is the positive training median. Errors are averaged within axis and then equally across axes. Macro scaled-log MAE is primary; raw MAE retains the original concentration scale.

### 1.3 Complete-test benchmark under the new mask

{complete_table}

V5+CCE has the best primary metric. XGBoost has lower raw MAE and scaled-log RMSE. The Transformer improves typical, scale-balanced error, while the tree baselines retain advantages for raw units or larger log-space misses.

## 2. UMAP and representation structure

The first-pass `[FOOD]` state was extracted for 75,875 train+validation profiles using all observed numerical values and an empty target grid. The empty grid does not invoke one target-specific mask; it audits the full-profile encoder learned under the new training policy. UMAP is descriptive only and was not used for training or checkpoint selection.

![New V5+CCE UMAP](figures/umap_v5_cce.png)

{category_table}

V5+CCE reaches {cce_category.macro_purity_10:.3f} macro purity@10 versus {v4_category.macro_purity_10:.3f} for V4, with linear macro-F1 {cce_category.macro_f1:.3f}. Raw nutrients plus mask still reach {raw_category.macro_f1:.3f}, so the FOOD embedding is not uniformly more category-informative than its numerical inputs.

Missingness alone reaches {category.set_index('method').loc['Missingness_only'].macro_f1:.3f} category macro-F1 and {source_metrics['balanced_accuracy']:.3f} source balanced accuracy. Visible source clusters and observation-count gradients are therefore material coverage confounding.

Across UMAP seeds, mean ten-neighbour overlap with the first seed is {umap_overlap:.3f}. After hiding 10%, 30%, and 50% of observed inputs, mean cosine similarity to the full V5+CCE representation is {robustness_mean.set_index('fraction_hidden').loc[0.1].cosine_to_full:.3f}, {robustness_mean.set_index('fraction_hidden').loc[0.3].cosine_to_full:.3f}, and {robustness_mean.set_index('fraction_hidden').loc[0.5].cosine_to_full:.3f}. UMAP supports representation auditing, not a stand-alone foundation-model claim.

![Representation category metrics](figures/representation_category_metrics.png)

## 3. Taiwan TFDA zero-shot transfer

Taiwan data are excluded from training, tuning, checkpoint selection, and scale estimation. The strict panel contains 2,212 foods, 128,470 food-axis jobs, and 80 axes. The name-novel panel contains 1,789 foods. Every method applies the new dependency mask. RF, XGBoost, and KNN are fit only on the original training partition.

![Taiwan zero-shot](figures/taiwan_zero_shot.png)

{zero_table}

V5+CCE has the lowest strict-panel scaled-log MAE. The 2,000 profile-paired bootstrap intervals against each baseline are in `tables/taiwan_zero_shot_paired_bootstrap.csv`; the name-novel panel preserves the same ranking. Raw and scaled-log metrics can still identify different best models and should be interpreted according to their definitions.

## 4. Taiwan TFDA few-shot adaptation

Only name-novel foods are used. A fixed 400-food test set is held out from an adaptation pool of 1,389 foods. Within each seed, 8 subset 32 subset 128. Pretrained V5+CCE is compared with the same architecture from random initialization. Both use 20 epochs, AdamW, learning rate `1e-4`, batch size 128, and the fixed final epoch without test-based selection.

![Taiwan few-shot](figures/taiwan_fewshot_curve.png)

{few_table}

Frozen zero-shot scaled-log MAE on the fixed test set is {few_zero['macro_scaled_log_mae']:.4f}. The pretrained branch scores {few_pre.loc[8].macro_scaled_log_mae_mean:.4f}, {few_pre.loc[32].macro_scaled_log_mae_mean:.4f}, and {few_pre.loc[128].macro_scaled_log_mae_mean:.4f}. Eight-shot adaptation is slightly worse than zero-shot; 32-shot is a small gain; 128-shot improves by {pct_reduction(few_pre.loc[128].macro_scaled_log_mae_mean, few_zero['macro_scaled_log_mae']):.1f}%. Pretraining beats scratch for every shot and seed, providing the most direct sample-efficiency evidence.

## 5. Completion with few visible observations

The experiment selects the 12 most supported validation targets and 200 jobs per target. Each job first hides its target and dependency ancestors/descendants, then keeps 5, 10, 20, 40, or all remaining context values. Finite budgets use three repeats. RF/KNN use their standard dependency-mask training features without extra mask-mix augmentation.

![Sparse observation completion](figures/sparse_observation_curve.png)

{sparse_table}

V5+CCE scores {sparse_pivot.loc['10','V5_CCE']:.3f} and {sparse_pivot.loc['20','V5_CCE']:.3f} with 10 and 20 visible components and outperforms RF/KNN there; V5 and V5+CCE are close at five components. With 40 and all context values, RF scores {sparse_pivot.loc['40','RF']:.3f} and {sparse_pivot.loc['all','RF']:.3f} and becomes the strongest method. The Transformer advantage is therefore concentrated in genuinely sparse context, while RF is strong when context is well observed. The all-context condition exposes more information than the former broad-family task, so the two reports' sparse scores are not direct architecture comparisons.

## 6. Full-validation sensitivity

The audit covers 11,175 validation profiles, 323,809 observed target jobs, and all 187 targets. After dependency masking, one visible positive input is selected deterministically for each eligible job. All models receive the same selected input under factors 0.001 to 1000, ±5%/10%/20%, and deletion. A separate visible-zero probe sets one selected zero to `0.1*s_a`. This provides full target-job coverage with paired sampled inputs; it is not an exhaustive enumeration of every input-target pair.

Prediction shift, error delta, and perturbed error are reported separately. Error delta is perturbed minus clean error; perturbed error includes both baseline accuracy and additional degradation.

![Local sensitivity](figures/sensitivity_dose_response.png)

{sensitivity_table}

Under ±10% perturbations, V5+CCE prediction shifts and error increases are small. Under deletion, its error delta is {sens.loc[('V5_CCE','deletion')].error_delta:.5f} versus {sens.loc[('RF','deletion')].error_delta:.5f} for RF; perturbed errors are {sens.loc[('V5_CCE','deletion')].perturbed_error:.5f} and {sens.loc[('RF','deletion')].perturbed_error:.5f}. These are model-response diagnostics, not nutritional interventions or causal effects.

![Stress response](figures/sensitivity_stress.png)

Factor 1000 is a numerical stress condition rather than a plausible food. All outputs remain finite. Five hundred profile-paired bootstrap intervals for CCE-minus-RF are provided in `tables/sensitivity_paired_bootstrap.csv`.

## 7. Integrated assessment

1. **The new mask is part of training.** Task generation and every evaluation path share one target-specific policy.
2. **External transfer remains strong.** V5+CCE leads the three baselines on Taiwan scaled-log MAE, including the name-novel subset.
3. **Pretraining improves sample efficiency.** Same-architecture scratch is consistently worse, and 128-shot adaptation clearly beats frozen zero-shot.
4. **UMAP evidence is descriptive.** CCE strengthens category structure, while raw inputs remain stronger and source missingness is highly predictive.
5. **Robustness depends on condition and metric.** Local responses are smooth; deletion and extreme multipliers require separate examination of movement, degradation, and final error.

A defensible summary is:

> After restricting masking to explicit FAO/INFOODS-supported dependencies and predicting all other axes independently, the retrained V5+CCE still shows clear zero-shot transfer to the Taiwan database. Same-architecture few-shot ablation further shows that pretraining materially improves sample efficiency. Representation and sensitivity audits confirm meaningful cross-axis use while identifying database missingness and context coverage as important confounders.

## 8. Limitations

1. The old and new masks define different tasks. The narrower new task has lower absolute difficulty, so cross-report score changes are not architectural gains.
2. Taiwan covers 80 of 187 supervised axes. Exact name novelty does not prove semantic novelty.
3. Few-shot evaluation uses one fixed test split and three seeds. Shared hyperparameters control the comparison but are not separately optimal.
4. UMAP uses an empty query grid and describes the encoder's full-profile state rather than a specific completion query.
5. Sensitivity samples one input per job rather than every input-target pair, and perturbations may violate food chemistry or mass balance.
6. Neural results come from one pretrained seed, so pretraining-seed uncertainty is unmeasured.

## 9. Reproducibility and attachments

The report package contains only aggregate tables, figures, and hashes. It excludes checkpoints, raw Taiwan data, and row-level predictions. Analysis manifests record data, policy, checkpoint, and code hashes.
"""

    (args.output_dir / "report_zh.md").write_text(zh, encoding="utf-8")
    (args.output_dir / "report_en.md").write_text(en, encoding="utf-8")
    readme = """# Dependency-mask foundation evidence report\n\nBilingual aggregate report generated from the dependency-mask v1 checkpoints and analyses. No model weights, raw data, or row-level predictions are included.\n\n- `report_zh.md` / `report_en.md`: source reports\n- `food_foundation_evidence_dependency_mask_zh.pdf` / `_en.pdf`: rendered reports\n- `figures/`: report figures\n- `tables/`: aggregate results\n"""
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    manifest = {
        "status": "aggregate_report_sources_complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "dependency-mask v1; UMAP, Taiwan zero/few-shot, sparse completion, sensitivity",
        "raw_data_included": False, "model_weights_included": False,
        "row_level_predictions_included": False,
        "source_hashes": {
            "policy_manifest": digest(args.main_results_dir / "manifest.json"),
            "representation_manifest": digest(representation / "manifest.json"),
            "taiwan_zero_shot_manifest": digest(zero / "manifest.json"),
            "taiwan_fewshot_manifest": digest(fewshot / "manifest.json"),
            "sparse_manifest": digest(sparse / "manifest.json"),
            "sensitivity_manifest": digest(sensitivity / "manifest.json"),
        },
    }
    write_json(args.output_dir / "report_manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
