"""Build bilingual LR6E4 evidence drafts only after completed, audited analysis and fit.

No model execution, raw numeric profile access, scientific acceptance, or implicit
figure review. The prior reviewed report remains immutable and is incorporated
as historical context. New prose still requires actual review before delivery.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / 'experiments/foodnutrigpt_v9_research/report_snapshot_r9_capacity256_v1'
ANALYSIS = ROOT / 'reports/v9_r9_lr6e4_analysis_v1'
FIT = ROOT / 'reports/v9_r9_lr6e4_fit_v1'
PLAN = ROOT / 'experiments/foodnutrigpt_v9_research/r9/lr6e4_v1'
METRICS = ['scaled_log_mae', 'log_mae', 'raw_mae', 'positive_scaled_log_mae', 'zero_scaled_log_mae']
RETRIEVAL = ['recall_at_1', 'recall_at_5', 'recall_at_10', 'mrr']


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def verify_hashes(mapping):
    for name, expected in mapping.items():
        if digest(ROOT / name) != expected.lower():
            raise ValueError('Changed evidence: ' + name)


def ready():
    requirements = {
        ANALYSIS / 'summary.json': 'complete_registered_lr6e4_analysis',
        ANALYSIS / 'status.json': 'complete', FIT / 'summary.json': 'complete',
        FIT / 'status.json': 'complete',
        PRIOR / 'review_verification.json': 'reviewed_complete_capacity256_stage_report',
        ROOT / 'reports/v9_r9_lr6e4_functional_v1/verification.json': 'complete',
        PLAN / 'config.json': 'registered',
        PLAN / 'launch.json': 'verified_launched_with_scheduled_followup',
        ROOT / 'reports/v9_r9_lr6e4_axis_changes_v1/summary.json': 'complete_aggregate_axis_contrast',
        ROOT / 'reports/v9_r9_lr6e4_cases_v1/summary.json': 'complete_existing_lr6e4_case_selection_and_value_check',
    }
    return [str(path.relative_to(ROOT)) for path, state in requirements.items()
            if not path.exists() or read(path).get('status') != state]


def number(value):
    if not math.isfinite(value):
        raise ValueError('Nonfinite report metric')
    return f'{value:.6f}'


def table(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['|' + '|'.join(map(str, row)) + '|' for row in rows])


def relative(path, output):
    return Path(os.path.relpath(path, output)).as_posix()


def rebase(text, origin, output):
    def replace(match):
        target = match[2]
        if re.match(r'^[a-z]+:|^#|^/', target):
            return match[0]
        path, separator, anchor = target.partition('#')
        absolute = (origin.parent / path).resolve()
        if not absolute.exists():
            raise FileNotFoundError(absolute)
        # Counterpart/evidence links point to the new snapshot rather than history.
        if path in ['REPORT_ZH.md', 'REPORT_EN.md', 'evidence.json']:
            return match[0]
        return f'[{match[1]}]({relative(absolute, output)}{separator}{anchor})'
    return re.sub(r'\[([^\]\n]+)\]\(([^)\n]+)\)', replace, text)


def tables(analysis, fit):
    records, comparisons = analysis['records'], analysis['comparisons']
    scores = {key.upper(): record['metrics'] for key, record in records.items()}
    retrieval = {key.upper(): record['retrieval'] for key, record in records.items()}
    for reference in ['rf32', 'xgb32']:
        scores[reference.upper()] = {task: comparisons[reference][task]['scores']['baseline']
                                    for task in ['completion', 'name_only']}
        directory = ROOT / records['lr6e4']['manifest']['spec'][
            'rf_reference' if reference == 'rf32' else 'xgb_reference']
        if digest(directory / 'retrieval/metrics.json') != comparisons[reference]['retrieval']['hashes']['baseline']['metrics']:
            raise ValueError('Frozen reference retrieval metrics changed')
        retrieval[reference.upper()] = read(directory / 'retrieval/metrics.json')
    scores['KNN32'] = {'name_only': comparisons['knn32']['name_only']['scores']['baseline']}
    knn_path = ROOT / 'output/v9_r7/exact_name_knn32/metrics.json'
    if digest(knn_path) != comparisons['knn32']['retrieval']['hashes']['baseline']['metrics']:
        raise ValueError('Frozen KNN retrieval metrics changed')
    retrieval['KNN32'] = read(knn_path)
    result = {}
    for task in ['completion', 'name_only']:
        rows = [[method, *[number(score[task]['nutrition'][metric]) for metric in METRICS]]
                for method, score in scores.items() if task in score]
        result[task] = table(['Method', '142-axis scaled-log MAE', 'Legacy log-MAE',
                              'Raw MAE (g/100g)', 'Positive MAE', 'Explicit-zero MAE'], rows)
    result['subsets'] = table(['Method', 'Task', 'Subset', 'Axes', 'Scaled-log MAE'],
        [[method, task, subset, score[task][subset]['axes'], number(score[task][subset]['scaled_log_mae'])]
         for method, score in scores.items() for task in score for subset in ['food_metabolome', 'all']])
    result['retrieval'] = table(['Method', 'Visible fraction', 'Recall@1', 'Recall@5', 'Recall@10', 'MRR'],
        [[method, str(row['visible_fraction']), *[number(row[key]) for key in RETRIEVAL]]
         for method, metric in retrieval.items() for row in metric['metrics']])
    intervals = []
    for reference, tasks in comparisons.items():
        for task in ['completion', 'name_only']:
            if task not in tasks:
                continue
            for metric in ['scaled_log_mae', 'log_mae']:
                row = tasks[task]['paired_intervals'][metric]
                intervals.append([reference, task, metric, number(100 * row['relative_improvement']),
                                  *[number(100 * value) for value in row['relative_improvement_95_interval']]])
    result['intervals'] = table(['Reference', 'Task', 'Metric', 'LR6E4 gain (%)', '95% lower (%)', '95% upper (%)'], intervals)
    result['retrieval_intervals'] = table(['Reference', 'Visible fraction', 'Metric', 'LR6E4 minus reference', '95% lower', '95% upper'],
        [[reference, fraction, key, number(row['candidate_minus_baseline']), *[number(value) for value in row['difference_95_interval']]]
         for reference, tasks in comparisons.items() for fraction, metrics in tasks['retrieval']['comparisons'].items()
         for key, row in metrics.items()])
    result['decomposition'] = table(['Fixed-denominator component', 'LR6E4 minus MAE'],
        [[key, number(value)] for key, value in analysis['fixed_denominator_lr6e4_minus_mae'].items()])
    result['fit'] = table(['Method', 'Partition', '142-axis MAE', 'Positive MAE', 'Explicit-zero MAE'],
        [[method.upper(), partition, *[number(data[method]['nutrition'][key]) for key in
                                      ['scaled_log_mae', 'positive_scaled_log_mae', 'zero_scaled_log_mae']]]
         for method in ['mae', 'lr6e4'] for partition, data in
         [('train', fit['source_free_training_fit']), ('validation', fit['source_free_validation'])]])
    result['clipping'] = table(['Method', 'First 10 clip fraction', 'Last 10 clip fraction', 'Selected epoch', 'Elapsed (s)'],
        [[role.upper(), number(row['first10_clip_fraction']), number(row['last10_clip_fraction']),
          row['manifest']['best_epoch'], f"{row['manifest']['elapsed_seconds']:.3f}"]
         for role, row in records.items()])
    result['gates'] = table(['Registered screening gate vs same-seed MAE', 'Pass'],
                           [[key, str(bool(value))] for key, value in analysis['screening_gates'].items()])
    return result


def supplement(lang, analysis, fit, shared, output):
    zh = lang == 'ZH'
    words = lambda chinese, english: chinese if zh else english
    titles = [
        words('研究问题与预先假设', 'Research question and preregistered hypothesis'),
        words('父版本与唯一方法改动', 'Parent and sole method intervention'),
        words('可复现信息与实际成本', 'Reproducibility and measured cost'),
        words('完整三任务结果与固定参照', 'All three tasks and fixed references'),
        words('机制诊断与功能核验', 'Mechanism diagnostics and functional checks'),
        words('因果解释边界', 'Limits of causal interpretation'),
        words('筛选与版本决定', 'Screening and version decision'),
        words('下一轮问题与测试状态', 'Next question and test status')]
    sub = lambda i: f'### 14.{i}. {titles[i-1]}'
    record, parent = [analysis['records'][key]['manifest'] for key in ['lr6e4', 'mae']]
    plan = read(PLAN / 'config.json')
    launch = read(PLAN / 'launch.json')
    functional_path = ROOT / 'reports/v9_r9_lr6e4_functional_v1/verification.json'
    functional = read(functional_path)
    failure_path = ROOT / 'reports/v9_r9_lr6e4_launch_failure_v1/record.json'
    failure = read(failure_path)
    smoke = functional['small_batch_overfit']
    axes_path = ROOT / 'reports/v9_r9_lr6e4_axis_changes_v1/summary.json'
    cases_path = ROOT / 'reports/v9_r9_lr6e4_cases_v1/summary.json'
    axes, cases = read(axes_path), read(cases_path)
    verify_hashes(axes['input_hashes'])
    verify_hashes(cases['input_hashes'])
    if (record['spec'] != dict(parent['spec'], name=record['candidate'], learning_rate=.0006)
            or record['initial_state_sha256'] != parent['initial_state_sha256']
            or failure['optimizer_steps'] != 0 or failure['method_registration_changed']
            or launch['plan_sha256'] != digest(PLAN / 'config.json')
            or cases['candidate'] != record['candidate']):
        raise ValueError('Declared learning-rate control or operational evidence differs')
    eligible = all(analysis['screening_gates'].values())
    parts = [
        words('## 14. 第7项候选：192维模型学习率3e-4→6e-4',
              '## 14. Candidate 7: width-192 learning rate 3e-4 to 6e-4'),
        words('本节为已完成实验的机器证据草稿；实际曲线、逐轴、案例解读和版本决定尚待审阅。不得将草稿当作完整版本或最终模型接受。',
              'This section is a machine-evidence draft for the completed experiment. Actual curve, axis and case interpretation and the version decision still require review. A draft does not constitute a complete version or final model acceptance.'),
        sub(1), words(
            '上一容量组合在训练与验证集均退步，不能直接解释为更好的训练拟合导致过拟合。本轮回到192维MAE控制，检验固定60轮预算下增加更新幅度是否改善优化与泛化。此前1e-4到3e-4的对照仅提供动机，不证明6e-4必然更优。',
            'The previous capacity group worsened both training and validation fit, so it does not directly support better training fit followed by overfitting. This experiment returns to the width-192 MAE control and tests whether larger updates improve optimization and validation performance within 60 epochs. The earlier 1e-4 versus 3e-4 contrast motivates the test but does not establish that 6e-4 will help.'),
        sub(2), words(
            '唯一登记因素为初始学习率3e-4→6e-4，完整绝对余弦学习率日程加倍，最低学习率比例仍为.01。模型为192维、3层、6头、FF768、dropout .15、rank16、直接MAE与来源校准权重1；AdamW weight_decay1e-4、batch64、clip1、种子20260922、全部家族任务和60轮预算不变。相同初始化、参数量和任务顺序已核验。AdamW每步解耦衰减幅度同样随学习率改变，不能把干预称作仅初始一步或与正则化无关的速度变化。',
            'The sole registered factor is initial learning rate 3e-4 to 6e-4: the entire absolute cosine schedule doubles while its minimum-to-initial ratio remains .01. Width 192, three layers, six heads, FF768, dropout .15, rank16, direct MAE, source-calibration weight 1, AdamW weight_decay 1e-4, batch 64, clip 1, seed 20260922, all family tasks and the 60-epoch budget stay fixed. Initialization, parameter counts and task orders match. AdamW decoupled shrinkage per step also changes with learning rate; the intervention is neither restricted to the first step nor independent of regularization.'),
        sub(3), table(['Item', 'Value'], [
            ['Code commit', record['code_commit']],
            ['Checkpoint SHA256', record['checkpoint_sha256']],
            ['Shared initial-state SHA256', record['initial_state_sha256']],
            ['Data SHA256', record['data_hash']], ['Panel SHA256', record['panel_hash']],
            ['Name cache SHA256', record['name_cache_hash']],
            ['Total / requires_grad parameters', f"{record['parameter_count']} / {record['trainable_parameter_count']}"],
            ['Seed', record['seed']], ['Selected epoch', record['best_epoch']],
            ['Completed epochs', record['epoch_completed']], ['Run elapsed (s)', f"{record['elapsed_seconds']:.3f}"],
            ['Full training-fit inference (s)', f"{fit['elapsed_seconds']:.3f}"],
            ['Pre-run estimate (s)', f"{plan['estimated_training_seconds']:.3f}"],
            ['Follow-up interval (min)', plan['followup_delay_minutes']],
            ['Pretraining failed-launch optimization steps', failure['optimizer_steps']]]),
        words(
            '环境保持Windows、Python 3.10.19、PyTorch 2.7.1+cu128及RTX 5070 Ti 16 GB。依据同机完整192维60轮运行较慢耗时乘1.1估时，另预留10分钟后处理，并安排235分钟回访，没有epoch观察器。首次控制器因Windows PowerShell无法识别Get-FileHash而在训练前退出，未创建训练输出或执行优化；确认进程退出后，使用已验证的PowerShell 7.6.5和相同控制器重新启动，保存两次日志。上表训练用时不包含该失败启动与人工准备时间。',
            'The environment remains Windows, Python 3.10.19, PyTorch 2.7.1+cu128 and RTX 5070 Ti 16 GB. Runtime was estimated as 1.1 times the slower completed width-192 60-epoch reference, with ten additional minutes for postprocessing and a 235-minute follow-up rather than an epoch observer. The first controller exited before training because Windows PowerShell could not resolve Get-FileHash; no training output or optimizer update was created. After confirming process exit, the identical controller was launched using verified PowerShell 7.6.5, preserving both logs. Reported training duration excludes this failed launch and manual preparation.'),
        f"[Registration]({relative(PLAN/'PLAN.md',output)}) · [Configuration]({relative(PLAN/'config.json',output)}) · [Version]({relative(PLAN/'README.md',output)}) · [Launch]({relative(PLAN/'launch.json',output)}) · [Failed launch]({relative(failure_path,output)}) · [Analysis]({relative(ANALYSIS/'summary.json',output)}) · [Training fit]({relative(FIT/'summary.json',output)})",
        sub(4), words(
            'MAE是同种子3e-4控制；LR6E4是本次6e-4候选。RF/XGB/KNN均读取冻结产物。每个神经模型的补全、仅名称与检索来自同一个按验证补全主指标选择的检查点。第6节MAE三种子结果与本节单种子对照不能混用。',
            'MAE denotes the same-seed 3e-4 control and LR6E4 the 6e-4 candidate. RF/XGB/KNN artifacts are frozen. Completion, name-only and retrieval for each neural model use one checkpoint selected by validation completion primary MAE. The three-seed MAE result in Section 6 must not be mixed with this single-seed comparison.'),
        '**补全 / Completion**', shared['completion'], '**仅名称 / Name-only**', shared['name_only'],
        '**45/187 axes**', shared['subsets'], '**检索 / Retrieval**', shared['retrieval'],
        words(
            '固定49,913个候选名称，名称向量由名称预测营养得到，不读取候选真实营养；营养查询不含名称。正确答案按原始名称精确匹配，尚无已确认别名映射。区间以食物候选组为单位并条件于这些选定检查点；不覆盖训练种子总体、标签真实性或反复选择的不确定性。以下营养相对改善为正表示误差下降，检索差值为正表示候选排名指标提高。',
            'The fixed library has 49,913 names; candidate vectors are nutrient predictions from names, without measured candidate nutrition. Nutrient queries contain no names. Correct answers use exact original names with no confirmed alias map. Intervals resample food candidate groups and condition on these selected checkpoints; they exclude seed-population, label-validity and repeated-selection uncertainty. Positive nutrient improvement means lower error; positive retrieval differences mean a higher ranking metric.'),
        shared['intervals'], shared['retrieval_intervals'],
        sub(5), shared['decomposition'], shared['fit'], shared['clipping'],
        f"![Learning rate 3e-4 versus 6e-4 learning curves]({relative(ANALYSIS/'learning_curves.png',output)})",
        words(
            '正值低估、高估及显式零采用同一主指标分母，精确重构主误差变化；条件正值/零值平均的分母不同，不能直接相加。全部1,828,536训练目标使用无来源残差的基础推理，既有MAE训练预测只重新评分。本节数字尚不能替代实际曲线与案例审阅。',
            'Positive underprediction, overprediction and explicit-zero contributions share the primary metric denominator and reconstruct its change exactly. Conditional positive/zero averages use different denominators and cannot simply be added. All 1,828,536 training targets use source-free base inference; existing MAE predictions are only rescored. These numbers do not substitute for actual curve and case review.'),
        table(['Aggregate axis check', 'Count'], [
            ['Nutrition axes', axes['axis_count']],
            ['Axes point better than MAE', axes['axes_point_better_than_mae']],
            ['Unadjusted interval supports improvement', axes['axes_unadjusted_interval_supports_improvement']],
            ['Unadjusted interval supports regression', axes['axes_unadjusted_interval_supports_regression']],
            ['Validation support below30', axes['sparse_validation_axes_below30']],
            ['Axes point better than frozen RF', axes['lr6e4_axes_better_than_rf']],
            ['Amino-acid axes point worse than MAE', axes['amino_acid_axes_lr6e4_worse_than_mae']]]),
        f"[All axis contrasts]({relative(axes_path.parent/'axis_changes.csv',output)}) · [All source/family/support partitions]({relative(ANALYSIS/'lr6e4_minus_mae_partitions.csv',output)}) · [Case checks]({relative(cases_path,output)})",
        words(
            '轴区间没有多重检验校正；稀疏轴须连同支持数解读。40条案例是对冻结RF误差差值的两端，已核对键、标签、预测和元数据；不代表总体或独立样本，不据此修正或排除标签。原始食物级数值仅保存在本地。',
            'Axis intervals are unadjusted for multiple comparisons; sparse axes require support-aware interpretation. The 40 cases are the two tails of error differences against fixed RF, with keys, labels, predictions and metadata checked. They are neither representative nor independent samples and do not justify relabeling or exclusion. Individual numeric profiles remain local.'),
        table(['Functional exercise', 'Value'], [
            ['Training batch tasks', 32], ['Diagnostic steps', smoke['steps']],
            ['Loss before', number(smoke['before'])], ['Loss after', number(smoke['after'])]]),
        words(
            '预训练42项检查通过，覆盖共享训练、学习率干预及分析边界。真实训练行GPU预检确认父初值、前向、初始损失/梯度和RNG一致；隐藏标签及来源不影响基础前向、未观测轴可查询、8个训练名称×187轴输出有限。初始和已学模型均精确重载。最初未设置PYTHONPATH导致测试收集失败，修正命令后通过；上述诊断不是正式候选性能。',
            'The 42 pretraining checks cover shared training, the learning-rate intervention and analysis boundaries. Real training-row GPU checks match parent initialization, forward outputs, initial loss/gradients and RNG. Hidden labels and source do not affect base forward predictions; unobserved axes can be queried and 8 training names by 187 axes yield finite outputs. Initial and learned models reload exactly. Initial test collection failed because PYTHONPATH was unset; the corrected command passed. These diagnostics do not measure formal candidate performance.'),
        sub(6), words(
            '本次可比较的是整个学习率日程幅度干预，包含随学习率变化的AdamW更新与衰减，并非唯一数值机制的证明。相同初值和任务顺序不能消除后续优化路径或GPU非确定性差异。训练与验证的食物和支持不同，拟合差距不能单独证明过拟合。实际审阅需分别写明已观察事实、对照支持的解释与未排除的替代解释。',
            'The comparison identifies the full learning-rate schedule amplitude intervention, including its AdamW updates and shrinkage, rather than a unique numerical mechanism. Matching initialization and task orders does not remove subsequent trajectory differences or GPU nondeterminism. Training and validation differ in foods and support, so their fit gap alone cannot prove overfitting. Actual interpretation must separate observed facts, controlled evidence and remaining alternatives.'),
        sub(7), shared['gates'],
        words(('满足' if eligible else '未满足')+'三个预注册筛选条件；仍需实际审阅形成版本决定。单种子不能确认稳定改进或超越RF。',
              ('Meets' if eligible else 'Does not meet')+' all three registered screening conditions; the version decision still requires actual review. One seed cannot confirm stable improvement or superiority over RF.'),
        words(
            '通过筛选仅允许另行登记20260923/24，不自动启动或改变现有门槛。最终接受需三种子与固定RF的共同协议证据。失败或无改善同样形成完整版本，不能删去。',
            'Passing only permits separately registered seeds 20260923/24; it does not automatically launch them or change the gates. Final acceptance requires three-seed evidence against fixed RF under the common protocol. Failures and null improvements must also remain complete documented versions.'),
        sub(8), words(
            '下一项最小对照或复验由完整审阅决定。数据、树参照和测试状态保持冻结；来源留出、少样本迁移及foundation model能力仍未被证明。',
            'The next minimal contrast or replication follows complete review. Data, tree references and test status stay frozen; source-held-out transfer, few-shot transfer and foundation-model capabilities remain unproven.')
    ]
    return '\n\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    pending = ready()
    if args.check_only:
        print(json.dumps({'ready': not pending, 'pending': pending, 'output_written': False, 'training_performed': False}))
        return
    if pending:
        raise ValueError('Wait for complete learning-rate analysis and training-fit diagnosis: ' + ', '.join(pending))
    if args.output_dir is None or args.output_dir.exists():
        raise ValueError('A new report directory is required')
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / 'experiments/foodnutrigpt_v9_research').resolve()):
        raise ValueError('Report snapshot must be inside the research experiment directory')
    analysis, fit = [read(path / 'summary.json') for path in [ANALYSIS, FIT]]
    prior_review = read(PRIOR / 'review_verification.json')
    for path, expected in prior_review['document_hashes'].items():
        if digest(PRIOR / path) != expected:
            raise ValueError('Prior reviewed document changed')
    for item in [analysis, fit]:
        verify_hashes(item['input_hashes'])
        if any(item[key] for key in ['training_performed', 'data_modified', 'baseline_refit', 'complete_test_opened']):
            raise ValueError('Evidence scope changed')
    for name, expected in analysis['figure_hashes'].items():
        if digest(ANALYSIS / name) != expected:
            raise ValueError('Analysis figure changed')
    if (not analysis['single_factor_controls_verified'] or fit['training_job_count'] != 1828536
            or not fit['existing_mae_predictions_rescored_exact'] or not fit['new_lr6e4_predictions_disk_reloaded_exact']):
        raise ValueError('Expected completed controls and fit replay')
    for role in ['mae', 'lr6e4']:
        if fit['source_free_validation'][role] != analysis['records'][role]['metrics']['completion']:
            raise ValueError('Fit and analysis validation metrics differ')
    shared = tables(analysis, fit)
    documents, supplements = {}, {}
    for lang in ['ZH', 'EN']:
        prior = rebase((PRIOR / f'REPORT_{lang}.md').read_text(encoding='utf-8'), PRIOR / f'REPORT_{lang}.md', output)
        prior_scope = prior.splitlines()[4]
        scope = ('草稿：第1–13节保留上一版已审阅的历史证据，其中“最新”和“下一轮”均描述历史时点；第14节新增学习率3e-4到6e-4实验的证据。本轮实际曲线审阅、因果解释与版本决定尚未完成，不能称为最终报告。' if lang == 'ZH' else
                 'DRAFT: Sections 1–13 retain previously reviewed history; their latest/next experiment statements describe that historical stage. Section 14 adds completed learning-rate 3e-4 to 6e-4 evidence. Actual curve review, interpretation and the version decision remain incomplete; this is not a final report.')
        prior = prior.replace(prior_scope, scope, 1)
        supplements[lang] = supplement(lang, analysis, fit, shared, output)
        documents[lang] = prior.rstrip() + '\n\n' + supplements[lang] + '\n'
    # Each language uses the exact same newly generated numeric table rows.
    for shared_table in shared.values():
        if any(content.count(shared_table) != 1 for content in supplements.values()):
            raise ValueError('New bilingual table content differs or is duplicated')
    checked_links = 0
    for content in documents.values():
        for target in re.findall(r'\]\(([^)\n]+)\)', content):
            if re.match(r'^[a-z]+:|^#|^/', target):
                continue
            path = target.partition('#')[0]
            if path not in ['REPORT_ZH.md', 'REPORT_EN.md', 'evidence.json'] and not (output / path).exists():
                raise FileNotFoundError(output / path)
            checked_links += 1
    paths = [ANALYSIS / 'summary.json', FIT / 'summary.json',
             PRIOR / 'review_verification.json', PRIOR / 'REPORT_ZH.md', PRIOR / 'REPORT_EN.md',
             PLAN / 'PLAN.md', PLAN / 'config.json', Path(__file__),
             ROOT / 'reports/v9_r9_lr6e4_functional_v1/verification.json',
             PLAN / 'launch.json', ROOT / 'reports/v9_r9_lr6e4_launch_failure_v1/record.json',
             ROOT / 'reports/v9_r9_lr6e4_axis_changes_v1/summary.json',
             ROOT / 'reports/v9_r9_lr6e4_cases_v1/summary.json']
    hashes = {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}
    evidence = {'status': 'generated_bilingual_draft_review_required', 'input_hashes': hashes,
                'local_links_checked': checked_links,
                'screening_gates': analysis['screening_gates'], 'manual_proofreading_complete': False,
                'visual_review_complete': False, 'version_decision_complete': False, 'report_complete': False,
                'goal_achieved': False, 'training_performed': False, 'data_modified': False,
                'baseline_refit': False, 'complete_test_opened': False,
                'scope': 'Aggregate-only report draft; prior reviewed history retained. All newly added tables identical across languages. No claim of visual/manual review or final model acceptance.'}
    verify_hashes(hashes)
    output.mkdir(parents=True)
    for lang, content in documents.items():
        (output / f'REPORT_{lang}.md').write_text(content, encoding='utf-8')
    evidence['document_hashes'] = {f'REPORT_{lang}.md': digest(output / f'REPORT_{lang}.md') for lang in documents}
    (output / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': evidence['status'], 'output_dir': str(output), 'report_complete': False}))


if __name__ == '__main__':
    main()
