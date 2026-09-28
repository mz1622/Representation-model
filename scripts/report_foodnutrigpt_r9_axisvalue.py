"""Build bilingual AXISVALUE evidence drafts only after completed, audited analysis and fit.

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
PRIOR = ROOT / 'experiments/foodnutrigpt_v9_research/report_snapshot_r9_mse_v1'
ANALYSIS = ROOT / 'reports/v9_r9_axisvalue_analysis_v1'
FIT = ROOT / 'reports/v9_r9_axisvalue_fit_v1'
PLAN = ROOT / 'experiments/foodnutrigpt_v9_research/r9/axisvalue_v1'
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
        ANALYSIS / 'summary.json': 'complete_registered_axisvalue_analysis',
        ANALYSIS / 'status.json': 'complete', FIT / 'summary.json': 'complete',
        FIT / 'status.json': 'complete',
        PRIOR / 'review_verification.json': 'reviewed_complete_mse_stage_report',
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
        directory = ROOT / records['axisvalue']['manifest']['spec'][
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
    result['intervals'] = table(['Reference', 'Task', 'Metric', 'AXISVALUE gain (%)', '95% lower (%)', '95% upper (%)'], intervals)
    result['retrieval_intervals'] = table(['Reference', 'Visible fraction', 'Metric', 'AXISVALUE minus reference', '95% lower', '95% upper'],
        [[reference, fraction, key, number(row['candidate_minus_baseline']), *[number(value) for value in row['difference_95_interval']]]
         for reference, tasks in comparisons.items() for fraction, metrics in tasks['retrieval']['comparisons'].items()
         for key, row in metrics.items()])
    result['decomposition'] = table(['Fixed-denominator component', 'AXISVALUE minus MAE'],
        [[key, number(value)] for key, value in analysis['fixed_denominator_axisvalue_minus_mae'].items()])
    result['fit'] = table(['Method', 'Partition', '142-axis MAE', 'Positive MAE', 'Explicit-zero MAE'],
        [[method.upper(), partition, *[number(data[method]['nutrition'][key]) for key in
                                      ['scaled_log_mae', 'positive_scaled_log_mae', 'zero_scaled_log_mae']]]
         for method in ['mae', 'axisvalue'] for partition, data in
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
    record = analysis['records']['axisvalue']['manifest']
    eligible = analysis['eligible_for_separately_registered_seed_confirmation']
    labels = [
        ('研究问题与预先假设', 'Research question and preregistered hypothesis'),
        ('父版本与受控改动', 'Parent and controlled intervention'),
        ('可复现信息', 'Reproduction'), ('完整结果', 'Complete results'),
        ('机制诊断', 'Mechanism diagnostics'), ('因果分析边界', 'Limits of causal interpretation'),
        ('筛选结果与版本决定', 'Screening and version decision'), ('下一轮问题与测试状态', 'Next questions and test status')]
    def sub(index):
        return f"### 11.{index}. {labels[index - 1][0 if zh else 1]}"
    verdict = (('满足' if eligible else '未满足') + '预先规定的筛选条件。只有种子20260922的结果，尚不能接受为稳定改进或宣称优于RF。') if zh else (
        ('Meets' if eligible else 'Does not meet') + ' the preregistered screening gates. Only seed 20260922 has been evaluated; this does not establish a stable improvement or superiority over RF.')
    parts = [
        '## 11. 营养轴×数值残差：完整证据草稿，解释待审阅' if zh else
        '## 11. Axis-by-value residual: completed evidence draft, interpretation pending review',
        sub(1),
        '假设显式营养轴×数值交互能够改善共享数值编码器的条件表示。此前MSE配方被拒绝，本轮保留MAE，单独改变数值token。氨基酸、脂肪酸及正/零误差只作机制诊断，不替代142轴主指标。' if zh else
        'The hypothesis is that an explicit axis-by-value interaction improves conditional numeric representations. The preceding MSE recipe was rejected. This experiment retains MAE and changes only numeric tokens. Amino-acid, fatty-acid and positive/zero errors are diagnostics, not substitutes for the 142-axis primary metric.',
        sub(2),
        '第4/12个配方以同种子MAE为父控制，从头初始化。输入由e_axis+g(t)改为e_axis+g(t)+t*r_axis，新增252×192=48,384个零初始化参数；隐藏值不贡献残差。旧参数初值、构造RNG及零残差下的前向均精确匹配。192维、3层、6头、FF768、dropout0.15、rank16、MAE、来源损失权重1、来源L2=1e-4、batch64、AdamW lr3e-4/weight decay1e-4、clip1和60轮余弦日程不变。无树重拟合、无数据变化。' if zh else
        'Candidate 4 of 12 uses the same-seed MAE model as its control and starts from scratch. Tokens change from e_axis+g(t) to e_axis+g(t)+t*r_axis, adding 252×192 = 48,384 zero-initialized parameters. Hidden values contribute no residual. Shared initial weights, constructor RNG and forward predictions at zero residual match exactly. Width 192, 3 layers, 6 heads, feedforward width 768, dropout 0.15, head rank 16, MAE, source-loss weight 1, source L2 of 1e-4, batch size 64, AdamW learning rate 3e-4/weight decay 1e-4, clipping at 1 and the 60-epoch cosine schedule are unchanged. No tree refitting or data changes.',
        sub(3),
        table(['Item', 'Value'], [
            ['Code commit', record['code_commit']], ['Checkpoint SHA256', record['checkpoint_sha256']],
            ['Full initial-state SHA256', record['initial_state_sha256']],
            ['Shared parent initial-state SHA256', record['parent_initial_state_sha256']],
            ['Data SHA256', record['data_hash']], ['Panel SHA256', record['panel_hash']],
            ['Name cache SHA256', record['name_cache_hash']], ['Parameters', record['parameter_count']],
            ['Trainable parameters', record['trainable_parameter_count']], ['Added parameters', record['added_parameters']],
            ['Selected epoch', record['best_epoch']], ['Run elapsed (s)', f"{record['elapsed_seconds']:.3f}"],
            ['Fit inference elapsed (s)', f"{fit['elapsed_seconds']:.3f}"]]),
        '全部60轮任务顺序、目标暴露量和学习率经独立重放核对。两份各323,809条预测、49,913个候选向量和19,089条排名精确重放。环境和命令见版本记录及运行清单；运行耗时含评价，MAE父运行不重复计入新增成本。' if zh else
        'Independent replay checks all 60 epoch orders, target exposures and learning rates, two prediction tables of 323,809 rows each, 49,913 candidate vectors and 19,089 retrieval ranks. The version record and manifest retain the environment and commands. Elapsed run time includes evaluation; the reused MAE control is not counted again as new training.',
        f"[Registration]({relative(PLAN / 'PLAN.md', output)}) · [Configuration]({relative(PLAN / 'config.json', output)}) · [Version record]({relative(PLAN / 'README.md', output)}) · [Analysis]({relative(ANALYSIS / 'summary.json', output)}) · [Training fit]({relative(FIT / 'summary.json', output)})",
        sub(4),
        'MAE与AXISVALUE均为种子22，RF/XGB/KNN为冻结结果；每个神经模型三任务使用同一补全选点。条件正值/零值误差的分母不同，不能相加重构主指标。历史MAE三种子均值见第6节，不与这里的单种子混用。' if zh else
        'MAE and AXISVALUE both use seed 22; RF/XGB/KNN results are frozen. Each neural model uses its completion-selected checkpoint for all tasks. Conditional positive and zero errors have different denominators and cannot be added to recover primary error. The historical three-seed MAE mean in Section 6 is distinct from these single-seed results.',
        '**补全 / Completion**', shared['completion'], '**仅名称 / Name-only**', shared['name_only'],
        '**45/187 axes**', shared['subsets'], '**检索 / Retrieval**', shared['retrieval'],
        '检索固定49,913个候选名称，候选向量仅由名称预测，不使用候选真实营养值；查询不含名称。正确答案仍为原始名称精确匹配，尚无确认别名映射。食品组条件性区间不覆盖种子总体、标签有效性或重复选型的不确定性。' if zh else
        'Retrieval uses 49,913 fixed candidate names. Candidate vectors come only from name predictions, with no measured candidate nutrition; queries contain no names. Correct answers still use exact original-name matching, without a confirmed alias map. Conditional food-group intervals do not cover seed-population, label-validity or repeated-selection uncertainty.',
        shared['intervals'], shared['retrieval_intervals'],
        sub(5), shared['decomposition'], shared['fit'], shared['clipping'],
        f"![MAE control and axis-value learning curves]({relative(ANALYSIS / 'learning_curves.png', output)})",
        '低估、高估及零值三项共同分母贡献精确重构新模型减MAE的主误差。训练拟合在全部1,828,536个已观测训练目标上进行无来源残差推理，MAE既有预测重新评分一致。图已经生成不等于实际查看；本草稿未作视觉审阅声明。还需结合逐轴/来源支持、典型案例解释收益或代价。' if zh else
        'The underprediction, overprediction and explicit-zero contributions use a common denominator and reconstruct the primary difference exactly. Training fit evaluates all 1,828,536 observed training targets through source-free inference; saved MAE predictions rescore exactly. Figure generation is not visual inspection, and this draft makes no review claim. Axis/source support and selected success and failure cases still require interpretation.',
        '功能预检11项测试通过。真实32任务50步训练损失从0.217358降至0.044068，只证明可学习性；正式运行重置参数及RNG。最初pytest因缺少PYTHONPATH在收集阶段失败，修正执行环境后通过，没有借此重训正式候选。' if zh else
        'Eleven functional tests passed. A 50-step check on 32 actual training tasks reduced loss from 0.217358 to 0.044068, demonstrating learnability only; formal training resets parameters and RNG. An initial pytest invocation failed during collection because PYTHONPATH was missing. It passed after the execution environment was corrected, without restarting a formal candidate.',
        sub(6),
        '本轮同时增加轴专属交互参数和容量，也改变后续梯度与优化轨迹。即使验证改善，也不能唯一归因于交互结构；仍需等容量对照才能支持更强机制判断。训练与验证食品/支持不同，拟合差距不能单独证明过拟合。数据、指标和门槛未根据结果调整，标签异常、别名和来源转载问题仍未解决。' if zh else
        'This intervention adds both axis-specific interaction parameters and capacity, while changing subsequent gradients and optimization. A validation gain would not uniquely establish an interaction mechanism; a matched-capacity control would be needed for that stronger claim. Different foods and support in training and validation prevent fit gaps alone from establishing overfitting. Data, metrics and gates were not adjusted to outcomes. Label anomalies, aliases and source borrowing remain unresolved.',
        sub(7), shared['gates'], verdict,
        '此处只呈现自动筛选结果。版本接受/拒绝和因果解释仍须完成实际曲线、逐轴与案例审阅，另存明确决定。生成报告不会启动新训练。通过筛选也只能进入另行登记的23/24复验，不能替代最终三种子对冻结RF的确认。' if zh else
        'This section presents the automatic screen only. Accept/reject interpretation still requires actual curve, axis and case review, followed by a separate recorded decision. Report generation starts no training. Passing the screen can only motivate separately registered seeds 23/24; it cannot replace final three-seed confirmation against frozen RF.',
        sub(8),
        '完整审阅后按登记规则决定复验或提出下一单因素假设。数据和RF/XGB继续冻结，历史测试不打开；来源留出、低标签迁移和可靠foundation model能力仍未验证。无论结果正负，都保留全部候选与失败记录。' if zh else
        'After complete review, apply the registered rule to decide on replication or the next single-factor hypothesis. Data and RF/XGB remain frozen and historical test remains closed. Source-held-out transfer, low-label transfer and a reliable foundation-model claim remain unverified. Retain every candidate and failure regardless of outcome.'
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
        raise ValueError('Wait for complete AXISVALUE analysis and training-fit diagnosis: ' + ', '.join(pending))
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
            or not fit['existing_mae_predictions_rescored_exact'] or not fit['new_axisvalue_predictions_disk_reloaded_exact']):
        raise ValueError('Expected completed controls and fit replay')
    for role in ['mae', 'axisvalue']:
        if fit['source_free_validation'][role] != analysis['records'][role]['metrics']['completion']:
            raise ValueError('Fit and analysis validation metrics differ')
    shared = tables(analysis, fit)
    documents, supplements = {}, {}
    for lang in ['ZH', 'EN']:
        prior = rebase((PRIOR / f'REPORT_{lang}.md').read_text(encoding='utf-8'), PRIOR / f'REPORT_{lang}.md', output)
        prior_scope = prior.splitlines()[4]
        scope = ('草稿：第1–10节保留已审阅的R0–R9/MSE阶段历史；第11节新增轴×数值残差的完整证据。当前方法的实际曲线审阅、因果解释与版本决定尚未完成，不能称为最终报告。' if lang == 'ZH' else
                 'DRAFT: Sections 1–10 retain the reviewed R0–R9/MSE history. Section 11 adds completed axis-by-value residual evidence. Actual curve review, causal interpretation and the current version decision remain incomplete; this is not a final report.')
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
             PLAN / 'PLAN.md', PLAN / 'config.json', Path(__file__)]
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
