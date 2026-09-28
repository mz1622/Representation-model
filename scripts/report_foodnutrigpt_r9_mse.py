"""Build bilingual MSE evidence drafts only after completed, audited analysis and fit.

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
PRIOR = ROOT / 'experiments/foodnutrigpt_v9_research/report_snapshot_r9_confirmation_v1'
ANALYSIS = ROOT / 'reports/v9_r9_mse_analysis_v1'
FIT = ROOT / 'reports/v9_r9_mse_fit_v1'
CALIBRATION = ROOT / 'reports/v9_r9_mae_calibration_stability_v1'
PLAN = ROOT / 'experiments/foodnutrigpt_v9_research/r9/mse_v1'
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
        ANALYSIS / 'summary.json': 'complete_registered_mse_analysis',
        ANALYSIS / 'status.json': 'complete', FIT / 'summary.json': 'complete',
        FIT / 'status.json': 'complete', CALIBRATION / 'summary.json': 'complete',
        PRIOR / 'review_verification.json': 'reviewed_complete_stage_report',
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


def tables(analysis, fit, calibration):
    records, comparisons = analysis['records'], analysis['comparisons']
    scores = {key.upper(): record['metrics'] for key, record in records.items()}
    retrieval = {key.upper(): record['retrieval'] for key, record in records.items()}
    for reference in ['rf32', 'xgb32']:
        scores[reference.upper()] = {task: comparisons[reference][task]['scores']['baseline']
                                    for task in ['completion', 'name_only']}
        directory = ROOT / records['mse']['manifest']['spec'][
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
    result['intervals'] = table(['Reference', 'Task', 'Metric', 'MSE gain (%)', '95% lower (%)', '95% upper (%)'], intervals)
    result['retrieval_intervals'] = table(['Reference', 'Visible fraction', 'Metric', 'MSE minus reference', '95% lower', '95% upper'],
        [[reference, fraction, key, number(row['candidate_minus_baseline']), *[number(value) for value in row['difference_95_interval']]]
         for reference, tasks in comparisons.items() for fraction, metrics in tasks['retrieval']['comparisons'].items()
         for key, row in metrics.items()])
    result['decomposition'] = table(['Fixed-denominator component', 'MSE minus MAE'],
        [[key, number(value)] for key, value in analysis['fixed_denominator_mse_minus_mae'].items()])
    result['fit'] = table(['Method', 'Partition', '142-axis MAE', 'Positive MAE', 'Explicit-zero MAE'],
        [[method.upper(), partition, *[number(data[method]['nutrition'][key]) for key in
                                      ['scaled_log_mae', 'positive_scaled_log_mae', 'zero_scaled_log_mae']]]
         for method in ['mae', 'mse'] for partition, data in
         [('train', fit['source_free_training_fit']), ('validation', fit['source_free_validation'])]])
    result['clipping'] = table(['Objective', 'First 10 clip fraction', 'Last 10 clip fraction', 'Selected epoch', 'Elapsed (s)'],
        [[role.upper(), number(row['first10_clip_fraction']), number(row['last10_clip_fraction']),
          row['manifest']['best_epoch'], f"{row['manifest']['elapsed_seconds']:.3f}"]
         for role, row in records.items()])
    result['gates'] = table(['Registered screening gate vs same-seed MAE', 'Pass'],
                           [[key, str(bool(value))] for key, value in analysis['screening_gates'].items()])
    result['calibration'] = table(['MAE seed pair', 'Nutrition offset Pearson r', 'Difference RMS'],
        [[f"{row['seed_a']}/{row['seed_b']}", number(row['pearson_r']), number(row['difference_rms'])]
         for row in calibration['seed_pairs'] if row['subset'] == 'nutrition'])
    return result


def supplement(lang, analysis, fit, shared, output):
    zh = lang == 'ZH'
    heading = '## 9. MSE 单因素实验：完整证据、待审阅解释' if zh else '## 9. MSE objective contrast: completed evidence, interpretation pending review'
    eligible = analysis['eligible_for_separately_registered_seed_confirmation']
    verdict = (('已满足' if eligible else '未满足') + '进入独立登记种子复验的筛选条件；当前仍只有一个MSE种子，不能据此接受为最终模型。') if zh else (
        ('Meets' if eligible else 'Does not meet') + ' the gates for separately registered seed confirmation. Only one MSE seed exists; this is not final model acceptance.')
    record = analysis['records']['mse']['manifest']
    labels = [
        ('研究问题与假设', 'Research question and hypothesis'),
        ('父版本与受控改动', 'Parent and controlled intervention'),
        ('复现信息', 'Reproduction'), ('完整结果', 'Complete results'),
        ('机制诊断', 'Mechanism diagnostics'), ('因果解释边界', 'Limits of causal interpretation'),
        ('筛选结果与版本决定', 'Screening and version decision'), ('下一步与测试集状态', 'Next step and test status')]
    def sub(index):
        return f"### 9.{index}. {labels[index - 1][0 if zh else 1]}"
    text = [heading, sub(1),
        '假设平方残差训练能减轻MAE控制的正值低估，同时检查显式零、旧log-MAE及两个辅助任务的代价。PDF的宏轴MSE仅提供动机，其数据与评分不同，历史成绩不作为本轮对照。' if zh else
        'The hypothesis is that squared-residual training reduces positive underprediction seen with the MAE control. Explicit zeros, legacy log-MAE and both auxiliary tasks track possible costs. The PDF motivates macro-axis MSE, but its different data and scoring prevent historical score comparisons.',
        sub(2),
        '第3/12个方法候选，仅MAE→MSE；种子20260922、初值、60轮样本顺序、学习率日程、数据、来源权重和评分均核对一致。模型192维/3层/6头/FF768/rank16/dropout0.15；AdamW lr3e-4、weight decay1e-4、batch64、clip1。来源校准和base损失取均值，残差L2系数1e-4。无两阶段、无树重拟合。' if zh else
        'Candidate 3 of 12 changes only MAE to MSE. Seed 20260922, initialization, all 60 epoch orders, learning-rate schedule, data, source weights and scoring match the control. Model: width192/3 layers/6 heads/FF768/rank16/dropout0.15. AdamW lr3e-4, weight decay1e-4, batch64, clip1. Base and calibrated losses are averaged; source-residual L2 is1e-4. No two-stage training or tree refitting.',
        sub(3),
        table(['Item', 'Value'], [['Code commit', record['code_commit']], ['Checkpoint SHA256', record['checkpoint_sha256']],
            ['Initial-state SHA256', record['initial_state_sha256']], ['Data SHA256', record['data_hash']],
            ['Panel SHA256', record['panel_hash']], ['Name cache SHA256', record['name_cache_hash']],
            ['Selected epoch', record['best_epoch']], ['Training elapsed (s)', f"{record['elapsed_seconds']:.3f}"],
            ['Training-fit inference elapsed (s)', f"{fit['elapsed_seconds']:.3f}"]]),
        '环境及精确源码身份保存在运行清单。耗时包含运行内核验和评价，不是纯训练速度；MAE为复用父运行，不能重复计入新增训练成本。' if zh else
        'The manifest retains environment and exact source identities. Elapsed time includes within-run checks and evaluation, rather than isolated training speed. MAE reuses the parent and must not be counted again as new training.',
        f"[Registration]({relative(PLAN / 'PLAN.md', output)}) · [Configuration]({relative(PLAN / 'config.json', output)}) · [Analysis]({relative(ANALYSIS / 'summary.json', output)}) · [Training fit]({relative(FIT / 'summary.json', output)})",
        sub(4),
        'MAE和MSE行均为种子22；与第6节MAE三种子均值严格区分。RF/XGB/KNN均是冻结结果。每个模型三项任务使用补全选中的同一检查点。主误差/条件正值/条件零值的分母不同，后两者不能直接相加。' if zh else
        'MAE and MSE rows both use seed22 and are distinct from the three-seed MAE mean in Section6. RF/XGB/KNN are frozen. Each model uses its completion-selected checkpoint for every task. Conditional positive/zero errors have different denominators and cannot be added to recover primary error.',
        '**补全 / Completion**', shared['completion'], '**仅名称 / Name-only**', shared['name_only'],
        '**45/187 axes**', shared['subsets'], '**检索 / Retrieval**', shared['retrieval'],
        '固定49,913个名称候选，候选向量仅来自名称预测；查询不含名称。正确答案为原始名称精确匹配，尚未建立已确认别名映射。食品组条件性区间不覆盖种子总体、标签真实性或多轮选型的不确定性。' if zh else
        'There are49,913 fixed candidate names, with vectors generated only from name predictions. Queries contain no names. Correct answers use exact original-name matching; no confirmed alias map exists. Conditional food-group intervals exclude seed-population, label-validity and repeated-selection uncertainty.',
        shared['intervals'], shared['retrieval_intervals'],
        sub(5), shared['decomposition'], shared['fit'], shared['clipping'],
        f"![MAE and MSE learning curves]({relative(ANALYSIS / 'learning_curves.png', output)})",
        '上述三项共分母误差贡献之和等于MSE减MAE的主误差。训练拟合使用全部1,828,536个已观测目标及无来源残差推理，MAE旧预测已精确重算。图已生成不等于已经目视审阅；本草稿不作该声明。' if zh else
        'The three common-denominator contributions sum to primary MSE-minus-MAE error. Training fit uses all1,828,536 observed targets and source-free inference; saved MAE predictions were rescored exactly. Figure generation is not visual review; this draft makes no review claim.',
        sub(6),
        '损失替换是受控干预，但同时改变梯度大小、裁剪频率和优化轨迹。即使正值或主指标改善，也不能唯一归因为均值/中位数目标；失败只能否定此固定优化设置下的配方。训练/验证食品、支持数和选点不同，拟合差距本身不能证明过拟合。逐轴/来源分解为探索结果，未据其改数据、权重或门槛。' if zh else
        'The objective substitution is controlled but also changes gradient scale, clipping and optimization trajectory. Improvements cannot uniquely establish a mean-versus-median mechanism; failure rejects this recipe under its fixed optimization settings. Train/validation foods, supports and selection differ, so fit gaps alone do not establish overfitting. Axis/source decompositions are exploratory; data, weights and gates were not changed in response.',
        sub(7), shared['gates'], verdict,
        '后续配方或种子须另行登记，本报告生成器不会启动训练或自动作科学接受决定。还需实际曲线审阅、逐轴/来源与失败案例解释，以及最终版本README核对。' if zh else
        'Further recipes or seeds require separate registration. This generator starts no training and makes no scientific acceptance decision. Actual figure review, axis/source and failure-case interpretation, and final version README review remain required.',
        sub(8),
        '先完成本轮解释与双语审阅，再依据登记条件决定是否追加种子或提出下一单因素假设。数据、树结果和测试使用政策不变；未见来源及低标签迁移尚未验证。' if zh else
        'Complete interpretation and bilingual review before deciding whether to replicate seeds or register a new single-factor hypothesis. Data, tree results and test policy remain unchanged. Unseen-source and low-label transfer remain unverified.',
        '## 10. MAE 来源残差稳定性：描述性附录' if zh else '## 10. MAE source-offset stability: descriptive appendix',
        shared['calibration'],
        '该诊断只读取三个已审计MAE检查点的参数，无前向或优化。24个训练来源的营养轴残差相关性较高，但不能证明来源校准有益，也不能证明预测稳定；相关性可能受少数大残差轴影响，选点为60/60/57轮。这里没有校准开关消融。' if zh else
        'This diagnostic reads parameters from the three audited MAE checkpoints, with no forward pass or optimization. Nutrition offsets over24 training sources are correlated, but this does not establish calibration benefit or prediction stability. Large-offset axes may dominate correlation, and selected epochs are60/60/57. No calibration-on/off ablation was performed.',
        f"[Parameter evidence]({relative(CALIBRATION / 'summary.json', output)})"]
    return '\n\n'.join(text)


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
        raise ValueError('Wait for complete MSE analysis and training-fit diagnosis: ' + ', '.join(pending))
    if args.output_dir is None or args.output_dir.exists():
        raise ValueError('A new report directory is required')
    output = args.output_dir.resolve()
    if not output.is_relative_to((ROOT / 'experiments/foodnutrigpt_v9_research').resolve()):
        raise ValueError('Report snapshot must be inside the research experiment directory')
    analysis, fit, calibration = [read(path / 'summary.json') for path in [ANALYSIS, FIT, CALIBRATION]]
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
            or not fit['existing_mae_predictions_rescored_exact'] or not fit['new_mse_predictions_disk_reloaded_exact']):
        raise ValueError('Expected completed controls and fit replay')
    for role in ['mae', 'mse']:
        if fit['source_free_validation'][role] != analysis['records'][role]['metrics']['completion']:
            raise ValueError('Fit and analysis validation metrics differ')
    shared = tables(analysis, fit, calibration)
    documents = {}
    for lang in ['ZH', 'EN']:
        prior = rebase((PRIOR / f'REPORT_{lang}.md').read_text(encoding='utf-8'), PRIOR / f'REPORT_{lang}.md', output)
        prior_scope = prior.splitlines()[4]
        scope = ('草稿：第1–8节保留已审阅的MAE复验阶段历史；最新MSE完整实验和当前筛选结果见第9节，第10节补充描述性参数诊断。MSE解释、实际图形审阅和版本决定尚未完成；不可将此草稿称为最终报告。' if lang == 'ZH' else
                 'DRAFT: Sections1–8 retain the reviewed MAE confirmation history. Section9 adds the completed MSE experiment and current screening evidence; Section10 adds descriptive parameter diagnostics. MSE interpretation, actual figure review and version decision remain incomplete; this draft is not a final report.')
        prior = prior.replace(prior_scope, scope, 1)
        documents[lang] = prior.rstrip() + '\n\n' + supplement(lang, analysis, fit, shared, output) + '\n'
    # Each language uses the exact same newly generated numeric table rows.
    for shared_table in shared.values():
        if any(content.count(shared_table) != 1 for content in documents.values()):
            raise ValueError('Bilingual table content differs or is duplicated')
    checked_links = 0
    for content in documents.values():
        for target in re.findall(r'\]\(([^)\n]+)\)', content):
            if re.match(r'^[a-z]+:|^#|^/', target):
                continue
            path = target.partition('#')[0]
            if path not in ['REPORT_ZH.md', 'REPORT_EN.md', 'evidence.json'] and not (output / path).exists():
                raise FileNotFoundError(output / path)
            checked_links += 1
    paths = [ANALYSIS / 'summary.json', FIT / 'summary.json', CALIBRATION / 'summary.json',
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
