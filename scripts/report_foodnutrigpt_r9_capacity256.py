"""Build bilingual CAPACITY256 evidence drafts only after completed, audited analysis and fit.

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
PRIOR = ROOT / 'experiments/foodnutrigpt_v9_research/report_snapshot_r9_source0_v1'
ANALYSIS = ROOT / 'reports/v9_r9_capacity256_analysis_v1'
FIT = ROOT / 'reports/v9_r9_capacity256_fit_v1'
PLAN = ROOT / 'experiments/foodnutrigpt_v9_research/r9/capacity256_v1'
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
        ANALYSIS / 'summary.json': 'complete_registered_capacity256_analysis',
        ANALYSIS / 'status.json': 'complete', FIT / 'summary.json': 'complete',
        FIT / 'status.json': 'complete',
        PRIOR / 'review_verification.json': 'reviewed_complete_source0_stage_report',
        ROOT / 'reports/v9_r9_capacity256_functional_v1/verification.json': 'complete',
        ROOT / 'reports/v9_r9_capacity256_runtime_v1/summary.json': 'complete_training_only_runtime_estimate',
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
        directory = ROOT / records['capacity256']['manifest']['spec'][
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
    result['intervals'] = table(['Reference', 'Task', 'Metric', 'CAPACITY256 gain (%)', '95% lower (%)', '95% upper (%)'], intervals)
    result['retrieval_intervals'] = table(['Reference', 'Visible fraction', 'Metric', 'CAPACITY256 minus reference', '95% lower', '95% upper'],
        [[reference, fraction, key, number(row['candidate_minus_baseline']), *[number(value) for value in row['difference_95_interval']]]
         for reference, tasks in comparisons.items() for fraction, metrics in tasks['retrieval']['comparisons'].items()
         for key, row in metrics.items()])
    result['decomposition'] = table(['Fixed-denominator component', 'CAPACITY256 minus MAE'],
        [[key, number(value)] for key, value in analysis['fixed_denominator_capacity256_minus_mae'].items()])
    result['fit'] = table(['Method', 'Partition', '142-axis MAE', 'Positive MAE', 'Explicit-zero MAE'],
        [[method.upper(), partition, *[number(data[method]['nutrition'][key]) for key in
                                      ['scaled_log_mae', 'positive_scaled_log_mae', 'zero_scaled_log_mae']]]
         for method in ['mae', 'capacity256'] for partition, data in
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
    record = analysis['records']['capacity256']['manifest']
    parent = analysis['records']['mae']['manifest']
    eligible = analysis['eligible_for_separately_registered_seed_confirmation']
    functional_path = ROOT / 'reports/v9_r9_capacity256_functional_v1/verification.json'
    runtime_path = ROOT / 'reports/v9_r9_capacity256_runtime_v1/summary.json'
    functional, runtime = read(functional_path), read(runtime_path)
    smoke = functional['small_batch_overfit']
    labels = [('研究问题与预先假设', 'Question and hypothesis'),
        ('父版本与受控改动', 'Parent and intervention'), ('可复现信息', 'Reproduction'),
        ('完整结果', 'Complete results'), ('机制诊断', 'Mechanism diagnostics'),
        ('因果分析边界', 'Causal limits'), ('筛选与版本决定', 'Screen and decision'),
        ('下一轮问题与测试状态', 'Next questions and test status')]
    def sub(index):
        return f"### 13.{index}. {labels[index-1][0 if zh else 1]}"
    def words(chinese, english):
        return chinese if zh else english
    parts = [
        words('## 13. PDF容量组合256/8/FF1024：证据草稿，解释待审阅',
              '## 13. PDF capacity group 256/8/FF1024: evidence draft, interpretation pending review'),
        sub(1), words(
            '关闭来源校准的上一配方未通过筛选，因此仍以192维、source weight=1的MAE为控制。原PDF采用256维、8头、FF1024，本轮检验这个容量组合能否改善营养上下文表示。假设成立时无来源训练拟合和验证误差可能同时改善；更大模型也可能产生优化或泛化代价。PDF原始数据和评分不同，其历史分数不能直接比较。',
            'The preceding source-calibration-removal recipe failed the screen, so the control remains the width-192 MAE model with source weight=1. The reference PDF used width 256, eight heads and FF1024. This experiment tests whether that capacity combination improves nutrition-context representation. The hypothesis predicts potentially better source-free training fit and validation error; a larger model can also incur optimization or generalization costs. The PDF used different data and scoring, so its historical scores are not directly comparable.'),
        sub(2), words(
            '第6/12个候选仅改变登记的容量组：d_model 192→256、n_heads 6→8、feedforward_dim 768→1024，每头维度仍为32。保持3层、dropout 0.15、rank16、MAE、来源校准权重1、来源L2=1e-4、按(1+w)归一化的损失、batch64、AdamW lr3e-4/weight decay1e-4、clip1和60轮余弦日程。数据、名称缓存和有效32个文本方向、家族遮蔽、任务顺序、评分及RF/XGB均冻结。没有引入MSE、轴×数值残差或two-stage。',
            'Candidate 6 of 12 changes only the registered capacity group: d_model 192 to 256, n_heads 6 to 8, and feedforward_dim 768 to 1024; head dimension remains 32. It retains three layers, dropout 0.15, head rank 16, MAE, source-calibration weight 1, source L2=1e-4, the loss normalized by (1+w), batch size 64, AdamW lr3e-4/weight decay1e-4, clipping 1 and the 60-epoch cosine schedule. Data, name cache and 32 effective text directions, family masking, task order, scoring and RF/XGB are frozen. MSE, an axis-by-value residual and two-stage training are not introduced.'),
        sub(3), table(['Item', 'Value'], [
            ['Code commit', record['code_commit']], ['Checkpoint SHA256', record['checkpoint_sha256']],
            ['Capacity initial-state SHA256', record['initial_state_sha256']],
            ['Parent initial-state SHA256', parent['initial_state_sha256']],
            ['Data SHA256', record['data_hash']], ['Panel SHA256', record['panel_hash']],
            ['Name cache SHA256', record['name_cache_hash']],
            ['Capacity / parent parameters', f"{record['parameter_count']} / {parent['parameter_count']}"],
            ['Capacity / parent requires_grad parameters', f"{record['trainable_parameter_count']} / {parent['trainable_parameter_count']}"],
            ['Seed', record['seed']], ['Selected epoch', record['best_epoch']],
            ['Run elapsed (s)', f"{record['elapsed_seconds']:.3f}"],
            ['Training-fit inference elapsed (s)', f"{fit['elapsed_seconds']:.3f}"],
            ['Pre-run estimated elapsed (s)', f"{runtime['estimated_training_seconds']:.3f}"],
            ['Measured 256 / 192 step-time ratio', number(runtime['capacity_to_parent_step_ratio'])],
            ['Registered first follow-up delay (min)', runtime['first_followup_delay_minutes']]]),
        words(
            '独立审计重建本容量模型的初值，核对参数量及全部60轮任务顺序、目标暴露量和学习率，并精确重放两份各323,809条预测、49,913个名称候选向量和19,089条排名。环境为Windows、Python3.10.19、PyTorch2.7.1+cu128、RTX5070Ti16GB，保持父实验CUDA后端。短时估时使用相同104个训练batch，8步预热与96步计时、每种容量两轮交替次序，诊断权重丢弃；不评价验证结果。正式启动后设置267分钟回访，实际耗时单独报告。',
            'The independent audit reconstructs this capacity model’s initialization, checks parameter counts and all 60 epoch orders, target exposures and learning rates, and exactly replays two 323,809-row prediction tables, 49,913 candidate-name vectors and 19,089 ranks. Environment: Windows, Python 3.10.19, PyTorch 2.7.1+cu128 and RTX 5070 Ti 16 GB, retaining the parent CUDA backend. The runtime estimate uses identical 104 training batches, eight warm-up and 96 timed steps, with two interleaved rounds per capacity. Diagnostic weights are discarded and validation is not scored. A 267-minute follow-up was scheduled after launch; actual elapsed time is reported separately.'),
        f"[Registration]({relative(PLAN/'PLAN.md',output)}) · [Configuration]({relative(PLAN/'config.json',output)}) · [Version]({relative(PLAN/'README.md',output)}) · [Analysis]({relative(ANALYSIS/'summary.json',output)}) · [Training fit]({relative(FIT/'summary.json',output)}) · [Functional checks]({relative(functional_path,output)}) · [Runtime estimate]({relative(runtime_path,output)})",
        sub(4), words(
            'MAE和CAPACITY256均为种子20260922；RF/XGB/KNN沿用冻结结果。每个神经模型三任务使用同一个按补全主指标选择的检查点。第6节的历史MAE三种子汇总与这里单种子容量对照不同。',
            'MAE and CAPACITY256 both use seed 20260922; RF/XGB/KNN results are frozen. Each neural model uses one completion-selected checkpoint for all three tasks. The historical three-seed MAE aggregate in Section 6 is distinct from this single-seed capacity contrast.'),
        '**补全 / Completion**', shared['completion'], '**仅名称 / Name-only**', shared['name_only'],
        '**45/187 axes**', shared['subsets'], '**检索 / Retrieval**', shared['retrieval'],
        words(
            '检索固定49,913个名称候选，候选向量仅由名称预测，不能使用候选真实营养档案；营养查询不含名称。正确答案按原始名称精确匹配，没有已确认的别名映射。食品组条件区间不覆盖种子总体、标签有效性和重复选型的不确定性。',
            'Retrieval uses 49,913 fixed candidate names. Candidate vectors are predicted from names, without measured candidate nutrition; nutrient queries contain no names. Correct answers use exact original-name matching without confirmed aliases. Conditional food-group intervals do not cover seed-population, label-validity or repeated-selection uncertainty.'),
        shared['intervals'], shared['retrieval_intervals'],
        sub(5), shared['decomposition'], shared['fit'], shared['clipping'],
        f"![Capacity 192/6/768 versus 256/8/1024 learning curves]({relative(ANALYSIS/'learning_curves.png',output)})",
        words(
            '正值低估、正值高估和零值贡献使用共同分母，精确重构主误差变化。条件正值/零值宏平均的分母不同，不能直接相加。训练拟合对全部1,828,536个已观测目标进行无来源残差推理；既有MAE预测重新评分一致。图已生成不代表完成目视审阅，逐轴、来源、案例及曲线仍需实际解释。',
            'Positive underprediction, positive overprediction and explicit-zero contributions use a common denominator and reconstruct the primary change exactly. Conditional positive/zero macro averages use different denominators and cannot simply be added. Training fit uses source-free inference for all 1,828,536 observed targets; saved MAE predictions are rescored exactly. Figure generation does not establish visual review. Axes, sources, cases and curves still require actual interpretation.'),
        table(['Functional exercise', 'Value'], [
            ['Training tasks', 32], ['Steps', smoke['steps']],
            ['Loss before', number(smoke['before'])], ['Loss after', number(smoke['after'])]]),
        words(
            '训练前12项共享/合成功能测试通过。真实训练行核验复现192控制初值，并验证256构造器与显式替换配置的独立构造在权重、同宽构造RNG、前向及损失上精确匹配。隐藏标签和来源不影响基础前向，未观测轴可查询，8个训练名称×187轴输出有限。50步后来源残差已学习为非零，仍不进入基础推理，初始和已学模型均可精确重载。上述只证明实现和可学习性，不作为性能证据。',
            'Twelve shared/synthetic functional tests passed before training. Checks on real training rows reproduce the width-192 control initialization and match the width-256 factory exactly against an independent explicit configuration replacement in weights, same-width constructor RNG, forward outputs and loss. Hidden labels and source do not affect base predictions, unobserved axes can be queried, and eight training names by 187 axes produce finite outputs. After 50 steps, source residuals have learned nonzero values while remaining absent from base inference; initial and learned models reload exactly. These checks support implementation and learnability, not predictive performance.'),
        sub(6), words(
            '这是容量组合干预，宽度、头数、前馈规模、参数量、初值形状及随机数消耗共同改变。相同整数种子不表示跨容量的共享参数、构造RNG或dropout路径相同；不能将结果只归因于宽度，也不能据单种子证明真实生物机制。训练和验证的食品及支持不同，拟合差距不能单独证明过拟合。数据、指标和预登记筛选门槛不随结果调整。',
            'This is a capacity-group intervention: width, head count, feedforward size, parameter count, initialization shapes and random consumption change together. The same integer seed does not match shared parameters, constructor RNG or dropout streams across capacities. Results cannot be attributed to width alone or establish a biological mechanism from one seed. Training and validation differ in foods and support, so fit gaps alone do not prove overfitting. Data, metrics and registered screening gates are not adjusted to outcomes.'),
        sub(7), shared['gates'],
        words(('满足' if eligible else '未满足')+'登记的筛选条件；单种子不能确认为稳定改进或优于RF。',
              ('Meets' if eligible else 'Does not meet')+' the registered screen; one seed cannot confirm stable improvement or superiority over RF.'),
        words(
            '这里只呈现自动筛选。实际审阅后再保存接受、拒绝或证据不足的版本决定。通过筛选只允许另行登记20260923/24复验，不能替代对冻结RF的三种子确认。',
            'This is the automatic screen only. An accepted, rejected or insufficient-evidence decision must be saved after actual review. Passing permits separately registered replications with seeds 20260923 and 20260924; it does not replace three-seed confirmation against frozen RF.'),
        sub(8), words(
            '完成审阅后按登记条件决定复验或下一研究问题。历史数据和树结果继续冻结，测试保持关闭。来源留出、少样本迁移和可靠foundation model能力仍未验证。保留所有结果与失败，不把不同模型的任务最佳分数拼成一个模型。',
            'After review, apply the registered criteria to choose replication or the next research question. Historical data and tree results remain frozen and the test stays closed. Source-held-out transfer, few-shot transfer and a reliable foundation-model claim remain unverified. Retain all results and failures without combining different models’ best task scores.')
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
        raise ValueError('Wait for complete capacity-group analysis and training-fit diagnosis: ' + ', '.join(pending))
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
    if (not analysis['capacity_group_controls_verified'] or fit['training_job_count'] != 1828536
            or not fit['existing_mae_predictions_rescored_exact'] or not fit['new_capacity256_predictions_disk_reloaded_exact']):
        raise ValueError('Expected completed controls and fit replay')
    for role in ['mae', 'capacity256']:
        if fit['source_free_validation'][role] != analysis['records'][role]['metrics']['completion']:
            raise ValueError('Fit and analysis validation metrics differ')
    shared = tables(analysis, fit)
    documents, supplements = {}, {}
    for lang in ['ZH', 'EN']:
        prior = rebase((PRIOR / f'REPORT_{lang}.md').read_text(encoding='utf-8'), PRIOR / f'REPORT_{lang}.md', output)
        prior_scope = prior.splitlines()[4]
        scope = ('草稿：第1–12节保留上一版已审阅的历史证据，其中“最新”和“下一轮”均描述历史时点；第13节新增PDF容量组合实验的证据。本轮实际曲线审阅、因果解释与版本决定尚未完成，不能称为最终报告。' if lang == 'ZH' else
                 'DRAFT: Sections 1–12 retain previously reviewed history; their latest/next experiment statements describe that historical stage. Section 13 adds completed PDF capacity-group evidence. Actual curve review, interpretation and the version decision remain incomplete; this is not a final report.')
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
             ROOT / 'reports/v9_r9_capacity256_functional_v1/verification.json',
             ROOT / 'reports/v9_r9_capacity256_runtime_v1/summary.json']
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
