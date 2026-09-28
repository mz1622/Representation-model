"""Build bilingual SOURCE0 evidence drafts only after completed, audited analysis and fit.

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
PRIOR = ROOT / 'experiments/foodnutrigpt_v9_research/report_snapshot_r9_axisvalue_v1'
ANALYSIS = ROOT / 'reports/v9_r9_source0_analysis_v1'
FIT = ROOT / 'reports/v9_r9_source0_fit_v1'
PLAN = ROOT / 'experiments/foodnutrigpt_v9_research/r9/source0_v1'
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
        ANALYSIS / 'summary.json': 'complete_registered_source0_analysis',
        ANALYSIS / 'status.json': 'complete', FIT / 'summary.json': 'complete',
        FIT / 'status.json': 'complete',
        PRIOR / 'review_verification.json': 'reviewed_complete_axisvalue_stage_report',
        ROOT / 'reports/v9_r9_source0_functional_v1/verification.json': 'complete',
        ROOT / 'reports/v9_r9_source0_functional_v1/reference_execution.json': 'complete_deterministic_functional_reference',
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
        directory = ROOT / records['source0']['manifest']['spec'][
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
    result['intervals'] = table(['Reference', 'Task', 'Metric', 'SOURCE0 gain (%)', '95% lower (%)', '95% upper (%)'], intervals)
    result['retrieval_intervals'] = table(['Reference', 'Visible fraction', 'Metric', 'SOURCE0 minus reference', '95% lower', '95% upper'],
        [[reference, fraction, key, number(row['candidate_minus_baseline']), *[number(value) for value in row['difference_95_interval']]]
         for reference, tasks in comparisons.items() for fraction, metrics in tasks['retrieval']['comparisons'].items()
         for key, row in metrics.items()])
    result['decomposition'] = table(['Fixed-denominator component', 'SOURCE0 minus MAE'],
        [[key, number(value)] for key, value in analysis['fixed_denominator_source0_minus_mae'].items()])
    result['fit'] = table(['Method', 'Partition', '142-axis MAE', 'Positive MAE', 'Explicit-zero MAE'],
        [[method.upper(), partition, *[number(data[method]['nutrition'][key]) for key in
                                      ['scaled_log_mae', 'positive_scaled_log_mae', 'zero_scaled_log_mae']]]
         for method in ['mae', 'source0'] for partition, data in
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
    record = analysis['records']['source0']['manifest']
    eligible = analysis['eligible_for_separately_registered_seed_confirmation']
    functional_path = ROOT / 'reports/v9_r9_source0_functional_v1/verification.json'
    smoke = read(functional_path)['small_batch_overfit']
    labels = [('研究问题与预先假设', 'Question and hypothesis'),
        ('父版本与受控改动', 'Parent and intervention'), ('可复现信息', 'Reproduction'),
        ('完整结果', 'Complete results'), ('机制诊断', 'Mechanism diagnostics'),
        ('因果分析边界', 'Causal limits'), ('筛选与版本决定', 'Screen and decision'),
        ('下一轮问题与测试状态', 'Next questions and test status')]
    def sub(index):
        return f"### 12.{index}. {labels[index-1][0 if zh else 1]}"
    def words(chinese, english):
        return chinese if zh else english
    parts = [
        words('## 12. 关闭来源校准目标：证据草稿，解释待审阅',
              '## 12. Removing source calibration: evidence draft, interpretation pending review'),
        sub(1), words(
            '训练同时拟合基础输出与已知来源的校准输出，推理只评价基础输出。本轮检验关闭校准目标能否改善基础预测。此前来源偏置稳定性和来源分区误差均不证明校准有益或有害。',
            'Training fits both base and known-source calibrated outputs, whereas inference evaluates the base output. This experiment tests whether removing the calibration objective improves base predictions. Stable source offsets and source-partition errors establish neither benefit nor harm.'),
        sub(2), words(
            '第5/12个候选以同种子MAE为控制，仅将source weight从1改为0。损失仍为(L_base+w*L_calibrated)/(1+w)+1e-4*mean(R_train²)，保留归一化，避免同时改变基础损失尺度。模型、零初始化来源表、L2、优化器和样本来源平衡权重均保留；来源表在w=0时保持零。192维、3层、6头、FF768、dropout0.15、rank16、MAE、batch64、AdamW lr3e-4/weight decay1e-4、clip1及60轮余弦日程均不变。',
            'Candidate 5 of 12 uses the same-seed MAE control and changes only source weight from 1 to 0. The loss remains (L_base+w*L_calibrated)/(1+w)+1e-4*mean(R_train²); normalization avoids simultaneously rescaling the base objective. Architecture, zero-initialized source table, L2, optimizer and source-balanced sample weights are retained. Source offsets stay zero at w=0. Width192, 3 layers, 6 heads, FF768, dropout0.15, head rank16, MAE, batch64, AdamW lr3e-4/weight decay1e-4, clipping1 and the60-epoch cosine schedule are unchanged.'),
        sub(3), table(['Item','Value'], [
            ['Code commit', record['code_commit']], ['Checkpoint SHA256',record['checkpoint_sha256']],
            ['Initial-state SHA256 (same as parent)',record['initial_state_sha256']],
            ['Data SHA256',record['data_hash']], ['Panel SHA256',record['panel_hash']],
            ['Name cache SHA256',record['name_cache_hash']], ['Parameters',record['parameter_count']],
            ['Parameters with requires_grad',record['trainable_parameter_count']],
            ['Source weight',record['spec']['source_weight']], ['Selected epoch',record['best_epoch']],
            ['Run elapsed (s)',f"{record['elapsed_seconds']:.3f}"],
            ['Fit inference elapsed (s)',f"{fit['elapsed_seconds']:.3f}"]]),
        words(
            '全部60轮任务顺序、目标暴露量和学习率与父控制一致；两份各323,809条预测、49,913个候选向量和19,089条排名独立精确重放。最优与最终来源表均为零；该表计入requires_grad参数但有效梯度为零。环境为Windows、Python3.10.19、PyTorch2.7.1+cu128、RTX5070Ti16GB。正式训练保持父CUDA后端，确定性设置仅用于功能参考进程。训练前估时约3小时并设置190分钟回访；估时不替代实际成本。',
            'All60 epoch orders, target exposures and learning rates match the parent. Independent replay checks two323,809-row prediction tables,49,913 candidate vectors and19,089 ranks exactly. Selected and final source tables are zero; they remain counted as requires_grad parameters but have zero effective gradients. Environment: Windows, Python3.10.19, PyTorch2.7.1+cu128, RTX5070Ti16GB. Formal training retains the parent CUDA backend; deterministic settings apply only to the functional reference process. The pre-run estimate was about3 hours with a190-minute follow-up; an estimate is not actual compute cost.'),
        f"[Registration]({relative(PLAN/'PLAN.md',output)}) · [Configuration]({relative(PLAN/'config.json',output)}) · [Version]({relative(PLAN/'README.md',output)}) · [Analysis]({relative(ANALYSIS/'summary.json',output)}) · [Training fit]({relative(FIT/'summary.json',output)}) · [Functional checks]({relative(functional_path,output)})",
        sub(4), words(
            'MAE和SOURCE0均为种子22；RF/XGB/KNN沿用冻结结果。各神经模型三任务使用同一个补全选中的检查点。第6节的历史MAE三种子均值与本节单种子比较不同。',
            'MAE and SOURCE0 both use seed22; RF/XGB/KNN results are frozen. Each neural model uses one completion-selected checkpoint for all tasks. The historical three-seed MAE mean in Section6 is distinct from these single-seed comparisons.'),
        '**补全 / Completion**',shared['completion'],'**仅名称 / Name-only**',shared['name_only'],
        '**45/187 axes**',shared['subsets'],'**检索 / Retrieval**',shared['retrieval'],
        words(
            '检索固定49,913个名称候选，候选向量仅由名称预测，不能用其真实营养值；营养查询不含名称。正确答案仍按原始名称精确匹配，无确认别名映射。食品组条件区间不包括种子总体、标签有效性或重复选型的不确定性。',
            'Retrieval uses49,913 fixed candidate names and name-predicted vectors without measured candidate nutrition; nutrient queries contain no names. Correct answers use exact original-name matching without confirmed aliases. Food-group conditional intervals exclude seed-population, label-validity and repeated-selection uncertainty.'),
        shared['intervals'],shared['retrieval_intervals'],
        sub(5),shared['decomposition'],shared['fit'],shared['clipping'],
        f"![Source-loss weight1 versus0 learning curves]({relative(ANALYSIS/'learning_curves.png',output)})",
        words(
            '正值低估、正值高估和零值贡献使用共同分母，精确重构主误差变化。条件正值/零值宏平均各有分母，不能直接相加。训练拟合对全部1,828,536个已观测训练目标进行无来源残差推理；既有MAE预测重新评分一致。图已生成不等于已查看，逐轴/来源/案例和实际曲线仍需审阅。',
            'Positive underprediction, positive overprediction and explicit-zero contributions use a common denominator and reconstruct the primary change exactly. Conditional positive/zero macro averages have separate denominators and cannot simply be added. Training fit uses source-free inference for all1,828,536 observed targets, with saved MAE predictions rescored exactly. Figure generation is not visual review; axes, sources, cases and actual curves still require interpretation.'),
        table(['Functional exercise','Value'],[
            ['Training tasks',32],['Steps',smoke['steps']],
            ['Loss before',number(smoke['before'])],['Loss after',number(smoke['after'])]]),
        words(
            '14项共享/合成功能测试通过。普通GPU预检在优化器步前因共享梯度逐位比较失败；重复同一目标也出现微小差异。CPU和确定性GPU参考在各自设备内精确匹配，未放宽原断言。完整50步检查通过，来源表保持零，学习后public loader重载精确。失败与恢复单独留档；这些只支持实现与可学习性，不是验证性能证据。',
            'Fourteen shared/synthetic functional tests passed. Ordinary-GPU preflight failed a bitwise shared-gradient comparison before optimizer steps; repeating the same objective also produced tiny differences. CPU and deterministic-GPU references matched exactly within each device without relaxing assertions. The full50-step check passed, offsets remained zero and the learned public-loader reload was exact. Failure and recovery are recorded; these checks support implementation and learnability, not validation performance.'),
        sub(6),words(
            '参数量和初值未变。干预是来源校准权重及其后续优化轨迹；确定性参考中的初始共享梯度相同，不意味着普通GPU反向逐位可重复。单种子结果不能唯一归因于真实来源偏差，也不能概括为校准始终无用。训练/验证食品和支持不同，拟合差距不能单独证明过拟合。数据、指标和接受门槛不随结果调整。',
            'Parameter count and initialization are unchanged. The intervention is calibration weight and the subsequent optimization trajectory. Equal initial shared gradients in a deterministic reference do not imply bitwise repeatability of ordinary-GPU backward passes. One seed cannot uniquely attribute effects to real source bias or establish that calibration is always useless. Different training/validation foods and support prevent fit gaps alone from proving overfitting. Data, metrics and acceptance gates are not adjusted to outcomes.'),
        sub(7),shared['gates'],
        words(('满足' if eligible else '未满足')+'登记的筛选条件；单种子不能确认为稳定改进或优于RF。',
            ('Meets' if eligible else 'Does not meet')+' the registered screen; one seed cannot confirm stable improvement or superiority over RF.'),
        words(
            '此处只呈现自动筛选。接受、拒绝或证据不足须完成实际审阅并另存决定。通过筛选只允许另行登记23/24复验，不替代三种子对冻结RF的最终确认。',
            'This is the automatic screen only. Acceptance, rejection or insufficient evidence requires actual review and a separate decision. Passing permits separately registered seeds23/24, not a substitute for final three-seed confirmation against frozen RF.'),
        sub(8),words(
            '完成审阅后按登记条件决定复验或下一单因素问题。数据和树结果继续冻结，历史测试关闭。来源留出、少样本迁移和可靠foundation model能力仍未验证。保留全部结果和失败，不拼接不同模型的任务最佳成绩。',
            'After review, apply the registered conditions to decide on replication or the next single-factor question. Data and tree results remain frozen and historical test closed. Source-held-out transfer, few-shot transfer and a reliable foundation-model claim remain unverified. Retain all results and failures without combining different models’ best task scores.')
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
        raise ValueError('Wait for complete SOURCE0 analysis and training-fit diagnosis: ' + ', '.join(pending))
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
            or not fit['existing_mae_predictions_rescored_exact'] or not fit['new_source0_predictions_disk_reloaded_exact']):
        raise ValueError('Expected completed controls and fit replay')
    for role in ['mae', 'source0']:
        if fit['source_free_validation'][role] != analysis['records'][role]['metrics']['completion']:
            raise ValueError('Fit and analysis validation metrics differ')
    shared = tables(analysis, fit)
    documents, supplements = {}, {}
    for lang in ['ZH', 'EN']:
        prior = rebase((PRIOR / f'REPORT_{lang}.md').read_text(encoding='utf-8'), PRIOR / f'REPORT_{lang}.md', output)
        prior_scope = prior.splitlines()[4]
        scope = ('草稿：第1–11节保留上一版已审阅的历史证据，其中“最新”和“下一轮”均描述历史时点；第12节新增关闭来源校准目标的证据。本轮实际曲线审阅、因果解释与版本决定尚未完成，不能称为最终报告。' if lang == 'ZH' else
                 'DRAFT: Sections 1–11 retain previously reviewed history; their latest/next experiment statements describe that historical stage. Section12 adds completed source-calibration ablation evidence. Actual curve review, interpretation and the version decision remain incomplete; this is not a final report.')
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
             ROOT / 'reports/v9_r9_source0_functional_v1/verification.json',
             ROOT / 'reports/v9_r9_source0_functional_v1/reference_execution.json',
             ROOT / 'reports/v9_r9_source0_gradient_reference_v1/summary.json',
             ROOT / 'reports/v9_r9_source0_preflight_failure_v1/summary.json']
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
