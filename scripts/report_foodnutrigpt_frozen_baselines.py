"""Export bilingual baseline appendices from frozen completed runs, without refitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as handle:
        result = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
        return result.hexdigest()


def finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite report value.')
    if isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def table(headers, rows):
    return '\n'.join(['|' + '|'.join(headers) + '|', '|' + '|'.join(['---'] * len(headers)) + '|']
                     + ['|' + '|'.join(str(v) for v in row) + '|' for row in rows])


def render(payload, language):
    zh = language == 'ZH'
    title = '冻结RF/XGBoost：方法与结果附录' if zh else 'Frozen RF/XGBoost: methods and results appendix'
    lines = ['# ' + title, '',
        ('状态：已完成的基线附录；整个研究及Transformer比较仍未完成。两种语言由同一summary.json生成，未拟合模型、未改数据、未打开测试。'
         if zh else 'Status: completed baseline appendix; the overall study and Transformer comparison remain incomplete. Both languages use the same summary.json. No models were fitted, data changed or test opened.'), '',
        '## ' + ('共同输入与监督' if zh else 'Shared inputs and supervision'), '',
        ('每项配置独立拟合187个逐轴回归器，覆盖142营养轴及45代谢组轴；65个其他轴只提供已观测上下文。每轴使用全部合格训练档案，无5000行上限。共有1,828,536个训练目标，最大轴58,958行。显式零是标签，缺失不进训练目标，不跨来源合并数值。'
         if zh else 'Each configuration fits 187 separate axis regressors: 142 nutrition and 45 metabolome axes. The other 65 axes provide observed context only. Every eligible training profile is used for each target axis, with no 5,000-row cap. There are 1,828,536 training targets; the largest axis has 58,958 rows. Explicit zeros are labels, missing cells are excluded, and values are not pooled across sources.'), '',
        ('632个输入槽位为128名称槽位、252变换后数值和252可见标记。32维设置只启用相同训练集PCA基底的前32个名称方向，其余96槽置零；128维设置启用全部。名称严格只取food name。补全时目标家族全部隐藏，name-only时数值与可见标记全部隐藏。变换t=log(1+y/s)中的s只由训练集拟合，逐轴来源权重按均值归一后传入sample_weight。'
         if zh else 'The 632 input slots comprise 128 name slots, 252 transformed values and 252 visibility indicators. The 32-dimensional setting uses the first 32 directions of the same training-fitted PCA basis and zeros the other 96 slots; the 128-dimensional setting uses all directions. Text contains only the food name. Completion hides the entire target family; name-only hides all numeric values and visibility indicators. The scale s in t=log(1+y/s) is fitted on training data only. Per-axis source-balanced weights are divided by their mean and supplied as sample_weight.'), '',
        '## ' + ('拟合方法及实际调参范围' if zh else 'Fitting methods and realized search'), '',
        ('RF使用scikit-learn 1.5.2：400棵bootstrap树、平方误差、无深度上限、min_samples_leaf=1、max_features=0.5；32/128输入各完成一项。XGBoost 2.1.3使用800棵树、hist、learning_rate=0.03、min_child_weight=5、subsample=0.8、colsample_bytree=0.8、reg_lambda=1、reg:squarederror，不使用early stopping；深度及输入见表。两类逐轴random_state=20260922+axis_index、拟合4线程。RF推理按固定树顺序串行累加。完整get_params保存在summary.json，不把未显式设置的库默认参数冒充搜索结果。'
         if zh else 'RF uses scikit-learn 1.5.2: 400 bootstrap trees, squared error, unlimited depth, min_samples_leaf=1 and max_features=0.5, with completed 32- and 128-dimensional runs. XGBoost 2.1.3 uses 800 trees, hist, learning_rate=0.03, min_child_weight=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1 and reg:squarederror, without early stopping; depths and input dimensions appear below. Both methods use random_state=20260922+axis_index and four fitting threads. RF inference accumulates trees serially in a fixed order. Full get_params records are preserved in summary.json; library defaults are not presented as tuned choices.'), '',
        ('原登记8项树配置中6项完整；RF leaf3/feature0.5/name128在123/187轴中断，RF leaf1/feature1.0/name128未运行，原因是用户冻结树实验。二者无完整分数，不参加选优，不能宣称RF的3配置搜索完成。没有新增树种子；下列都是seed20260922的固定参照，不提供虚构的种子标准差。'
         if zh else 'Six of eight registered tree configurations completed. RF leaf3/feature0.5/name128 stopped at 123/187 axes and RF leaf1/feature1.0/name128 did not run because the user froze tree experiments. Neither has a complete score or enters selection; the three-configuration RF search was not completed. No additional tree seeds were run: all rows below are fixed seed20260922 references, without an invented seed standard deviation.'), '']
    recipe_rows, nutrition_rows, detail_rows, retrieval_rows = [], [], [], []
    for row in payload['runs']:
        fit = row['common_fit_parameters']
        recipe_rows.append([row['name'], row['name_dimensions'], fit['n_estimators'],
            'unlimited' if row['kind'] == 'rf' else fit['max_depth'],
            fit.get('min_samples_leaf', 'N/A'), fit.get('max_features', 'N/A'),
            f"{row['sum_axis_preparation_and_fit_seconds']:.1f}", f"{row['elapsed_seconds']:.1f}"])
        a, n = row['metrics']['completion'], row['metrics']['name_only']
        nutrition_rows.append([row['name']] + [f'{v:.6f}' for v in [a['nutrition']['scaled_log_mae'],
            a['nutrition']['log_mae'], n['nutrition']['scaled_log_mae'],
            a['food_metabolome']['scaled_log_mae'], a['all']['scaled_log_mae']]])
        detail_rows.append([row['name']] + [f"{a['nutrition'][key]:.6f}" for key in
            ['raw_mae', 'positive_scaled_log_mae', 'zero_scaled_log_mae']])
        for ret in row['retrieval']['metrics']:
            retrieval_rows.append([row['name'], ret['visible_fraction']] +
                [f'{ret[key]:.6f}' for key in ['mrr', 'recall_at_1', 'recall_at_5', 'recall_at_10']])
    lines += [table(['Run', 'Name dim', 'Trees', 'Depth', 'Min leaf', 'Feature fraction', 'Preparation + fit sum (s)', 'Run elapsed (s)'], recipe_rows), '',
        ('时间为各作业实测：逐轴准备与拟合阶段耗时之和，以及包括预测/核验的整次作业耗时。原始fit_seconds计时从特征构造之前开始，包含准备和检查，不能称为纯fit调用耗时。此前存在并发计算，也不能当作严格控制的算法速度对照。'
         if zh else 'Costs are measured per run: the sum of per-axis preparation-and-fitting times and total elapsed time including prediction and checks. The recorded fit_seconds starts before feature construction and includes preparation and checks, so it is not isolated fit-call time. Earlier concurrent workloads also prevent treating these as a controlled algorithm-speed comparison.'), '',
        '## ' + ('营养预测结果' if zh else 'Nutrition prediction results'), '',
        ('全部是内部验证结果。主指标是142轴宏平均缩放log-MAE，先在食品组×来源层面等权。误差越低越好；不同名称维度分开识别，不将跨维度差异全部归因于架构。'
         if zh else 'All results are internal validation. The primary metric is macro scaled-log MAE over 142 axes, with food-group and within-group source balancing. Lower is better. Name dimensions are identified separately; cross-dimension differences are not attributed solely to architecture.'), '',
        table(['Run', 'Completion 142', 'Legacy log 142', 'Name-only 142', 'Completion 45', 'Completion 187'], nutrition_rows), '',
        table(['Run', 'Raw MAE (g/100g)', 'Positive scaled MAE', 'Zero scaled MAE'], detail_rows), '',
        ('正值与零值条件误差分母不同，不能相加得到整体主指标。全部任务和子集的原始精度指标见summary.json。'
         if zh else 'Positive and zero conditional errors use different denominators and cannot be added to recover the primary metric. Full-precision metrics for every task and subset are in summary.json.'), '',
        '## ' + ('营养到名称检索' if zh else 'Nutrition-to-name retrieval'), '',
        ('同一套已拟合逐轴模型只根据候选名称生成营养向量；查询不含名称，候选不使用真实营养档案。候选库固定49,913个名称。0.3/1.0可见比例分别有8,610/10,479个合格查询，要求至少3个已知营养轴。名称正确性按完全相同原名，尚无已确认别名映射。下表全部为0–1比例；不同可见比例的查询群体不同，不是纯可见性对照。'
         if zh else 'The same fitted axis models generate candidate nutrition vectors using candidate names alone. Queries contain no name and candidates use no measured nutrition profiles. The library has 49,913 names. Visibility fractions 0.3/1.0 have 8,610/10,479 eligible queries, requiring at least three observed nutrition axes. Relevance uses the exact original name; no confirmed alias map exists. All numbers below are fractions in [0,1]. The visibility strata have different eligible query populations and are not a pure visibility intervention.'), '',
        table(['Run', 'Visible fraction', 'MRR', 'R@1', 'R@5', 'R@10'], retrieval_rows), '',
        '## ' + ('解释与复现边界' if zh else 'Interpretation and reproducibility'), '',
        ('R9同输入32维主要参照固定为RF leaf1/feature0.5及XGB depth10。它们是已完成集合中的参照，不代表完整搜索或所有随机种子的最优结果。尚未产生本轮Transformer完整结果；本附录不宣布神经模型胜负，也不证明标签全部可靠或foundation能力。'
         if zh else 'The matched 32-dimensional R9 references are frozen RF leaf1/feature0.5 and XGB depth10. They are references from the completed set, not optima of the unfinished search or across all random seeds. Full Transformer results for this round are not yet available. This appendix establishes neither a neural win nor universal label validity or foundation-model capability.'), '',
        ('逐轴训练时已检查反序、子批次和内存pickle重载，完成审计也核对了数据、特征、权重及保存的预测/检索排名。森林在预测后释放，没有保存完整森林，因此不能说当前产物支持任意新名称在线树推理。数字预测与候选矩阵保留在本地忽略目录；附录仅含汇总。'
         if zh else 'Per-axis execution checked reversed order, subbatches and in-memory pickle replay; completed audits also verified data, features, weights and saved predictions/retrieval ranks. Fitted forests were discarded after prediction, so saved artifacts do not support arbitrary new-name online tree inference. Numeric predictions and candidate matrices remain in local ignored directories; this appendix contains aggregates only.'), '',
        f"Data manifest SHA256: `{payload['data_sha256']}`",
        '', f"Freeze manifest SHA256: `{payload['freeze_sha256']}`", '',
        ('各运行代码、环境、完整参数、运行及拟合记录哈希见[机器结果](summary.json)。两种语言所有数值表格一致；已完成附录不等于整体终稿。'
         if zh else 'Code, environment, full parameters, run hashes and fit-record hashes are in the [machine-readable results](summary.json). Every numeric table is identical across languages. A completed appendix is not the final study report.'), '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    freeze_path = ROOT / 'reports/v9_r9_freeze_v1/manifest.json'
    freeze = json.loads(freeze_path.read_text())
    if freeze['status'] != 'frozen' or freeze['complete_test_opened']:
        raise ValueError('Expected frozen test-closed references.')
    hashes = {freeze_path.relative_to(ROOT).as_posix(): digest(freeze_path)}
    def read(path, require_frozen=True):
        relative = path.relative_to(ROOT).as_posix()
        actual = digest(path)
        if require_frozen and actual != freeze['input_hashes'][relative]:
            raise ValueError('Changed reference: ' + relative)
        hashes[relative] = actual
        result = json.loads(path.read_text(encoding='utf-8'))
        finite(result)
        return result
    records = []
    for name, baseline in freeze['baselines'].items():
        run = ROOT / baseline['directory']
        manifest = read(run / 'run_manifest.json')
        metrics = read(run / 'metrics.json')
        retrieval = read(run / 'retrieval/metrics.json')
        receipts = read(run / 'axis_fitting_manifest.json', require_frozen=False)
        if (manifest['status'] != 'complete' or manifest['complete_test_opened']
                or manifest['axes_completed'] != 187 or len(receipts) != 187
                or manifest['training_row_cap'] is not None or manifest['seed'] != 20260922):
            raise ValueError('Expected a complete, uncapped fixed-seed baseline.')
        if metrics != baseline['metrics'] or manifest['data_hash'] != freeze['data_hash']:
            raise ValueError('Baseline identity differs from frozen reference.')
        if retrieval['complete_test_opened'] or retrieval['candidate_count'] != 49913:
            raise ValueError('Unexpected retrieval protocol.')
        if retrieval['query_profiles'] != {'0.3': 8610, '1.0': 10479}:
            raise ValueError('Unexpected query counts.')
        common = {k: v for k, v in receipts[0]['fit_parameters'].items() if k != 'random_state'}
        if len({r['axis_index'] for r in receipts}) != 187:
            raise ValueError('Duplicate axis receipt.')
        for item in receipts:
            parameters = dict(item['fit_parameters'])
            if parameters.pop('random_state') != manifest['seed'] + item['axis_index'] or parameters != common:
                raise ValueError('Inconsistent per-axis parameters.')
        if sum(r['train_profiles'] for r in receipts) != 1828536 or max(r['train_profiles'] for r in receipts) != 58958:
            raise ValueError('Unexpected fitting support.')
        records.append({'name': name, 'kind': baseline['kind'], 'configuration': baseline['configuration'],
            'name_dimensions': baseline['active_name_dimensions'], 'seed': manifest['seed'],
            'run': baseline['directory'], 'common_fit_parameters': common,
            'axis_seed_rule': '20260922 + axis_index', 'code_commit': manifest['code_commit'],
            'environment': manifest['environment'], 'metrics': metrics, 'retrieval': retrieval,
            'elapsed_seconds': manifest['elapsed_seconds'],
            'sum_axis_preparation_and_fit_seconds': sum(r['fit_seconds'] for r in receipts),
            'manifest_sha256': digest(run / 'run_manifest.json'),
            'fitting_receipts_sha256': digest(run / 'axis_fitting_manifest.json'),
            'prior_completed_audit_sha256': baseline['audit_sha256']})
    payload = {'status': 'completed_frozen_baseline_appendix_only', 'runs': records,
        'data_sha256': freeze['data_hash'], 'freeze_sha256': digest(freeze_path), 'input_hashes': hashes,
        'script_sha256': digest(Path(__file__)), 'complete_test_opened': False,
        'baseline_refit': False, 'data_modified': False, 'overall_report_complete': False,
        'uncompleted_registered_runs': {'rf400leaf3half_name128': 'interrupted123of187_by_user_scope_change',
            'rf400leaf1all_name128': 'not_started_before_user_freeze'}}
    finite(payload)
    docs = {language: render(payload, language) for language in ['ZH', 'EN']}
    tables = {language: [line for line in content.splitlines() if line.startswith('|')] for language, content in docs.items()}
    if tables['ZH'] != tables['EN']:
        raise ValueError('Bilingual numeric tables differ.')
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Report input changed during generation: ' + name)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for language, content in docs.items():
        (args.output_dir / f'BASELINES_{language}.md').write_text(content, encoding='utf-8')
    print(json.dumps({'status': payload['status'], 'completed_runs': len(records),
        'bilingual_numeric_tables_identical': True, 'baseline_refit': False, 'output_dir': str(args.output_dir)}))


if __name__ == '__main__':
    main()
