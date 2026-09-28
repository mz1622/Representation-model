"""Audited final-model comparisons without a superiority gate or test access."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import pandas as pd
from foodcomp.research_completion_input import execution_contract
from foodcomp.research_confirmation import SEEDS
from foodcomp.research_final_statistics import (RATES, RATE_KEYS, CELL_KEYS, ERRORS,
    CONTRIBUTIONS, mean_seed_frames, number_summary, prediction_details, retrieval_groups,
    retrieval_identity_for_run)
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_r8_summary import select_registered_references
from foodcomp.research_statistics import paired_interval, paired_axis_intervals, paired_group_rates


def local(path):
    path = Path(str(path).replace('\\', '/'))
    return path if path.is_absolute() else ROOT / path


class Inputs:
    def __init__(self):
        self.hashes = {}

    def record(self, path, expected=None):
        path = local(path)
        value = digest(path)
        if expected is not None and value != expected:
            raise ValueError('Stale input: ' + str(path))
        key = path.relative_to(ROOT).as_posix()
        if key in self.hashes and self.hashes[key] != value:
            raise ValueError('Input changed while reading: ' + key)
        self.hashes[key] = value
        return path

    def read(self, path, expected=None):
        return json.loads(self.record(path, expected).read_text(encoding='utf-8'))

    def verify(self):
        for path, expected in self.hashes.items():
            if digest(ROOT / path) != expected:
                raise ValueError('Input changed during statistics: ' + path)


def closed(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {'test_opened', 'complete_test_opened'} and item is not False:
                raise ValueError('Test-open metadata is prohibited.')
            closed(item)
    elif isinstance(value, list):
        for item in value:
            closed(item)


def completed(value):
    closed(value)
    if value['status'] != 'complete':
        raise ValueError('An input run or audit is incomplete.')


def audit_match(inputs, path, run):
    audit = inputs.read(path)
    completed(audit)
    matched = [r for r in audit['records'] if local(r['run']).resolve() == run.resolve()]
    if len(matched) != 1:
        raise ValueError('Audit does not identify exactly one requested run.')
    inputs.record(run / 'run_manifest.json', matched[0]['manifest_sha256'])


def load_models(args, inputs, contract, contract_path):
    audit = inputs.read(args.neural_audit / 'verification.json')
    completed(audit)
    plan_path = ROOT / 'experiments/foodnutrigpt_v9_research/final_report_v1/config.json'
    inputs.record(plan_path, audit['plan_sha256'])
    models = {'mlp': {}}
    for record in audit['records']:
        seed, run, retrieval = record['seed'], local(record['run']), local(record['retrieval_directory'])
        if seed in models['mlp']:
            raise ValueError('Duplicate neural seed.')
        manifest = inputs.read(run / 'run_manifest.json', record['manifest_sha256'])
        inputs.record(run / 'best_model.pt', record['checkpoint_sha256'])
        for task in ['completion', 'name_only']:
            inputs.record(run / f'{task}_predictions.parquet', record['prediction_sha256'][task])
        scores = inputs.read(run / 'metrics.json', record['metrics_sha256'])
        inputs.record(retrieval / 'metrics.json', record['retrieval_metrics_sha256'])
        inputs.record(retrieval / 'ranks.parquet', record['retrieval_ranks_sha256'])
        if scores != record['metrics']:
            raise ValueError('Neural metrics differ from the independent audit.')
        models['mlp'][seed] = {'run': run, 'retrieval': retrieval, 'manifest': manifest, 'scores': scores}
    if set(models['mlp']) != set(SEEDS):
        raise ValueError('Missing registered neural seeds.')
    selection = None
    if not args.neural_only:
        summary = inputs.read(args.r8_summary)
        if (summary['status'] != 'all_registered_outputs_audited_report_and_decision_pending'
                or summary['registered_count'] != 12):
            raise ValueError('Full registered R8 search is required.')
        closed(summary)
        for path, expected in summary['artifact_hashes'].items():
            inputs.record(path, expected)
        selection = select_registered_references(contract['registered_candidates'], summary['records'])
        # The full collector appends paired-interval evidence to the selector result.
        if any(summary['selection'].get(k) != v for k, v in selection.items()):
            raise ValueError('Changed R8 selection.')
        selection = summary['selection']
        queue = inputs.read(args.tree_repeats / 'queue_manifest.json')
        completed(queue)
        inputs.record(args.r8_summary, queue['r8_summary_sha256'])
        inputs.record(plan_path, queue['plan_sha256'])
        if (queue['selected_by_method'] != selection['selected_by_method']
                or queue['seeds'] != list(SEEDS) or len(queue['runs']) != 4):
            raise ValueError('Tree repetition coverage or selection changed.')
        for kind in ['rf', 'xgb']:
            models[kind] = {}
            parent = queue['seed22_parents'][kind]
            run = local(parent['directory'])
            manifest = inputs.read(run / 'run_manifest.json', parent['manifest_sha256'])
            selected = summary['records'][selection['selected_by_method'][kind]]
            if manifest != selected['manifest'] or run.resolve() != local(selected['directory']).resolve():
                raise ValueError('Seed22 tree parent changed.')
            models[kind][SEEDS[0]] = {'run': run, 'retrieval': run / 'retrieval',
                'manifest': manifest, 'scores': inputs.read(run / 'metrics.json')}
            for record in [r for r in queue['runs'] if r['kind'] == kind]:
                seed, run = record['seed'], local(record['directory'])
                if seed not in SEEDS[1:] or seed in models[kind]:
                    raise ValueError('Invalid or duplicate tree seed.')
                current = inputs.read(run / 'run_manifest.json', record['manifest_sha256'])
                audit_path = local(record['audit_directory']) / 'verification.json'
                inputs.record(audit_path, record['audit_sha256'])
                audit_match(inputs, audit_path, run)
                for field in ['configuration', 'code_commit', 'code_hashes', 'execution_contract_sha256',
                              'data_hash', 'panel_hash', 'name_cache_hash', 'input_slots',
                              'active_name_dimensions', 'fit_n_jobs', 'rf_prediction_n_jobs', 'training_row_cap']:
                    if current[field] != manifest[field]:
                        raise ValueError('Tree settings differ across seeds: ' + field)
                models[kind][seed] = {'run': run, 'retrieval': run / 'retrieval', 'manifest': current,
                                     'scores': inputs.read(run / 'metrics.json')}
    common = None
    for method, runs in models.items():
        if set(runs) != set(SEEDS):
            raise ValueError('Incomplete model repetitions: ' + method)
        for seed, spec in runs.items():
            m = spec['manifest']
            completed(m)
            closed(spec['scores'])
            identity = [m[k] for k in ['data_hash', 'panel_hash', 'name_cache_hash',
                                      'input_slots', 'active_name_dimensions', 'code_commit']]
            if common is None:
                common = identity
            if identity != common or m['seed'] != seed or identity[3:5] != [632, 128]:
                raise ValueError('Different model inputs or seed.')
            if m['execution_contract_sha256'] != digest(contract_path):
                raise ValueError('Changed frozen training contract.')
    knn = ROOT / 'output/v9_r7/exact_name_knn128'
    manifest = inputs.read(knn / 'run_manifest.json')
    completed(manifest)
    if manifest['name_cache_hash'] != common[2] or manifest['data_hash'] != common[0]:
        raise ValueError('Different KNN input protocol.')
    audit_match(inputs, ROOT / 'reports/v9_r7_knn_completed_audit_v1/verification.json', knn)
    inputs.record(knn / 'nutrition_predictions.parquet', manifest['nutrition_prediction_sha256'])
    scores = inputs.read(knn / 'nutrition_metrics.json')
    models['name_knn'] = {None: {'run': knn, 'retrieval': knn, 'manifest': manifest,
        'scores': {'completion': scores, 'name_only': scores}}}
    return models, selection


def run_statistics(args, inputs, models, selection):
    data = ResearchData(ROOT / 'data/processed' / VERSION)
    inputs.record(data.root / 'manifest.json')
    axes = data.axes.loc[data.axes.loss_group.eq('nutrition') & data.axes.loss_eligible, 'axis_index'].to_numpy()
    all_axes, all_sources, scalar_rows, rank_rows, costs = [], [], [], [], []
    grouped, rank_groups, model_summary = {}, {}, {}
    query_reference, retrieval_identity = None, None
    for method, runs in models.items():
        grouped[method], rank_groups[method] = {}, {}
        model_summary[method] = {'tasks': {}, 'retrieval': {}, 'runs': []}
        for seed, spec in runs.items():
            run, ret, manifest = spec['run'], spec['retrieval'], spec['manifest']
            grouped[method][seed] = {}
            model_summary[method]['runs'].append({'seed': seed, 'directory': run.relative_to(ROOT).as_posix(),
                'manifest_sha256': digest(run / 'run_manifest.json'),
                'best_epoch': manifest.get('best_epoch'), 'elapsed_seconds': manifest['elapsed_seconds']})
            costs.append({'method': method, 'seed': seed, 'recorded_seconds': manifest['elapsed_seconds']})
            for task in ['completion', 'name_only']:
                path = run / ('nutrition_predictions.parquet' if method == 'name_knn' else f'{task}_predictions.parquet')
                predictions = pd.read_parquet(inputs.record(path))
                score, axis_frame, groups, source_frame = prediction_details(data, predictions)
                if score != spec['scores'][task]:
                    raise ValueError(f'Saved metrics cannot be reproduced: {method}/{seed}/{task}')
                grouped[method][seed][task] = groups
                all_axes.append(axis_frame.assign(method=method, seed=seed, task=task))
                all_sources.append(source_frame.assign(method=method, seed=seed, task=task))
                for subset in ['nutrition', 'food_metabolome', 'all']:
                    scalar_rows.append({'method': method, 'seed': seed, 'task': task, 'subset': subset, **score[subset]})
            metadata = inputs.read(ret / 'metrics.json')
            closed(metadata)
            inputs.record(ret / 'candidate_names.json', metadata['candidate_sha256'])
            identity = retrieval_identity_for_run(metadata, manifest)
            if retrieval_identity is None:
                retrieval_identity = identity
            if identity != retrieval_identity or identity['candidate_count'] != 49913:
                raise ValueError('Retrieval protocols differ.')
            query_keys, groups = retrieval_groups(pd.read_parquet(inputs.record(ret / 'ranks.parquet')), metadata)
            if query_reference is None:
                query_reference = query_keys
            pd.testing.assert_frame_equal(query_reference, query_keys)
            rank_groups[method][seed] = groups
            expected = {str(float(r['visible_fraction'])): r for r in metadata['metrics']}
            for fraction, frame in groups.items():
                rates = frame[RATES].mean().to_dict()
                np.testing.assert_allclose([rates[k] for k in RATES], [expected[fraction][k] for k in RATES],
                                           rtol=1e-12, atol=1e-15)
                rank_rows.append({'method': method, 'seed': seed, 'visible_fraction': fraction, **rates})
            print(f'Fully rescored {method} seed {seed}: both regression tasks and retrieval', flush=True)
    scalar_frame, rank_frame = pd.DataFrame(scalar_rows), pd.DataFrame(rank_rows)
    averages, rate_averages = {}, {}
    for method, runs in models.items():
        averages[method], rate_averages[method] = {}, {}
        for task in ['completion', 'name_only']:
            frames = {seed: grouped[method][seed][task] for seed in runs}
            averages[method][task] = frames[None] if method == 'name_knn' else mean_seed_frames(frames, CELL_KEYS, ERRORS + CONTRIBUTIONS)
            model_summary[method]['tasks'][task] = {}
            for subset in ['nutrition', 'food_metabolome', 'all']:
                frame = scalar_frame[(scalar_frame.method == method) & (scalar_frame.task == task) & (scalar_frame.subset == subset)]
                metrics = [c for c in frame.columns if c not in ['method', 'seed', 'task', 'subset', 'axes']]
                model_summary[method]['tasks'][task][subset] = {m: number_summary(frame[m]) for m in metrics}
        for fraction in ['0.3', '1.0']:
            frames = {seed: rank_groups[method][seed][fraction] for seed in runs}
            rate_averages[method][fraction] = frames[None] if method == 'name_knn' else mean_seed_frames(frames, ['exact_name_group_id'], RATES)
            frame = rank_frame[(rank_frame.method == method) & (rank_frame.visible_fraction == fraction)]
            model_summary[method]['retrieval'][fraction] = {m: number_summary(frame[m]) for m in RATES}
    comparisons, per_axis = {}, []
    for baseline in [m for m in models if m != 'mlp']:
        comparisons[baseline] = {'tasks': {}, 'retrieval': {}}
        for task in ['completion', 'name_only']:
            base, candidate = averages[baseline][task], averages['mlp'][task]
            intervals = {m: paired_interval(base, candidate, axes, metric=m, repeats=args.repeats) for m in ERRORS[:2]}
            pieces = []
            for frame in [base, candidate]:
                pieces.append(frame[frame.axis_index.isin(axes)].groupby('axis_index')[CONTRIBUTIONS].mean().mean())
            decomposition = (pieces[1] - pieces[0]).to_dict()
            np.testing.assert_allclose(sum(decomposition.values()),
                intervals['scaled_log_mae']['candidate'] - intervals['scaled_log_mae']['baseline'], rtol=1e-10, atol=1e-12)
            comparisons[baseline]['tasks'][task] = {'paired_intervals': intervals,
                'primary_candidate_minus_baseline_decomposition': decomposition}
            per_axis.append(paired_axis_intervals(base, candidate, repeats=args.repeats)
                            .assign(baseline=baseline, candidate='mlp', task=task))
        for fraction in ['0.3', '1.0']:
            comparisons[baseline]['retrieval'][fraction] = paired_group_rates(
                rate_averages[baseline][fraction], rate_averages['mlp'][fraction], RATES, repeats=args.repeats)
        print(f'Food-group intervals complete: MLP versus {baseline}', flush=True)
    inputs.verify()
    result = {'status': 'neural_reference_statistics_complete' if args.neural_only else 'final_comparison_statistics_complete',
        'final_report_complete': False, 'complete_test_opened': False, 'seeds': list(SEEDS),
        'bootstrap_repeats': args.repeats, 'bootstrap_seed': SEEDS[0], 'models': model_summary,
        'comparisons': comparisons, 'r8_selection_historical': selection, 'retrieval_protocol': retrieval_identity,
        'superiority_required_for_completion': False, 'input_hashes': inputs.hashes,
        'scope': 'Conditional internal-validation comparisons. Average independently scored errors/rates across three fixed seeds, then resample whole food groups. No prediction ensemble or best-seed selection. Seed SD reported separately; intervals do not quantify selection bias, seed-population uncertainty or label validity. KNN is one deterministic reference, not three fictitious seeds. Source tables have different axis coverage. Per-axis intervals have no multiplicity correction.',
        'workspace_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'script_sha256': digest(Path(__file__)),
        'statistics_source_sha256': {p: digest(ROOT / p) for p in ['src/foodcomp/research_final_statistics.py',
            'src/foodcomp/research_statistics.py', 'src/foodcomp/research_confirmation.py']}}
    scalar_frame.to_csv(args.output_dir / 'metrics_by_seed.csv', index=False)
    rank_frame.to_csv(args.output_dir / 'retrieval_by_seed.csv', index=False)
    pd.concat(all_axes, ignore_index=True).to_csv(args.output_dir / 'axis_metrics_by_seed.csv', index=False)
    pd.concat(all_sources, ignore_index=True).to_csv(args.output_dir / 'source_metrics_by_seed.csv', index=False)
    pd.concat(per_axis, ignore_index=True).merge(data.axes[['axis_index', 'canonical_name', 'loss_group']],
        on='axis_index', validate='many_to_one').to_csv(args.output_dir / 'axis_paired_intervals.csv', index=False)
    pd.DataFrame(costs).to_csv(args.output_dir / 'recorded_compute_seconds.csv', index=False)
    write_json(args.output_dir / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--r8-summary', type=Path, default=ROOT / 'reports/v9_r8_round_complete_v1/summary.json')
    parser.add_argument('--tree-repeats', type=Path, default=ROOT / 'output/v9_final_tree_replication_v1')
    parser.add_argument('--neural-audit', type=Path, default=ROOT / 'reports/v9_final_neural_audit_v1')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--neural-only', action='store_true', help='Explicit partial stage: neural versus deterministic KNN only.')
    parser.add_argument('--check-only', action='store_true', help='Validate input readiness without producing comparisons.')
    args = parser.parse_args()
    if args.repeats < 1000:
        raise ValueError('Formal group intervals require at least 1000 resamples.')
    required = [args.neural_audit / 'verification.json']
    if not args.neural_only:
        required += [args.r8_summary, args.tree_repeats / 'queue_manifest.json']
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        if args.check_only:
            print(json.dumps({'ready': False, 'missing': missing, 'outputs_written': False}))
            return
        raise FileNotFoundError('Missing complete evidence: ' + '; '.join(missing))
    inputs = Inputs()
    contract, path = execution_contract(ROOT)
    inputs.record(path)
    models, selection = load_models(args, inputs, contract, path)
    if args.check_only:
        inputs.verify()
        print(json.dumps({'ready': True, 'methods': list(models), 'outputs_written': False}))
        return
    if args.output_dir.exists():
        raise FileExistsError('Never overwrite a partial or completed statistics directory.')
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    write_json(args.output_dir / 'status.json', {'status': 'running', 'complete_test_opened': False})
    try:
        result = run_statistics(args, inputs, models, selection)
        write_json(args.output_dir / 'status.json', {'status': 'complete', 'elapsed_seconds': time.monotonic() - started,
            'summary_sha256': digest(args.output_dir / 'summary.json'), 'complete_test_opened': False,
            'final_report_complete': False})
        print(json.dumps({'status': result['status'], 'models': list(result['models'])}))
    except Exception as error:
        write_json(args.output_dir / 'status.json', {'status': 'failed', 'elapsed_seconds': time.monotonic() - started,
            'error_type': type(error).__name__, 'error': str(error), 'complete_test_opened': False})
        raise


if __name__ == '__main__':
    main()
