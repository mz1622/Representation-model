"""Close the registered R7 block only after both matched methods pass full audits."""
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import digest, write_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def comparison(name, retrieval=False):
    folder = ROOT / f'reports/v9_r7_{name}_v1'
    result = read(folder / 'summary.json')
    result['report_sha256'] = digest(folder / 'summary.json')
    if not retrieval:
        axes = pd.read_csv(folder / 'axis_paired_intervals.csv')
        axes = axes[axes.loss_group.eq('nutrition')]
        sources = pd.read_csv(folder / 'source_metrics.csv').pivot(
            index='source', columns='role', values='scaled_log_mae')
        result['improvement_counts'] = {
            'nutrition_axes_point_better': int((axes.candidate_minus_baseline < 0).sum()),
            'nutrition_axes_interval_better': int((axes.difference_95_high < 0).sum()),
            'nutrition_axes_interval_worse': int((axes.difference_95_low > 0).sum()),
            'sources_point_better': int((sources.candidate < sources.baseline).sum()),
            'sources': len(sources),
            'scope': 'Exploratory axis intervals; no multiplicity correction.'}
    return result


def main():
    dest = ROOT / 'experiments/foodnutrigpt_v9_research/r7/results_summary.json'
    figures = ROOT / 'reports/v9_r7_name_projection_figures_v1'
    if dest.exists() or figures.exists():
        raise FileExistsError('Use a new version for any repeated summary.')
    records = []
    audits = {}
    for kind in ['knn', 'mlp']:
        path = ROOT / f'reports/v9_r7_{kind}_completed_audit_v1/verification.json'
        audits[kind] = read(path)
        assert audits[kind]['status'] == 'complete'
        audits[kind]['report_sha256'] = digest(path)
        for dims in [32, 128]:
            run = ROOT / f'output/v9_r7/exact_name_{kind}{dims}'
            manifest = read(run / 'run_manifest.json')
            assert manifest['status'] == 'complete' and not manifest['complete_test_opened']
            if kind == 'knn':
                scores = read(run / 'nutrition_metrics.json')
                metrics = {'completion': scores, 'name_only': scores}
                retrieval = read(run / 'metrics.json')
            else:
                metrics = read(run / 'metrics.json')
                retrieval = read(ROOT / f'output/v9_r7/retrieval_name_mlp{dims}/metrics.json')
            records.append({'kind': kind, 'active_components': dims,
                'directory': str(run.relative_to(ROOT)), 'manifest': manifest,
                'manifest_sha256': digest(run / 'run_manifest.json'),
                'metrics': metrics, 'retrieval': retrieval})
    pairs = {}
    for kind in ['knn', 'mlp']:
        for task in ['name_only', 'retrieval']:
            key = f'{kind}128_vs32_{task}'
            pairs[key] = comparison(key, retrieval=task == 'retrieval')
        for dims in [32, 128]:
            key = f'{kind}{dims}_vs_legacy' + ('_mae' if kind == 'mlp' else '')
            pairs[key] = comparison(key)
    for dims in [32, 128]:
        for suffix in ['', '_retrieval']:
            key = f'mlp{dims}_vs_knn{dims}{suffix}'
            pairs[key] = comparison(key, retrieval=bool(suffix))
    # A dimension direction can pass screening without the model beating its baseline.
    screens = {}
    for kind in ['knn', 'mlp']:
        ci = pairs[f'{kind}128_vs32_name_only']['paired_intervals']['scaled_log_mae']
        screens[kind] = ci['relative_improvement'] >= .01 and ci['relative_improvement_95_interval'][0] > 0
    historical = {
        'name_knn32': {'nutrition': read(ROOT / 'output/v9_r5/retrieval_name_knn10_tree/nutrition_metrics.json'),
            'retrieval': read(ROOT / 'output/v9_r5/retrieval_name_knn10_tree/metrics.json')},
        'name_mlp32': {'metrics': read(ROOT / 'output/v9_r5/name_mlp60_mae/metrics.json'),
            'retrieval': read(ROOT / 'output/v9_r5/retrieval_name_mlp60_mae/metrics.json')}}
    fit = read(ROOT / 'reports/v9_r7_name_fit_v1/summary.json')
    diagnosis = read(ROOT / 'data/local/research_diagnostics/v9_r7_name_projection_v1/summary.json')
    assert diagnosis['status'] == 'complete' and not diagnosis['complete_test_opened']
    figures.mkdir(parents=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    colors = {'knn': '#297a68', 'mlp': '#4d67ad'}
    for kind in ['knn', 'mlp']:
        subset = [r for r in records if r['kind'] == kind]
        for metric, ax in [('scaled_log_mae', axs[0, 0]), ('log_mae', axs[0, 1])]:
            ax.plot([32, 128], [r['metrics']['name_only']['nutrition'][metric] for r in subset],
                marker='o', color=colors[kind], label=kind.upper())
        for fraction, ax in [(1., axs[1, 0]), (.3, axs[1, 1])]:
            ax.plot([32, 128], [next(m['recall_at_10'] for m in r['retrieval']['metrics']
                if m['visible_fraction'] == fraction) for r in subset], marker='o',
                color=colors[kind], label=kind.upper())
    for ax, title, ylabel in zip(axs.flat,
            ['Name-only nutrition', 'Legacy metric guard', 'Full observed nutrition retrieval', '30% visible nutrition retrieval'],
            ['142-axis scaled-log MAE (lower better)', '142-axis log-MAE (lower better)',
             'Recall@10 (higher better)', 'Recall@10 (higher better)']):
        ax.set(title=title, ylabel=ylabel, xlabel='Active name dimensions')
        ax.set_xticks([32, 128]); ax.grid(alpha=.2); ax.legend()
    fig.suptitle('R7: same exact PCA basis, same targets and fixed evaluation panels\n'
        'MLP has one training seed; KNN is deterministic. No completion superiority claim.', fontsize=12)
    for ext in ['png', 'svg']:
        fig.savefig(figures / f'name_projection.{ext}', dpi=160)
    result = {'version': 'V9-R7', 'status': 'registered_six_candidate_block_complete_single_seed',
        'config_sha256_at_summary': digest(ROOT / 'experiments/foodnutrigpt_v9_research/r7/config.json'),
        'script_sha256': digest(Path(__file__)), 'audits': audits, 'records': records,
        'historical_references': historical, 'comparisons': pairs, 'training_fit': fit,
        'gap_diagnosis_aggregate_only': diagnosis,
        'dimension_screen_passed': screens,
        'new_mlp_training_seconds': sum(r['manifest']['elapsed_seconds'] for r in records if r['kind'] == 'mlp'),
        'new_knn_fit_and_evaluation_seconds': sum(r['manifest']['elapsed_seconds'] for r in records if r['kind'] == 'knn'),
        'complete_test_opened': False, 'scientific_confirmation': False,
        'neural_improvement_accepted': False,
        'scope': 'Dimension screening only. Same-input strong KNN required for each neural comparison. '
            'Old32 trees cannot establish new128 completion superiority; training-seed and selection uncertainty remain.'}
    write_json(dest, result)
    print('Completed R7 summary; dimension screens:', screens)
    for key, pair in pairs.items():
        if 'paired_intervals' in pair:
            ci = pair['paired_intervals']['scaled_log_mae']
            print(key, ci['relative_improvement'], ci['relative_improvement_95_interval'],
                pair['primary_change_decomposition'], pair['improvement_counts'])


if __name__ == '__main__':
    main()
