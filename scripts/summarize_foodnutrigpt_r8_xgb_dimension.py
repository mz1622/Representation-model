"""Audit and display the completed fixed-depth R8 dimension comparison; no selection."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(handle.read()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    hashes = {}

    def read(relative):
        path = ROOT / relative
        hashes[relative] = digest(path)
        return json.loads(path.read_text(encoding='utf-8'))

    comparisons = {}
    for model, prefix in [('MLP', 'mlp128_vs32'), ('XGB d10', 'xgb_d10_128_vs32')]:
        for task in ['completion', 'name_only', 'retrieval']:
            relative = f'reports/v9_r8_{prefix}_{task}_v1/summary.json'
            record = read(relative)
            assert record['complete_test_opened'] is False
            assert record['confirmation'] is False
            assert record['data_sha256'] == '48aa0b22a816cc175c8d274628a747ca44a90339bda46d33aada7cee9ea9d923'
            for role in ['baseline', 'candidate']:
                if task != 'retrieval':
                    assert digest(ROOT / record[role + '_path']) == record[role + '_sha256']
                else:
                    for file, suffix in [('ranks', '.parquet'), ('metrics', '.json')]:
                        assert digest(ROOT / record[role] / (file + suffix)) == record['hashes'][role][file]
            comparisons[(model, task)] = record

    gap = read('reports/v9_r8_mlp128_xgb128d10_gap_v1/summary.json')
    matched = read('reports/v9_r8_mlp128_vs_xgb128d10_completion_v1/summary.json')
    assert gap['complete_test_opened'] is False
    assert gap['neural'] == matched['scores']['candidate']
    assert gap['tree'] == matched['scores']['baseline']
    assert gap['paired_intervals'] == matched['paired_intervals']
    assert gap['neural_prediction_hash'] == matched['candidate_sha256']
    assert gap['tree_prediction_hash'] == matched['baseline_sha256']
    assert digest(ROOT / matched['candidate_path']) == matched['candidate_sha256']
    assert digest(ROOT / matched['baseline_path']) == matched['baseline_sha256']
    assert gap['fixed_denominator_partitions_reconstruct_both_main_scores'] is True
    parts = gap['fixed_denominator_partitions']
    for stratum, key in [('positive', 'positive_delta_contribution'), ('explicit_zero', 'zero_delta_contribution')]:
        part = next(x for x in parts if x['stratification'] == 'label_stratum' and x['stratum'] == stratum)
        assert abs(part['mae_gap_contribution'] - matched['primary_change_decomposition'][key]) < 1e-12

    def score(model, role):
        return comparisons[(model, 'completion')]['scores'][role]['nutrition']['scaled_log_mae']

    mlp_gain = score('MLP', 'baseline') - score('MLP', 'candidate')
    tree_regression = score('XGB d10', 'candidate') - score('XGB d10', 'baseline')
    gap32 = score('MLP', 'baseline') - score('XGB d10', 'baseline')
    gap128 = score('MLP', 'candidate') - score('XGB d10', 'candidate')
    assert abs((gap32 - gap128) - (mlp_gain + tree_regression)) < 1e-12
    assert abs(gap128 - gap['primary_gap']) < 1e-12
    assert np.isfinite([mlp_gain, tree_regression, gap32, gap128]).all()
    payload = {
        'status': 'fixed_depth_dimension_stage_complete_full_round_pending',
        'gap32': gap32, 'gap128': gap128, 'gap_reduction': gap32 - gap128,
        'mlp_absolute_improvement': mlp_gain, 'xgb_absolute_regression': tree_regression,
        'tree_regression_fraction_of_arithmetic_gap_reduction': tree_regression / (gap32 - gap128),
        'all_scores_intervals_hashes_and_positive_zero_decomposition_verified': True,
        'scope': 'Descriptive arithmetic identity, not causal mediation or replicated model acceptance. Paired intervals condition on fitted single-seed models; no interval for gap-reduction interaction was calculated.',
        'complete_test_opened': False, 'scientific_confirmation': False,
        'model_improvement_accepted': False, 'final_tree_selected': False,
        'inputs_sha256': hashes, 'script_sha256': digest(Path(__file__)),
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / 'summary.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    colors = {'MLP': '#287aab', 'XGB d10': '#bf573f'}
    for ax, task, title in zip(axes[:2], ['completion', 'name_only'], ['Completion', 'Food-name-only']):
        for model in colors:
            rec = comparisons[(model, task)]
            values = [rec['scores'][role]['nutrition']['scaled_log_mae'] for role in ['baseline', 'candidate']]
            ax.plot([32, 128], values, 'o-', label=model, color=colors[model])
            for x, value in zip([32, 128], values):
                ax.annotate(f'{value:.4f}', (x, value), xytext=(0, 8), textcoords='offset points', ha='center', fontsize=9)
        ax.set(title=title, xlabel='Active name dimensions', ylabel='142-axis scaled-log MAE (lower is better)', xticks=[32, 128], xlim=(10, 150))
        ax.margins(y=.3)
        ax.grid(axis='y', alpha=.2)
        ax.legend(loc='best', fontsize=9)
    ax = axes[2]
    rows = [('MLP', 'completion'), ('XGB d10', 'completion'), ('MLP', 'name_only'), ('XGB d10', 'name_only')]
    for y, key in enumerate(rows):
        interval = comparisons[key]['paired_intervals']['scaled_log_mae']
        center = interval['relative_improvement'] * 100
        lo, hi = np.array(interval['relative_improvement_95_interval']) * 100
        ax.errorbar(center, y, xerr=[[center - lo], [hi - center]], fmt='o', color=colors[key[0]], capsize=4)
    ax.axvline(0, color='gray', linewidth=.8)
    ax.set(title='128 vs 32: paired food-group intervals', xlabel='Relative improvement (%)', yticks=range(4), yticklabels=['MLP / completion', 'XGB / completion', 'MLP / name-only', 'XGB / name-only'])
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    fig.suptitle('R8 fixed-depth stage: a smaller model gap is not the same as neural improvement', fontsize=12)
    fig.text(.5, .015, 'Single seed; validation only. Error bars: 95% paired group intervals, conditional on fitted models. Full tree tuning pending.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .93))
    for suffix in ['png', 'svg']:
        fig.savefig(args.output_dir / ('dimension_comparison.' + suffix), dpi=180)
    plt.close(fig)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
