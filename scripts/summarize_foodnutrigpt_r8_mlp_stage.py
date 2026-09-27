"""Summarize the completed R8 neural control pair while full tree tuning is pending."""
import argparse
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


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists(): raise FileExistsError(args.output_dir)
    audit_path = ROOT / 'reports/v9_r8_mlp_completed_audit_v2/verification.json'
    audit = read(audit_path); assert audit['status'] == 'complete'
    records = []
    for dims in [32, 128]:
        run = ROOT / f'output/v9_r8/mlp512_name{dims}'
        m = read(run / 'run_manifest.json'); assert m['status'] == 'complete'
        records.append({'active_name_dimensions': dims, 'manifest': m,
            'manifest_sha256': digest(run / 'run_manifest.json'), 'metrics': read(run / 'metrics.json'),
            'retrieval': read(ROOT / f'output/v9_r8/retrieval_mlp512_name{dims}/metrics.json'),
            'same_name_view_knn_nutrition': read(ROOT / f'output/v9_r7/exact_name_knn{dims}/nutrition_metrics.json'),
            'same_name_view_knn_retrieval': read(ROOT / f'output/v9_r7/exact_name_knn{dims}/metrics.json'),
            'name_knn_comparison': read(ROOT / f'reports/v9_r8_mlp{dims}_name_vs_knn_v1/summary.json'),
            'retrieval_knn_comparison': read(ROOT / f'reports/v9_r8_mlp{dims}_retrieval_vs_knn_v1/summary.json')})
    pairs = {task: read(ROOT / f'reports/v9_r8_mlp128_vs32_{task}_v1/summary.json')
        for task in ['completion', 'name_only', 'retrieval']}
    counts = {}
    for task in ['completion', 'name_only']:
        path = ROOT / f'reports/v9_r8_mlp128_vs32_{task}_v1'
        axis = pd.read_csv(path / 'axis_paired_intervals.csv'); axis = axis[axis.loss_group.eq('nutrition')]
        source = pd.read_csv(path / 'source_metrics.csv').pivot(index='source', columns='role', values='scaled_log_mae')
        counts[task] = {'axes_point_better': int((axis.candidate_minus_baseline < 0).sum()),
            'axes_interval_better': int((axis.difference_95_high < 0).sum()),
            'axes_interval_worse': int((axis.difference_95_low > 0).sum()),
            'sources_point_better': int((source.candidate < source.baseline).sum()), 'source_count': len(source)}
    fit = read(ROOT / 'reports/v9_r8_mlp_fit_v1/summary.json')
    visibility = read(ROOT / 'reports/v9_r8_mlp_visibility_v1/summary.json')
    assert visibility['status'] == 'complete'
    args.output_dir.mkdir(parents=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for dim in [32, 128]:
        h = pd.read_csv(ROOT / f'output/v9_r8/mlp512_name{dim}/history.csv')
        axs[0, 0].plot(h.epoch, h.validation_primary, label=f'{dim} active name dimensions')
    axs[0, 0].set(title='Fixed completion validation', xlabel='Epoch', ylabel='142-axis scaled-log MAE')
    axs[0, 0].legend(fontsize=8)
    x = np.arange(2)
    train = [next(m['training']['nutrition']['scaled_log_mae'] for m in fit['models']
        if m['run'] == f"mlp512_name{r['active_name_dimensions']}") for r in records]
    axs[0, 1].bar(x-.18, train, .36, label='Full original training context')
    axs[0, 1].bar(x+.18, [r['metrics']['completion']['nutrition']['scaled_log_mae'] for r in records], .36, label='Validation')
    axs[0, 1].set_ylim(0, max(r['metrics']['completion']['nutrition']['scaled_log_mae'] for r in records) * 1.3)
    axs[0, 1].set_xticks(x, ['32 active', '128 active']); axs[0, 1].set(title='Same selected checkpoints', ylabel='142-axis scaled-log MAE'); axs[0, 1].legend(fontsize=8)
    axs[1, 0].bar(x-.18, [r['metrics']['name_only']['nutrition']['scaled_log_mae'] for r in records], .36, label='Completion-trained MLP')
    axs[1, 0].bar(x+.18, [r['same_name_view_knn_nutrition']['nutrition']['scaled_log_mae'] for r in records], .36, label='Same-name-view KNN')
    axs[1, 0].set_xticks(x, ['32 active', '128 active']); axs[1, 0].set(title='All numeric input hidden', ylabel='142-axis scaled-log MAE'); axs[1, 0].legend(fontsize=8)
    for j, fraction in enumerate([1., .3]):
        values = [next(m['recall_at_10'] for m in r['retrieval']['metrics'] if m['visible_fraction'] == fraction) for r in records]
        axs[1, 1].bar(x+(-.18 if j==0 else .18), values, .36, label='Full observed' if j==0 else '30% visible')
    axs[1, 1].set_xticks(x, ['32 active', '128 active']); axs[1, 1].set(title='Nutrition-to-name via predicted candidate profiles', ylabel='Recall@10'); axs[1, 1].legend(fontsize=8)
    for ax in axs.flat: ax.grid(axis='y', alpha=.15)
    fig.suptitle('R8 matched completion MLPs: single-seed stage\nFull same-input RF/XGBoost tuning remains a separate required comparison', fontsize=12)
    for ext in ['png', 'svg']: fig.savefig(args.output_dir / f'completion_name.{ext}', dpi=160)
    result = {'status': 'mlp_stage_complete_full_r8_not_complete', 'records': records,
        'comparisons': pairs, 'improvement_counts': counts, 'training_fit': fit, 'visibility': visibility,
        'audit_sha256': digest(audit_path), 'script_sha256': digest(Path(__file__)),
        'mlp_training_seconds': sum(r['manifest']['elapsed_seconds'] for r in records),
        'complete_test_opened': False, 'scientific_confirmation': False, 'model_improvement_accepted': False,
        'scope': 'Matched name-input intervention only; one neural seed. Must finish all preregistered same-input tree configs before judging primary milestone.'}
    write_json(args.output_dir / 'summary.json', result)
    print(result['status'], counts)
    for task in ['completion', 'name_only']:
        print(task, pairs[task]['paired_intervals'], pairs[task]['primary_change_decomposition'])
    for r in records:
        print(r['active_name_dimensions'], r['metrics']['completion']['nutrition'], r['metrics']['name_only']['nutrition'], r['retrieval']['metrics'])


if __name__ == '__main__': main()
