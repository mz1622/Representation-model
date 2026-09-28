"""Summarize audited neural repetitions without selecting a seed or claiming tree superiority."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import digest, write_json

METRICS = ['scaled_log_mae', 'log_mae', 'raw_mae', 'positive_scaled_log_mae', 'zero_scaled_log_mae']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def stats(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('Three finite seed values required.')
    return {'mean': float(values.mean()), 'sample_sd': float(values.std(ddof=1)),
        'min': float(values.min()), 'max': float(values.max())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    audit_path = args.audit_dir / 'verification.json'
    audit = read(audit_path)
    assert audit['status'] == 'complete' and audit['all_three_fixed_recipe_seeds_audited']
    assert not audit['complete_test_opened']
    assert [r['seed'] for r in audit['records']] == [20260922, 20260923, 20260924]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    nutrition_rows, retrieval_rows, histories, identities = [], [], [], []
    for record in audit['records']:
        run = Path(record['run'])
        retrieval = Path(record['retrieval_directory'])
        for path, expected in [(run / 'run_manifest.json', record['manifest_sha256']),
                (run / 'best_model.pt', record['checkpoint_sha256']),
                (run / 'metrics.json', record['metrics_sha256']),
                (retrieval / 'metrics.json', record['retrieval_metrics_sha256']),
                (retrieval / 'ranks.parquet', record['retrieval_ranks_sha256'])]:
            assert digest(path) == expected, str(path)
        assert read(run / 'metrics.json') == record['metrics']
        for task in ['completion', 'name_only']:
            assert digest(run / f'{task}_predictions.parquet') == record['prediction_sha256'][task]
            score = record['metrics'][task]
            nutrition_rows.append({'seed': record['seed'], 'task': task, **score['nutrition'],
                'metabolome45_scaled_log_mae': score['food_metabolome']['scaled_log_mae'],
                'all187_scaled_log_mae': score['all']['scaled_log_mae']})
        for scores in read(retrieval / 'metrics.json')['metrics']:
            retrieval_rows.append({'seed': record['seed'], **scores})
        history = pd.read_csv(run / 'history.csv', float_precision='round_trip')
        assert len(history) == 60 and np.isfinite(history.select_dtypes('number')).all().all()
        assert int(history.loc[history.validation_primary.idxmin(), 'epoch']) == record['best_epoch']
        histories.append((record['seed'], record['best_epoch'], history))
        identities.append({'seed': record['seed'], 'run': str(run), 'checkpoint_sha256': record['checkpoint_sha256'],
            'manifest_sha256': record['manifest_sha256'], 'history_sha256': digest(run / 'history.csv'),
            'best_epoch': record['best_epoch'], 'training_seconds': record['training_seconds']})
    nutrition = pd.DataFrame(nutrition_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    aggregate = {}
    for task in ['completion', 'name_only']:
        group = nutrition[nutrition.task.eq(task)]
        aggregate[task] = {metric: stats(group[metric]) for metric in METRICS +
            ['metabolome45_scaled_log_mae', 'all187_scaled_log_mae']}
    retrieval_summary = {}
    for fraction, group in retrieval.groupby('visible_fraction'):
        retrieval_summary[str(float(fraction))] = {metric: stats(group[metric]) for metric in
            ['mrr', 'recall_at_1', 'recall_at_5', 'recall_at_10']}
    result = {'status': 'complete_neural_repetition_stage', 'audit_sha256': digest(audit_path),
        'script_sha256': digest(Path(__file__)), 'records': identities,
        'nutrition_seed_rows': nutrition_rows, 'nutrition_summary': aggregate,
        'retrieval_seed_rows': retrieval_rows, 'retrieval_summary': retrieval_summary,
        'complete_test_opened': False, 'final_tree_comparison_complete': False,
        'scope': 'Three independent fixed-configuration models, sample SD across three seeds; no best-seed selection or prediction ensemble. Not independent test, not confidence intervals for the population of seeds.'}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'summary.json', result)
    nutrition.to_csv(args.output_dir / 'nutrition_by_seed.csv', index=False)
    retrieval.to_csv(args.output_dir / 'retrieval_by_seed.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout='constrained')
    for seed, best, history in histories:
        line, = axes[0].plot(history.epoch, history.train_loss, label=str(seed), linewidth=1.4)
        axes[1].plot(history.epoch, history.validation_primary, color=line.get_color(), label=str(seed), linewidth=1.4)
        value = float(history.loc[history.epoch.eq(best), 'validation_primary'].iloc[0])
        axes[1].scatter([best], [value], color=line.get_color(), marker='*', s=100, zorder=3)
    axes[0].set_title('Training objective: 187 supervised axes')
    axes[0].set_ylabel('Weighted scaled-log MAE')
    axes[1].set_title('Validation primary: 142 nutrition axes')
    axes[1].set_ylabel('Macro scaled-log MAE')
    for ax in axes:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        ax.legend(title='Seed', frameon=False)
    fig.suptitle('Fixed MLP512 / exact name PCA128: three independent seeds')
    fig.savefig(args.output_dir / 'learning_curves.png', dpi=170)
    fig.savefig(args.output_dir / 'learning_curves.svg')
    plt.close(fig)
    print(json.dumps({'nutrition': aggregate, 'retrieval': retrieval_summary}))


if __name__ == '__main__':
    main()

