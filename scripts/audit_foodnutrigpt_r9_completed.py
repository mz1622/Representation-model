"""Independent full trajectory and three-task replay for a completed R9 Transformer."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_transformer_r9 import frozen_inputs, TransformerNutritionModel
from foodcomp.research_r0 import digest, score_predictions, write_json
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_neural import evaluate
from evaluate_foodnutrigpt_name_neighbors import evaluate_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frozen, freeze_path = frozen_inputs(ROOT)
    manifest = json.loads((args.run / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete' and not manifest['complete_test_opened']
    assert not manifest['baseline_refit'] and not manifest['data_modified']
    assert manifest['freeze_sha256'] == digest(freeze_path)
    for path, expected in manifest['code_hashes'].items():
        assert digest(ROOT / path) == expected, path
    checkpoint = args.run / 'best_model.pt'
    assert digest(checkpoint) == manifest['checkpoint_sha256']
    history = pd.read_csv(args.run / 'history.csv', float_precision='round_trip')
    spec = manifest['spec']
    assert np.array_equal(history.epoch, np.arange(1, spec['epochs'] + 1))
    assert np.isfinite(history.select_dtypes('number')).all().all()
    assert history.training_tasks.eq(manifest['training_tasks']).all()
    assert history.observed_target_cells.eq(manifest['observed_target_cells']).all()
    for row in history.itertuples():
        order = np.random.default_rng(spec['seed'] + row.epoch).permutation(manifest['training_tasks'])
        assert fingerprint_array(order) == row.training_order_sha256
        expected_lr = spec['learning_rate'] * (.01 + .99 * (1 + np.cos(np.pi * (row.epoch - 1) / spec['schedule_epochs'])) / 2)
        np.testing.assert_allclose(row.learning_rate, expected_lr, rtol=1e-12, atol=1e-15)
    assert int(history.loc[history.validation_primary.idxmin(), 'epoch']) == manifest['best_epoch']
    assert float(history.validation_primary.min()) == manifest['best_primary']
    torch.set_num_threads(4)
    model = TransformerNutritionModel(checkpoint, ROOT)
    data = model.data
    scores = json.loads((args.run / 'metrics.json').read_text())
    hashes = {}
    for task in ['completion', 'name_only']:
        path = args.run / f'{task}_predictions.parquet'
        saved = pd.read_parquet(path)
        actual = evaluate(model.model, data, model._cached_text, model.device, task)
        pd.testing.assert_frame_equal(saved, actual, check_exact=True)
        recomputed, _, _ = score_predictions(data, actual)
        assert recomputed == scores[task]
        hashes[task] = digest(path)
    ret = args.run / 'retrieval'
    meta = json.loads((ret / 'metrics.json').read_text())
    names = json.loads((ret / 'candidate_names.json').read_text())
    assert digest(ret / 'candidate_names.json') == meta['candidate_sha256']
    assert names == sorted(set(data.profiles.original_name.astype(str)))
    axes = np.flatnonzero(data.axes.loss_group.eq('nutrition') & data.axes.loss_eligible)
    predicted = model.candidate_profiles(names, target_axes=axes)
    scaled = np.log1p(predicted / data.scale[axes]).astype(np.float32)
    np.testing.assert_array_equal(scaled, np.load(args.run / 'candidate_scaled.npy'))
    ranks, metrics = evaluate_candidates(data, names, scaled, axes, str(model.device))
    pd.testing.assert_frame_equal(ranks, pd.read_parquet(ret / 'ranks.parquet'), check_exact=True)
    assert metrics == meta['metrics'] and len(ranks) == 19089
    frozen_inputs(ROOT)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'verification.json', {'status': 'complete', 'run': str(args.run),
        'manifest_sha256': digest(args.run / 'run_manifest.json'), 'checkpoint_sha256': digest(checkpoint),
        'freeze_sha256': digest(freeze_path), 'prediction_sha256': hashes,
        'all_epoch_orders_exposures_and_schedule_verified': True,
        'both323809_predictions_public_loader_replayed_exact': True,
        'all_candidate_vectors_and19089_ranks_replayed_exact': True,
        'metrics': scores, 'retrieval_metrics': metrics, 'complete_test_opened': False,
        'script_sha256': digest(Path(__file__))})
    print(json.dumps({'status': 'complete', 'run': str(args.run)}))


if __name__ == '__main__':
    main()
