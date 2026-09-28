"""Describe trained source-offset stability using audited MAE checkpoints on CPU only."""
import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import digest, write_json

RUNS = {20260922: ('output/v9_r9/tf192_mae_lr3e4_60', 'reports/v9_r9_tf192_mae_lr3e4_60_audit_v1'),
        20260923: ('output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260923', 'reports/v9_r9_confirmation_seed20260923_audit_v1'),
        20260924: ('output/v9_r9_confirmation/tf192_mae_lr3e4_60_seed20260924', 'reports/v9_r9_confirmation_seed20260924_audit_v1')}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rms(array):
    return float(np.sqrt(np.square(array).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    started = time.monotonic()
    torch.set_num_threads(1)
    freeze_path = ROOT / 'reports/v9_r9_freeze_v1/manifest.json'
    freeze = read(freeze_path)
    axes_path = ROOT / 'data/processed/foodnutrigpt_v9_r0_v1/axes.csv'
    if digest(axes_path) != freeze['input_hashes'][axes_path.relative_to(ROOT).as_posix()]:
        raise ValueError('Frozen axis metadata changed')
    axes = pd.read_csv(axes_path, usecols=['axis_index', 'canonical_name', 'mask_family',
                                        'loss_group', 'loss_eligible', 'train_profile_count'])
    axes = axes.sort_values('axis_index').reset_index(drop=True)
    if not np.array_equal(axes.axis_index, np.arange(252)) or axes.loss_eligible.sum() != 187:
        raise ValueError('Unexpected axis grid')
    hashes = {freeze_path.relative_to(ROOT).as_posix(): digest(freeze_path),
              axes_path.relative_to(ROOT).as_posix(): digest(axes_path)}
    per_axis, subsets, matrices, records = [], [], {}, []
    expected_indices = None
    masks = {'nutrition': (axes.loss_eligible & axes.loss_group.eq('nutrition')).to_numpy(),
             'food_metabolome': (axes.loss_eligible & axes.loss_group.eq('food_metabolome')).to_numpy(),
             'all_supervised': axes.loss_eligible.to_numpy()}
    for seed, (run, audit_dir) in RUNS.items():
        run, audit_dir = ROOT / run, ROOT / audit_dir
        manifest_path, audit_path = run / 'run_manifest.json', audit_dir / 'verification.json'
        manifest, audit = read(manifest_path), read(audit_path)
        checkpoint = run / 'best_model.pt'
        if (manifest['status'] != 'complete' or audit['status'] != 'complete'
                or manifest['seed'] != seed or manifest['spec']['objective'] != 'mae'
                or audit['manifest_sha256'] != digest(manifest_path)
                or audit['checkpoint_sha256'] != manifest['checkpoint_sha256']
                or digest(checkpoint) != manifest['checkpoint_sha256']
                or manifest['freeze_sha256'] != digest(freeze_path)
                or any(manifest[k] for k in ['complete_test_opened', 'data_modified', 'baseline_refit'])):
            raise ValueError('A complete matching audited MAE checkpoint is required')
        for path in [manifest_path, audit_path, checkpoint]:
            hashes[path.relative_to(ROOT).as_posix()] = digest(path)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = saved['model_state']
        indices = state['train_source_indices'].cpu().numpy()
        if len(indices) != 24 or len(set(indices)) != 24:
            raise ValueError('Expected24 unique training source indices')
        if expected_indices is None:
            expected_indices = indices.copy()
        else:
            np.testing.assert_array_equal(indices, expected_indices)
        raw = state['source_amount_residual.weight'].cpu()
        if raw.shape[1] != 252 or not torch.isfinite(raw).all():
            raise ValueError('Invalid residual parameter table')
        learned = raw[state['train_source_indices']]
        # Reconstruct the declared centering in float32 on CPU. No model forward,
        # source-calibrated validation prediction or raw composition table is used.
        centred = learned - learned.mean(dim=0, keepdim=True)
        array = centred.double().numpy()
        np.testing.assert_allclose(array.mean(axis=0), 0., atol=1e-7, rtol=0)
        if np.any(array[:, ~axes.loss_eligible.to_numpy()] != 0):
            raise ValueError('Unexpected learned context-only-axis source offset')
        matrices[seed] = array
        frame = axes[axes.loss_eligible].copy()
        frame['seed'] = seed
        frame['centred_source_rms'] = np.sqrt(np.square(array[:, axes.loss_eligible]).mean(axis=0))
        frame['centred_source_mean_absolute'] = np.abs(array[:, axes.loss_eligible]).mean(axis=0)
        frame['centred_source_max_absolute'] = np.abs(array[:, axes.loss_eligible]).max(axis=0)
        per_axis.append(frame)
        for subset, mask in masks.items():
            values = array[:, mask]
            subsets.append({'seed': seed, 'subset': subset, 'axes': int(mask.sum()),
                            'source_count': len(indices), 'rms': rms(values),
                            'mean_absolute': float(np.abs(values).mean()),
                            'max_absolute': float(np.abs(values).max())})
        penalty = float(learned.square().mean())
        records.append({'seed': seed, 'selected_epoch': manifest['best_epoch'],
                        'checkpoint_sha256': manifest['checkpoint_sha256'],
                        'raw_training_source_table_mean_square_252axes': penalty,
                        'regularization_term': penalty * manifest['spec']['source_residual_l2'],
                        'max_abs_centering_roundoff': float(np.abs(array.mean(axis=0)).max())})
        del saved, state, raw, learned, centred
    pair_rows = []
    for a, b in itertools.combinations(RUNS, 2):
        for subset, mask in masks.items():
            x, y = matrices[a][:, mask].ravel(), matrices[b][:, mask].ravel()
            if min(np.std(x), np.std(y)) == 0:
                raise ValueError('Global offset correlation undefined; report explicitly before extending analysis')
            pair_rows.append({'seed_a': a, 'seed_b': b, 'subset': subset,
                              'pearson_r': float(np.corrcoef(x, y)[0, 1]),
                              'difference_rms': rms(x-y),
                              'difference_rms_over_average_offset_rms': rms(x-y) / ((rms(x)+rms(y))/2)})
    for name, expected in hashes.items():
        if digest(ROOT / name) != expected:
            raise ValueError('Input identity changed during diagnosis: ' + name)
    args.output_dir.mkdir(parents=True)
    pd.concat(per_axis, ignore_index=True).to_csv(args.output_dir / 'per_axis_parameter_summaries.csv', index=False)
    pd.DataFrame(subsets).to_csv(args.output_dir / 'subset_parameter_summaries.csv', index=False)
    pd.DataFrame(pair_rows).to_csv(args.output_dir / 'seed_pair_stability.csv', index=False)
    summary = {'status': 'complete', 'records': records, 'subsets': subsets, 'seed_pairs': pair_rows,
               'input_hashes': hashes, 'script_sha256': digest(Path(__file__)),
               'elapsed_seconds': time.monotonic()-started, 'device': 'cpu', 'torch_threads': 1,
               'model_forward_performed': False, 'training_performed': False,
               'raw_composition_values_loaded': False, 'validation_labels_loaded': False,
               'data_modified': False, 'baseline_refit': False, 'complete_test_opened': False,
               'current_mse_run_read_or_modified': False, 'goal_achieved': False,
               'scope': 'Learned-parameter descriptors only, equally weighting24 training sources. The models were previously selected on validation. High parameter correlation does not show calibration improves source-free predictions; low correlation does not prove the cause of prediction variance. Offsets are in scaled-log space and are not concentrations. Different source/axis supervision coverage is not controlled. No causal calibration on/off contrast has been performed.'}
    write_json(args.output_dir / 'summary.json', summary)
    print(json.dumps({'status': 'complete', 'records': records, 'subsets': subsets, 'seed_pairs': pair_rows}, indent=2))


if __name__ == '__main__':
    main()
