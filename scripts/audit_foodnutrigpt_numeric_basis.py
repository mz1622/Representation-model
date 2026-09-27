"""Check a standalone numeric basis on training-only rows; no learned model or scores."""
import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_numeric_basis import FrozenNumericBasis
from foodcomp.research_r0 import VERSION, digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    start = time.monotonic()
    torch.set_num_threads(2)
    prior_path = ROOT / 'reports/v9_numeric_resolution_v1/summary.json'
    prior = json.loads(prior_path.read_text(encoding='utf-8'))
    assert prior['status'] == 'complete_training_only_feasibility_no_model'
    assert prior['complete_test_opened'] is False and prior['models_trained'] == 0
    knots_path = Path(prior['knots_directory']) / 'candidate_knots.json'
    assert knots_path.resolve().is_relative_to((ROOT / 'data/local/research_diagnostics').resolve())
    assert digest(knots_path) == prior['knots_sha256']
    knots = json.loads(knots_path.read_text(encoding='utf-8'))
    data_root = ROOT / 'data/processed' / VERSION
    for name, expected in prior['artifact_hashes'].items():
        assert digest(data_root / name) == expected
    profiles = pd.read_csv(data_root / 'profiles.csv.gz', usecols=['profile_index', 'partition'])
    selected = profiles.loc[profiles.partition.eq('train'), 'profile_index'].to_numpy()[:768]
    assert len(selected) == 768 and len(np.unique(selected)) == 768
    axes = pd.read_csv(data_root / 'axes.csv')
    cells = pd.read_parquet(data_root / 'canonical_cells.parquet',
        columns=['profile_index', 'axis_index', 'value', 'partition', 'quarantined'],
        filters=[('partition', '==', 'train'), ('quarantined', '==', False), ('profile_index', 'in', selected.tolist())])
    assert cells.partition.eq('train').all() and not cells.quarantined.any()
    rowmap = {int(row): i for i, row in enumerate(selected)}
    raw = np.full((len(selected), len(axes)), np.nan)
    raw[cells.profile_index.map(rowmap).to_numpy(), cells.axis_index.to_numpy()] = cells.value.to_numpy()
    observed = np.isfinite(raw)
    values = np.where(observed, np.log1p(raw / axes.research_scale.to_numpy()), 0).astype(np.float32)
    visible = observed & ~axes.mask_family.isin(['fatty_acid', 'mineral']).to_numpy()[None, :]
    records = []
    devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
    for policy in ['all_observed', 'positive_only']:
        for bins in [8, 16, 32]:
            per_axis = [knots[f'{policy}/axis{a}/bins{bins}'] for a in range(len(axes))]
            for device in devices:
                rng = torch.get_rng_state().clone()
                basis = FrozenNumericBasis(per_axis, bins).to(device)
                assert torch.equal(rng, torch.get_rng_state()) and not list(basis.parameters())
                x = torch.as_tensor(values, device=device)
                mask = torch.as_tensor(visible, device=device)
                expected = basis(x, mask)
                assert torch.equal(expected[:, :len(axes)], torch.where(mask, x, 0.))
                assert torch.equal(expected[:, len(axes):2*len(axes)], mask.float())
                changed = x.clone(); changed[~mask] = float('nan')
                assert torch.equal(expected, basis(changed, mask))
                order = torch.arange(len(x)-1, -1, -1, device=device)
                assert torch.equal(expected[order], basis(x[order], mask[order]))
                assert torch.equal(expected, torch.cat([basis(x[:257], mask[:257]), basis(x[257:], mask[257:])]))
                assert torch.equal(basis(x, torch.zeros_like(mask)), torch.zeros_like(expected))
                zero = basis(torch.zeros_like(x), torch.ones_like(mask))
                assert torch.equal(zero[:, len(axes):2*len(axes)], torch.ones_like(x))
                assert torch.count_nonzero(zero[:, 2*len(axes):]) == 0
                memory = io.BytesIO(); torch.save(basis.state_dict(), memory); memory.seek(0)
                restored = FrozenNumericBasis([[] for _ in range(len(axes))], bins).to(device)
                restored.load_state_dict(torch.load(memory, map_location=device, weights_only=True))
                assert torch.equal(expected, restored(x, mask))
                records.append({'policy': policy, 'requested_intervals': bins, 'device': device,
                    'rows': len(x), 'axes': len(axes), 'numeric_output_features': basis.output_features,
                    'effective_intervals': int(basis.active.sum()), 'constant_or_empty_axes': int((~basis.active.any(1)).sum()),
                    'raw_and_visibility_preserved_exact': True, 'hidden_payload_independent_exact': True,
                    'row_order_and_subbatch_independent_exact': True, 'state_reload_exact': True,
                    'name_only_zero_and_explicit_zero_distinction': True, 'trainable_parameters': 0})
    result = {'status': 'standalone_basis_functional_complete_not_model_integrated', 'records': records,
        'resolution_summary_sha256': digest(prior_path), 'knots_sha256': digest(knots_path),
        'data_sha256': prior['data_sha256'], 'selected_training_rows_sha256': __import__('hashlib').sha256(selected.astype('<i8').tobytes()).hexdigest(),
        'script_sha256': digest(Path(__file__)), 'basis_code_sha256': digest(ROOT / 'src/foodcomp/research_numeric_basis.py'),
        'elapsed_seconds': time.monotonic()-start, 'models_trained': 0, 'complete_test_opened': False,
        'validation_numeric_rows_loaded': False, 'model_improvement_accepted': False,
        'scope': 'Standalone deterministic input block on the first768 training profiles across252 axes. Within-backend exactness only. Not integrated into training, loss, prediction/query APIs or candidate checkpoints; no performance claim and no R9 registration.'}
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / 'verification.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
