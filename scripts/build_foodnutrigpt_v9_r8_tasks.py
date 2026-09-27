"""Build new R8 input receipts while preserving every R1 target, weight and task."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_r1 import PANEL_VERSION, build_panel
from foodcomp.research_name_projection import load_variant
from foodcomp.research_completion_input import completion_view


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    data = ResearchData(ROOT / 'data/processed' / VERSION)
    registry = read(ROOT / 'reports/v9_r7_name_cache_v1/verification.json')
    parent = ROOT / 'data/processed' / PANEL_VERSION
    original = read(parent / 'manifest.json')
    assert digest(parent / 'tasks.npz') == original['tasks_sha256']
    records = []
    for dims in [32, 128]:
        cache = ROOT / registry['paths'][str(dims)]
        assert digest(cache / 'manifest.json') == registry['manifest_sha256'][str(dims)]
        text, _, _ = load_variant(cache, data_hash=digest(data.root / 'manifest.json'), active_components=dims)
        output = ROOT / f'data/processed/foodnutrigpt_v9_r8_tasks_name{dims}_v2'
        build_panel(data, text, cache, output)
        manifest = read(output / 'manifest.json')
        # These are verified locally generated archives. The historical family_names
        # array has object dtype; numeric training arrays still load without pickle.
        with np.load(output / 'tasks.npz', allow_pickle=True) as current, np.load(parent / 'tasks.npz', allow_pickle=True) as old:
            assert set(current.files) == set(old.files)
            for key in current.files:
                np.testing.assert_array_equal(current[key], old[key])
        for new, old in zip(manifest['receipts'], original['receipts']):
            for key in ['family', 'tasks', 'targets', 'raw_labels_sha256', 'source_weights_sha256',
                    'target_mask_sha256', 'training_rows_sha256']:
                assert new[key] == old[key], (dims, key)
        for key in ['data_hash', 'view', 'tasks', 'observed_target_cells', 'families', 'validation_jobs_sha256']:
            assert manifest[key] == original[key], key
        manifest.update(version='foodnutrigpt_v9_r8_tasks_v1', input_slots=632,
            name_slots=128, active_name_dimensions=dims,
            parent_targets_rows_weights_unchanged=True, parent_manifest_sha256=digest(parent / 'manifest.json'),
            builder_sha256=digest(Path(__file__)),
            baseline_reuse='Old32 baselines are historical only. R8 trees must refit on these exact632 features.')
        write_json(output / 'manifest.json', manifest)
        loaded, _, loaded_root = completion_view(data, ROOT, dims)
        np.testing.assert_array_equal(loaded, text); assert loaded_root == output
        records.append({'dimensions': dims, 'directory': str(output.relative_to(ROOT)),
            'manifest_sha256': digest(output / 'manifest.json'), 'tasks': manifest['tasks'],
            'observed_target_cells': manifest['observed_target_cells']})
    destination = ROOT / 'reports/v9_r8_input_panels_v2'
    destination.mkdir(exist_ok=False)
    write_json(destination / 'verification.json', {'status': 'complete', 'records': records,
        'all_families_tree_neural_632features_equal': True,
        'parent_all_tasks_targets_weights_exact': True, 'complete_test_opened': False,
        'prior_failure': 'v1 comparison refused the historical object-dtype family_names array; failed_builder.py and failure.log retained under name32_v1. No training ran.'})
    print(records)


if __name__ == '__main__':
    main()
