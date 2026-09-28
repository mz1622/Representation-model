"""Freeze existing data and completed trees for Transformer-only R9 experiments."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_completion_input import execution_contract, completion_view
from foodcomp.research_r0 import ResearchData, VERSION, digest, score_predictions, write_json

TREES = {'rf400leaf1half_name32': 'rf32leaf1half', 'rf400leaf1half_name128': 'rf128leaf1half',
    'xgb800d10_name32': 'xgb32', 'xgb800d10_name128': 'xgb128d10',
    'xgb800d6_name128': 'xgb128d6', 'xgb800d14_name128': 'xgb128d14'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    contract, path = execution_contract(ROOT)
    data = ResearchData(ROOT / 'data/processed' / VERSION)
    _, cache, panel = completion_view(data, ROOT, 32)
    inputs = {}
    def record(p):
        inputs[p.relative_to(ROOT).as_posix()] = digest(p)
    record(data.root / 'manifest.json')
    for name, expected in data.manifest['artifact_hashes'].items():
        assert digest(data.root / name) == expected
        record(data.root / name)
    for parent in [cache, panel]:
        for p in parent.iterdir():
            if p.is_file():
                record(p)
    record(path)
    baselines = {}
    for name, tag in TREES.items():
        run = ROOT / 'output/v9_r8' / name
        manifest = json.loads((run / 'run_manifest.json').read_text())
        audit_path = ROOT / f'reports/v9_r8_{tag}_completed_audit_v1/verification.json'
        audit = json.loads(audit_path.read_text())
        assert manifest['status'] == audit['status'] == 'complete'
        assert not manifest['complete_test_opened'] and not audit['complete_test_opened']
        assert audit['records'][0]['manifest_sha256'] == digest(run / 'run_manifest.json')
        saved = json.loads((run / 'metrics.json').read_text())
        for task in ['completion', 'name_only']:
            p = run / f'{task}_predictions.parquet'
            actual, _, _ = score_predictions(data, pd.read_parquet(p))
            assert actual == saved[task], (name, task)
            record(p)
        for p in [run / 'run_manifest.json', run / 'metrics.json', audit_path,
                  run / 'retrieval/metrics.json', run / 'retrieval/ranks.parquet', run / 'retrieval/candidate_names.json']:
            record(p)
        baselines[name] = {'kind': manifest['kind'], 'active_name_dimensions': manifest['active_name_dimensions'],
            'directory': run.relative_to(ROOT).as_posix(), 'seed': manifest['seed'],
            'configuration': manifest['configuration'], 'metrics': saved,
            'manifest_sha256': digest(run / 'run_manifest.json'), 'audit_sha256': digest(audit_path)}
    strongest_rf = min((k for k, v in baselines.items() if v['kind'] == 'rf'),
                       key=lambda k: baselines[k]['metrics']['completion']['nutrition']['scaled_log_mae'])
    assert strongest_rf == 'rf400leaf1half_name32'
    config_path = ROOT / 'experiments/foodnutrigpt_v9_research/r9/config.json'
    config = json.loads(config_path.read_text())
    assert digest(data.root / 'manifest.json') == config['data_sha256']
    args.output_dir.mkdir(parents=True)
    result = {'status': 'frozen', 'version': 'V9-R9', 'data_hash': digest(data.root / 'manifest.json'),
        'name_cache_hash': digest(cache / 'manifest.json'), 'panel_hash': digest(panel / 'manifest.json'),
        'name_cache': cache.relative_to(ROOT).as_posix(), 'panel_root': panel.relative_to(ROOT).as_posix(),
        'input_hashes': inputs, 'baselines': baselines, 'primary_rf': strongest_rf,
        'matched_xgb': 'xgb800d10_name32', 'config_sha256': digest(config_path),
        'script_sha256': digest(Path(__file__)), 'complete_test_opened': False,
        'fitting_performed': False, 'data_written': False,
        'scope': 'Read-only freeze and full rescore of existing completed baseline outputs; no new tree fit, no data revision. Partial RF run excluded.'}
    write_json(args.output_dir / 'manifest.json', result)
    print(json.dumps({'status': 'frozen', 'completed_trees': len(baselines), 'primary_rf': strongest_rf,
                      'input_files': len(inputs), 'data_written': False}))


if __name__ == '__main__':
    main()
