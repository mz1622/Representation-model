"""Registered R8 name views with independently audited family-task manifests."""
import json
from pathlib import Path
from .research_r0 import digest
from .research_name_projection import load_variant

EXECUTION_FILES = [
    'src/foodcomp/research_completion_input.py', 'src/foodcomp/research_tree_prediction.py',
    'scripts/train_foodnutrigpt_v9_r1.py', 'scripts/train_foodnutrigpt_v9_r8_completion.py',
    'scripts/run_foodnutrigpt_v9_r8_trees.py', 'src/foodcomp/research_r0.py',
    'src/foodcomp/research_r1.py', 'src/foodcomp/research_neural.py',
    'src/foodcomp/research_name_projection.py', 'src/foodcomp/research_inference.py',
    'src/foodcomp/research_auxiliary.py', 'src/foodcomp/research_text.py',
    'src/foodcomp/research_profile_retrieval.py', 'src/foodcomp/research_alignment.py',
    'scripts/train_global_foodnutrigpt_v8_single_stage.py',
    'scripts/train_global_foodnutrigpt_v9_source_calibrated.py',
    'scripts/evaluate_foodnutrigpt_name_neighbors.py',
    'scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py',
]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def completion_view(data, repo, dimensions):
    repo = Path(repo)
    if dimensions not in (32, 128):
        raise ValueError('Only the registered32/128 views are supported.')
    registry = read(repo / 'reports/v9_r7_name_cache_v1/verification.json')
    cache = repo / registry['paths'][str(dimensions)]
    if digest(cache / 'manifest.json') != registry['manifest_sha256'][str(dimensions)]:
        raise ValueError('Name registry fingerprint changed.')
    text, _, _ = load_variant(cache, data_hash=digest(data.root / 'manifest.json'),
        active_components=dimensions)
    panel = repo / f'data/processed/foodnutrigpt_v9_r8_tasks_name{dimensions}_v2'
    manifest = read(panel / 'manifest.json')
    if (manifest.get('version') != 'foodnutrigpt_v9_r8_tasks_v1'
            or manifest.get('name_cache_hash') != digest(cache / 'manifest.json')
            or manifest.get('data_hash') != digest(data.root / 'manifest.json')
            or manifest.get('input_slots') != 632
            or manifest.get('active_name_dimensions') != dimensions
            or not manifest.get('parent_targets_rows_weights_unchanged')
            or not manifest.get('exhaustive_tree_input_row_target_weight_contract_passed')):
        raise ValueError('Missing or mismatched R8 family-input audit.')
    if digest(panel / 'tasks.npz') != manifest['tasks_sha256']:
        raise ValueError('R8 task file changed.')
    return text, cache, panel


def require_r7_completion(repo):
    result = read(Path(repo) / 'experiments/foodnutrigpt_v9_research/r7/results_summary.json')
    if (result.get('status') != 'registered_six_candidate_block_complete_single_seed'
            or not all(result.get('dimension_screen_passed', {}).get(kind) for kind in ['mlp', 'knn'])
            or result.get('complete_test_opened', True)):
        raise ValueError('R7 preregistered completion/screening gate has not passed.')
    report = (Path(repo) / 'experiments/foodnutrigpt_v9_research/r7/README.md').read_text(encoding='utf-8')
    if not all(f'## {number}.' in report for number in range(1, 9)):
        raise ValueError('R7 final eight-section research report is required before R8 training.')


def execution_contract(repo):
    """Pin actual executable bytes; later documentation commits can be recorded separately."""
    repo = Path(repo)
    path = repo / 'reports/v9_r8_execution_contract_v1/manifest.json'
    contract = read(path)
    if contract.get('status') != 'frozen' or set(contract['code_hashes']) != set(EXECUTION_FILES):
        raise ValueError('Unfrozen R8 execution contract.')
    for name, expected in contract['code_hashes'].items():
        if digest(repo / name) != expected:
            raise ValueError(f'R8 executable source changed after registration: {name}')
    for name, expected in contract['input_hashes'].items():
        if digest(repo / name) != expected:
            raise ValueError(f'R8 input receipt changed: {name}')
    return contract, path
