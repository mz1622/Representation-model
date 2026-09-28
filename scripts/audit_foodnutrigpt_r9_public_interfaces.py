"""Check public R9 modality/query contracts on a completed model and train names.

This is an inference-contract check, not a transfer or retrieval-performance test.
No new names are encoded, no validation/test query values are used, and the
receipt contains no food names, measured composition values or embeddings.
"""
import argparse
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from foodcomp.research_r0 import digest, write_json
from foodcomp.research_transformer_r9 import TransformerNutritionModel, frozen_inputs


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rejected(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return True
    raise AssertionError('Invalid public request was not rejected')


def audit(run, output):
    started = time.monotonic()
    manifest = read(run / 'run_manifest.json')
    if (manifest['status'] != 'complete' or manifest['kind'] != 'transformer_direct'
            or manifest['version'] != 'V9-R9' or manifest['epoch_completed'] != 60):
        raise ValueError('A completed R9 Transformer is required')
    frozen_inputs(ROOT)
    identity = {p.relative_to(ROOT).as_posix(): digest(p) for p in [
        run / 'run_manifest.json', run / 'best_model.pt',
        ROOT / 'src/foodcomp/research_inference.py',
        ROOT / 'src/foodcomp/research_transformer_r9.py', Path(__file__)]}
    if digest(run / 'best_model.pt') != manifest['checkpoint_sha256']:
        raise ValueError('Completed checkpoint changed')
    torch.set_num_threads(2)
    model = TransformerNutritionModel(run / 'best_model.pt', ROOT, 'cpu')
    data = model.data
    eligible = data.axes.loss_group.eq('nutrition').to_numpy() & data.axes.loss_eligible.to_numpy(bool)
    row = next(int(i) for i in data.train if
               (data.observed[i] & eligible).sum() >= 3
               and (~data.observed[i] & data.axes.loss_eligible.to_numpy(bool)).any())
    names = list(dict.fromkeys(data.profiles.original_name.iloc[data.train].astype(str)))[:3]
    if len(names) != 3 or any(name not in model._name_index for name in names):
        raise ValueError('Three cached training names required')
    known = np.flatnonzero(data.observed[row] & eligible)[:3]
    # Access only this training profile. Raw values are never emitted.
    raw = data.scale[known] * np.expm1(data.values[row, known].astype(float))
    observed = {int(axis): float(value) for axis, value in zip(known, raw)}
    explicit_zero = {int(known[0]): 0.0}
    checks = {}
    vectors = {mode: model.encode(names[0], observed, modality=mode)
               for mode in ['name', 'nutrition', 'fused']}
    for vector in vectors.values():
        if vector.shape != (manifest['spec']['d_model'],) or not np.isfinite(vector).all():
            raise AssertionError('Invalid representation output')
    np.testing.assert_array_equal(vectors['name'], model.encode(names[0], explicit_zero, modality='name'))
    np.testing.assert_array_equal(vectors['name'], model.encode(names[0], {}, modality='name'))
    np.testing.assert_array_equal(vectors['nutrition'], model.encode(names[1], observed, modality='nutrition'))
    np.testing.assert_array_equal(vectors['nutrition'], model.encode(None, observed, modality='nutrition'))
    checks['all_three_modalities_finite_expected_width'] = True
    checks['name_representation_independent_of_numeric_context'] = True
    checks['nutrition_representation_independent_of_supplied_name'] = True
    checks['unknown_modality_rejected'] = rejected(model.encode, names[0], observed, modality='unknown')
    unknown = int(np.flatnonzero(~data.observed[row] & data.axes.loss_eligible.to_numpy(bool))[0])
    predicted = model.predict(names[0], observed, [unknown])
    if len(predicted) != 1 or not np.isfinite(list(predicted.values())).all():
        raise AssertionError('Unobserved target prediction failed')
    checks['caller_supplied_unobserved_axis_predicted'] = True
    checks['visible_target_rejected'] = rejected(model.predict, names[0], observed, [int(known[0])])
    context_only = int(np.flatnonzero(~data.axes.loss_eligible.to_numpy(bool))[0])
    checks['context_only_output_rejected'] = rejected(model.predict, names[0], {}, [context_only])
    result = model.retrieve_names(observed, names, top_k=3)
    repeat = model.retrieve_names(observed, list(reversed(names)) + names[:1], top_k=3)
    if result != repeat:
        raise AssertionError('Candidate ordering or duplicates changed the ranking')
    if 'food_name' in inspect.signature(model.retrieve_names).parameters:
        raise AssertionError('Nutrition query interface receives a name')
    sorted_names = sorted(names)
    values, visible = model.profile_arrays(observed)
    axes = np.flatnonzero(visible[0])
    candidates = np.log1p(model.candidate_profiles(sorted_names, target_axes=axes) / data.scale[axes])
    distances = ((candidates - values[:, axes]) ** 2).mean(1)
    expected = np.argsort(distances, kind='stable')
    if [item['name'] for item in result] != [sorted_names[i] for i in expected]:
        raise AssertionError('Public ranking differs from name-only profile matching')
    np.testing.assert_array_equal([item['score'] for item in result], -distances[expected])
    if any(item['score_type'] != 'negative_scaled_log_mse; not probability' for item in result):
        raise AssertionError('Public scores incorrectly describe probabilities')
    zero_result = model.retrieve_names(explicit_zero, names, top_k=2)
    if len(zero_result) != 2 or not np.isfinite([r['score'] for r in zero_result]).all():
        raise AssertionError('Explicit zero is not retained as a retrieval observation')
    checks['retrieval_query_has_no_name_parameter'] = True
    checks['retrieval_uses_name_predicted_profiles_and_negative_mse'] = True
    checks['candidate_order_and_duplicate_names_invariant'] = True
    checks['explicit_zero_retrieval_query_retained'] = True
    checks['empty_numeric_retrieval_query_rejected'] = rejected(model.retrieve_names, {}, names)
    if model._encoder is not None:
        raise AssertionError('Audit unexpectedly created new text embeddings')
    for path, expected_hash in identity.items():
        if digest(ROOT / path) != expected_hash:
            raise ValueError('Audit input changed: ' + path)
    frozen_inputs(ROOT)
    receipt = {
        'status': 'complete_public_interface_contract_check',
        'run': run.relative_to(ROOT).as_posix(), 'seed': manifest['seed'],
        'checkpoint_sha256': manifest['checkpoint_sha256'], 'input_hashes': identity,
        'checks': checks, 'device': 'cpu', 'cpu_threads': 2,
        'representation_width': manifest['spec']['d_model'],
        'candidate_training_names': len(names), 'training_query_profiles': 1,
        'new_text_embeddings_created': False, 'training_performed': False,
        'data_modified': False, 'baseline_refit': False, 'complete_test_opened': False,
        'validation_query_values_used': False, 'new_performance_result': False,
        'scope': 'One completed checkpoint, cached train names and one train profile. '
                 'Checks public modality/query/ranking contracts only. Axis-token mean '
                 'representations are probes, not trained contrastive embeddings. '
                 'This does not establish transfer, retrieval quality, or all-name coverage.',
        'elapsed_seconds': time.monotonic() - started,
    }
    write_json(output / 'verification.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run, output = args.run.resolve(), args.output_dir.resolve()
    run.relative_to(ROOT)
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError('Keep previous audit versions')
    output.mkdir(parents=True)
    try:
        result = audit(run, output)
        print(json.dumps({'status': result['status'], 'checks_passed': len(result['checks']),
                          'elapsed_seconds': result['elapsed_seconds']}))
    except Exception as error:
        write_json(output / 'failure.json', {'status': 'failed',
            'error_type': type(error).__name__, 'error': str(error),
            'new_performance_result': False, 'complete_test_opened': False})
        raise


if __name__ == '__main__':
    main()
