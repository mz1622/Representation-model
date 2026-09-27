"""R8 selection uses the complete registered search and the same128-name input."""
import math


def select_registered_references(registered, records):
    """Return point-screen results, never scientific acceptance of a single seed."""
    names = [item['name'] for item in registered]
    if len(names) != 12 or len(set(names)) != 12 or set(records) != set(names):
        raise ValueError('The complete unique twelve-configuration R8 budget is required.')
    for item in registered:
        record = records[item['name']]
        if record['registered_configuration'] != item:
            raise ValueError('A completed record differs from the registered configuration.')
        if record['status'] != 'complete' or record['complete_test_opened']:
            raise ValueError('All registered records must be complete and test-closed.')
        if record['metrics']['completion']['nutrition']['axes'] != 142:
            raise ValueError('Changed primary axis coverage.')
        for task in ['completion', 'name_only']:
            for metric in ['scaled_log_mae', 'log_mae']:
                value = record['metrics'][task]['nutrition'][metric]
                if not math.isfinite(value) or value < 0:
                    raise ValueError('Nonfinite or negative observed metric.')
    neural = [item for item in registered if item.get('kind') == 'mlp' and item['active_dimensions'] == 128]
    if len(neural) != 1:
        raise ValueError('Exactly one registered128-input neural candidate is required.')
    selected, searched = {}, {}
    for kind in ['rf', 'xgb']:
        configs = [item for item in registered if item.get('kind') == kind and item['active_dimensions'] == 128]
        if len(configs) != 3:
            raise ValueError('Both methods require their complete three-configuration128-input search.')
        # Registration order breaks a bitwise equality; no auxiliary task selects a tree.
        searched[kind] = [item['name'] for item in configs]
        selected[kind] = min(searched[kind], key=lambda name:
            records[name]['metrics']['completion']['nutrition']['scaled_log_mae'])
    stronger = min(selected.values(), key=lambda name:
        (records[name]['metrics']['completion']['nutrition']['scaled_log_mae'], names.index(name)))
    n = records[neural[0]['name']]['metrics']['completion']['nutrition']
    t = records[stronger]['metrics']['completion']['nutrition']
    if t['scaled_log_mae'] <= 0 or t['log_mae'] <= 0:
        raise ValueError('Relative improvement requires positive baseline errors.')
    gain = 1 - n['scaled_log_mae'] / t['scaled_log_mae']
    regression = n['log_mae'] / t['log_mae'] - 1
    return {'searched_same128_input': searched, 'selected_by_method': selected,
        'stronger_same_input_tree': stronger, 'matched_neural': neural[0]['name'],
        'completion_relative_improvement': gain, 'legacy_log_relative_regression': regression,
        'point_screen_passed': n['scaled_log_mae'] <= .95 * t['scaled_log_mae'] and n['log_mae'] <= 1.02 * t['log_mae'],
        'screen_thresholds': {'minimum_primary_improvement': .05, 'maximum_legacy_regression': .02},
        'three_seed_confirmation_required': True, 'scientific_confirmation': False,
        'model_improvement_accepted': False,
        'scope': 'Single-seed point screening after the entire registered same-input search. '
                 'Intervals and three-seed results remain required; no task-specific best-score splicing.'}
