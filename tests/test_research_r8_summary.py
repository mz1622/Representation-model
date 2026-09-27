import copy
import json
from pathlib import Path

import pytest

from foodcomp.research_r8_summary import select_registered_references


def fixture_records():
    root = Path(__file__).resolve().parents[1]
    registered = json.loads((root / 'experiments/foodnutrigpt_v9_research/r8/config.json').read_text(encoding='utf-8'))['candidates']
    records = {item['name']: {'registered_configuration': copy.deepcopy(item), 'status': 'complete',
        'complete_test_opened': False, 'metrics': {task: {'nutrition': {'axes': 142,
        'scaled_log_mae': .3, 'log_mae': .1}} for task in ['completion', 'name_only']}}
        for item in registered}
    return registered, records


def test_selection_uses_all_same_input_configs_and_primary_not_other_tasks():
    registered, records = fixture_records()
    records['xgb800d10_name32']['metrics']['completion']['nutrition']['scaled_log_mae'] = .01
    records['rf400leaf1half_name128']['metrics']['completion']['nutrition']['scaled_log_mae'] = .20
    records['xgb800d14_name128']['metrics']['completion']['nutrition']['scaled_log_mae'] = .18
    records['xgb800d6_name128']['metrics']['name_only']['nutrition']['scaled_log_mae'] = .001
    records['mlp512_name128']['metrics']['completion']['nutrition']['scaled_log_mae'] = .17
    result = select_registered_references(registered, records)
    assert result['selected_by_method'] == {'rf': 'rf400leaf1half_name128', 'xgb': 'xgb800d14_name128'}
    assert result['stronger_same_input_tree'] == 'xgb800d14_name128'
    assert result['point_screen_passed']
    assert result['three_seed_confirmation_required'] and not result['model_improvement_accepted']
    records['mlp512_name128']['metrics']['completion']['nutrition']['log_mae'] = .103
    assert not select_registered_references(registered, records)['point_screen_passed']


@pytest.mark.parametrize('mutation', ['missing', 'incomplete', 'open_test', 'nonfinite', 'changed_config'])
def test_incomplete_or_invalid_evidence_cannot_select_reference(mutation):
    registered, records = fixture_records()
    target = records['rf400leaf1all_name128']
    if mutation == 'missing':
        del records['rf400leaf1all_name128']
    elif mutation == 'incomplete':
        target['status'] = 'running'
    elif mutation == 'open_test':
        target['complete_test_opened'] = True
    elif mutation == 'nonfinite':
        target['metrics']['completion']['nutrition']['scaled_log_mae'] = float('nan')
    else:
        target['registered_configuration']['trees'] = 20
    with pytest.raises(ValueError):
        select_registered_references(registered, records)


def test_exact_ties_use_registration_order_and_screen_includes_boundary():
    registered, records = fixture_records()
    for item in registered:
        if item.get('kind') in ['xgb', 'rf'] and item['active_dimensions'] == 128:
            records[item['name']]['metrics']['completion']['nutrition']['scaled_log_mae'] = .2
    records['mlp512_name128']['metrics']['completion']['nutrition'].update(scaled_log_mae=.19, log_mae=.102)
    result = select_registered_references(registered, records)
    assert result['stronger_same_input_tree'] == 'xgb800d6_name128'
    assert result['point_screen_passed'] and not result['scientific_confirmation']
