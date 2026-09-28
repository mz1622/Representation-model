from copy import deepcopy
import numpy as np
import pandas as pd
import pytest
from foodcomp.research_budget_window import validate_budget_snapshot


def inputs():
    history = pd.DataFrame({'epoch': np.arange(1, 61), 'validation_primary': np.linspace(1., .5, 60),
                            'elapsed_seconds': np.arange(1, 61) * 100.})
    parent = {'status': 'complete', 'version': 'V9-R9', 'complete_test_opened': False,
        'baseline_refit': False, 'data_modified': False, 'spec': {'epochs': 60, 'schedule_epochs': 60},
        'seed': 20260922, 'data_hash': 'data', 'name_cache_hash': 'name', 'panel_hash': 'panel'}
    saved = {k: deepcopy(parent[k]) for k in ['version', 'spec', 'seed', 'data_hash', 'name_cache_hash', 'panel_hash']}
    saved.update(kind='transformer_direct', best_epoch=20)
    return history, parent, saved


def test_window_uses_earliest_validation_minimum_not_last_epoch():
    history, parent, saved = inputs()
    history.loc[[6, 19], 'validation_primary'] = .4
    saved['best_epoch'] = 7
    result = validate_budget_snapshot(history, parent, saved, 20)
    assert result['selected_epoch'] == 7
    assert result['recorded_seconds_through_budget'] == 2000
    saved['best_epoch'] = 20
    with pytest.raises(ValueError):
        validate_budget_snapshot(history, parent, saved, 20)


@pytest.mark.parametrize('change', ['incomplete', 'nan', 'missing_epoch', 'wrong_recipe', 'wrong_data', 'wrong_epoch', 'refit'])
def test_invalid_budget_evidence_fails(change):
    history, parent, saved = inputs()
    if change == 'incomplete': parent['status'] = 'running'
    elif change == 'nan': history.loc[0, 'validation_primary'] = np.nan
    elif change == 'missing_epoch': history = history.iloc[1:]
    elif change == 'wrong_recipe': saved['spec']['schedule_epochs'] = 20
    elif change == 'wrong_data': saved['data_hash'] = 'other'
    elif change == 'wrong_epoch': saved['best_epoch'] = 21
    else: parent['baseline_refit'] = True
    with pytest.raises(ValueError):
        validate_budget_snapshot(history, parent, saved, 20)
