"""Identify nested, validation-selected training-budget windows without refitting."""
import numpy as np


def validate_budget_snapshot(history, parent, saved, budget):
    if budget not in [20, 60]:
        raise ValueError('Only the registered 20/60 windows are permitted.')
    if (parent['status'] != 'complete' or parent['version'] != 'V9-R9'
            or parent['complete_test_opened'] or parent['baseline_refit'] or parent['data_modified']):
        raise ValueError('A completed, unchanged R9 parent is required.')
    if parent['spec']['epochs'] != 60 or parent['spec']['schedule_epochs'] != 60:
        raise ValueError('Budget comparison requires a common 60-epoch schedule.')
    if not np.array_equal(history.epoch, np.arange(1, 61)):
        raise ValueError('Missing, repeated or unordered training epochs.')
    if not np.isfinite(history.select_dtypes('number')).all().all():
        raise ValueError('Nonfinite training history.')
    if saved['version'] != 'V9-R9' or saved['kind'] != 'transformer_direct':
        raise ValueError('This analysis requires an R9 Transformer.')
    if saved['spec'] != parent['spec'] or saved['seed'] != parent['seed']:
        raise ValueError('Checkpoint recipe differs from its parent.')
    for key in ['data_hash', 'name_cache_hash', 'panel_hash']:
        if saved[key] != parent[key]:
            raise ValueError('Checkpoint input identity differs from its parent: ' + key)
    window = history[history.epoch <= budget]
    row = window.loc[window.validation_primary.idxmin()]
    if saved['best_epoch'] != int(row.epoch):
        raise ValueError('Checkpoint is not the earliest primary minimum in this window.')
    return {'budget_epochs': budget, 'selected_epoch': int(row.epoch),
        'validation_primary': float(row.validation_primary),
        'recorded_seconds_through_budget': float(window.iloc[-1].elapsed_seconds),
        'window_selection': 'earliest strictly minimal full-panel nutrition primary within the fixed horizon',
        'interpretation': 'Nested budget-and-selection policies on one trajectory, not two independent runs or a pure fixed-epoch causal effect. Longer windows get additional validation-selection opportunities.'}
