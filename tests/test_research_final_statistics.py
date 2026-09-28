import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from foodcomp.research_confirmation import SEEDS
from foodcomp.research_final_statistics import (
    mean_seed_frames, number_summary, prediction_details, retrieval_groups, retrieval_identity_for_run)


def test_seed_scores_are_averaged_without_best_seed_selection():
    frames = {seed: pd.DataFrame({'group': ['a', 'b'], 'error': [i + 1., 3. - i]})
              for i, seed in enumerate(SEEDS)}
    result = mean_seed_frames(frames, ['group'], ['error'])
    np.testing.assert_array_equal(result.error, [2, 2])
    assert number_summary([1])['sample_std'] is None
    assert number_summary([1, 2, 3])['sample_std'] == 1


def test_retrieval_neural_checkpoint_and_tree_cache_identify_same_input():
    manifest = {'data_hash': 'data', 'name_cache_hash': 'cache', 'checkpoint_hash': 'ckpt'}
    metadata = {'candidate_sha256': 'candidates', 'candidate_count': 10,
        'query_profiles': {}, 'data_sha256': 'data', 'scoring': 's', 'correct_answers': 'exact'}
    neural = retrieval_identity_for_run({**metadata, 'checkpoint_sha256': 'ckpt'}, manifest)
    tree = retrieval_identity_for_run({**metadata, 'name_cache_sha256': 'cache'}, manifest)
    assert neural == tree
    with pytest.raises(ValueError):
        retrieval_identity_for_run({**metadata, 'checkpoint_sha256': 'wrong'}, manifest)
    with pytest.raises(ValueError):
        retrieval_identity_for_run({**metadata, 'name_cache_sha256': 'wrong'}, manifest)


@pytest.mark.parametrize('failure', ['seed', 'coverage', 'nan', 'duplicate'])
def test_seed_validation_rejects_incomplete_or_invalid_inputs(failure):
    frames = {seed: pd.DataFrame({'group': ['a', 'b'], 'error': [1., 2.]}) for seed in SEEDS}
    if failure == 'seed':
        frames.pop(SEEDS[-1])
    elif failure == 'coverage':
        frames[SEEDS[-1]].loc[0, 'group'] = 'c'
    elif failure == 'nan':
        frames[SEEDS[-1]].loc[0, 'error'] = np.nan
    else:
        frames[SEEDS[-1]] = pd.concat([frames[SEEDS[-1]], frames[SEEDS[-1]]])
    with pytest.raises(ValueError):
        mean_seed_frames(frames, ['group'], ['error'])


def rank_example():
    ranks = pd.DataFrame({'visible_fraction': [.3] * 3 + [1.] * 3,
        'profile_index': [0, 1, 2] * 2, 'source_key': ['x', 'x', 'y'] * 2,
        'exact_name_group_id': ['a'] * 6, 'observed_axes': [3] * 6,
        'rank': [1, 1, 4] * 2})
    meta = {'complete_test_opened': False, 'candidate_count': 10,
            'query_profiles': {'0.3': 3, '1.0': 3}}
    return ranks, meta


def test_retrieval_sources_equal_not_profile_count_weighted():
    ranks, meta = rank_example()
    _, groups = retrieval_groups(ranks, meta)
    assert groups['1.0'].mrr.iloc[0] == .625  # (source x: 1 + source y: 1/4)/2
    assert groups['1.0'].recall_at_1.iloc[0] == .5


@pytest.mark.parametrize('bad', [np.nan, np.inf, 0, 1.5, 11])
def test_invalid_retrieval_ranks_fail(bad):
    ranks, meta = rank_example()
    ranks['rank'] = ranks['rank'].astype(float)
    ranks.loc[0, 'rank'] = bad
    with pytest.raises(ValueError):
        retrieval_groups(ranks, meta)


def test_additive_positive_zero_errors_keep_main_denominator():
    data = SimpleNamespace(
        profiles=pd.DataFrame({'profile_index': [0, 1, 2], 'exact_name_group_id': ['a', 'a', 'b'],
                               'source_key': ['x', 'y', 'x']}),
        axes=pd.DataFrame({'axis_index': [0, 1], 'canonical_name': ['A', 'B'],
                          'loss_group': ['nutrition', 'food_metabolome'], 'loss_eligible': [True, True]}),
        targets=np.array([0, 1]), scale=np.array([1., 1.]))
    jobs = pd.DataFrame({'profile_index': [0, 1, 2, 0], 'axis_index': [0, 0, 0, 1],
                         'target': [3., 0., 1., 1.]})
    data.targets_for_jobs = lambda: jobs.copy()
    predictions = jobs.drop(columns='target').assign(prediction=[1., 1., 3., 1.])
    score, axes, groups, sources = prediction_details(data, predictions)
    total = groups[groups.axis_index.eq(0)][['positive_under', 'positive_over', 'explicit_zero']].mean().sum()
    assert total == pytest.approx(score['nutrition']['scaled_log_mae'])
    assert axes.loc[axes.axis_index.eq(0), 'zero_candidate_support'].item() == 1
    assert np.isfinite(sources.scaled_log_mae).all()
