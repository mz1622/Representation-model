"""Final-report statistics: average independent model errors, never predictions."""
import numpy as np
import pandas as pd

from .research_confirmation import SEEDS
from .research_r0 import score_predictions

RATES = ['mrr', 'recall_at_1', 'recall_at_5', 'recall_at_10']
RATE_KEYS = ['visible_fraction', 'profile_index', 'source_key',
             'exact_name_group_id', 'observed_axes']
CELL_KEYS = ['exact_name_group_id', 'axis_index']
ERRORS = ['scaled_log_mae', 'log_mae', 'raw_mae']
CONTRIBUTIONS = ['positive_under', 'positive_over', 'explicit_zero']


def retrieval_identity_for_run(metadata, manifest):
    identity = {k: metadata[k] for k in ['candidate_sha256', 'candidate_count', 'query_profiles',
        'data_sha256', 'scoring', 'correct_answers']}
    # Neural retrieval records its checkpoint instead of repeating the cache fingerprint.
    # The caller independently verifies that checkpoint and its audited manifest.
    if 'name_cache_sha256' in metadata:
        cache = metadata['name_cache_sha256']
    elif (manifest.get('checkpoint_hash') is not None
          and metadata.get('checkpoint_sha256') == manifest['checkpoint_hash']):
        cache = manifest['name_cache_hash']
    else:
        raise ValueError('Retrieval has no verified cache/checkpoint identity.')
    if cache != manifest['name_cache_hash'] or identity['data_sha256'] != manifest['data_hash']:
        raise ValueError('Retrieval and training input identities differ.')
    identity['name_cache_sha256'] = cache
    return identity


def number_summary(values):
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError('Summary requires finite, nonempty scalar observations.')
    return {'count': len(array), 'mean': float(array.mean()),
            'sample_std': float(array.std(ddof=1)) if len(array) > 1 else None,
            'min': float(array.min()), 'max': float(array.max())}


def mean_seed_frames(frames, keys, columns):
    """Require all three seeds with identical coverage; supplied values are scores."""
    if set(frames) != set(SEEDS):
        raise ValueError('Exactly the three registered seeds are required.')
    reference, values = None, []
    for seed in SEEDS:
        frame = frames[seed].sort_values(keys).reset_index(drop=True)
        if not len(frame) or frame.duplicated(keys).any() or frame[keys].isna().any().any():
            raise ValueError('Empty, duplicate or missing score keys.')
        if reference is None:
            reference = frame[keys].copy()
        elif not reference.equals(frame[keys]):
            raise ValueError('Seed panels differ.')
        value = frame[columns].to_numpy(float)
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError('Nonfinite or negative score.')
        values.append(value)
    result = reference.copy()
    result[columns] = np.mean(values, axis=0)
    return result


def retrieval_groups(ranks, metadata):
    """Source-equal food groups; validate ranks before any reduction."""
    if metadata['complete_test_opened'] or metadata['candidate_count'] <= 0:
        raise ValueError('Invalid retrieval scope.')
    frame = ranks.copy()
    if (not len(frame) or frame[RATE_KEYS].isna().any().any()
            or frame.duplicated(['visible_fraction', 'profile_index']).any()):
        raise ValueError('Invalid or duplicate query keys.')
    values = frame['rank'].to_numpy(float)
    observed = frame.observed_axes.to_numpy(float)
    if (not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all()
            or not ((values >= 1) & (values <= metadata['candidate_count'])).all()
            or not np.isfinite(observed).all() or (observed < 3).any()
            or not np.equal(observed, np.floor(observed)).all()):
        raise ValueError('Invalid retrieval rank or query support.')
    if set(frame.visible_fraction) != {.3, 1.0}:
        raise ValueError('Expected the two registered retrieval panels.')
    frame['mrr'] = 1 / values
    for k in [1, 5, 10]:
        frame[f'recall_at_{k}'] = (values <= k).astype(float)
    result = {}
    for fraction, subset in frame.groupby('visible_fraction'):
        key = str(float(fraction))
        if len(subset) != metadata['query_profiles'][key]:
            raise ValueError('Query counts differ from the registered panel.')
        result[key] = (subset.groupby(['exact_name_group_id', 'source_key'])[RATES]
                       .mean().groupby('exact_name_group_id').mean().reset_index())
    return frame[RATE_KEYS].sort_values(RATE_KEYS).reset_index(drop=True), result


def prediction_details(data, predictions):
    """Shared evaluator plus additive strata and per-source summaries."""
    score, axes, groups = score_predictions(data, predictions)
    jobs = data.targets_for_jobs().merge(predictions[['profile_index', 'axis_index', 'prediction']],
        on=['profile_index', 'axis_index'], validate='one_to_one')
    jobs = jobs.merge(data.profiles[['profile_index', 'exact_name_group_id', 'source_key']],
        on='profile_index', validate='many_to_one')
    cells = jobs.groupby(CELL_KEYS + ['source_key'], as_index=False).agg(
        target=('target', 'median'), prediction=('prediction', 'median'))
    target, prediction = cells.target.to_numpy(), cells.prediction.to_numpy()
    scale = data.scale[cells.axis_index.to_numpy()]
    signed = np.log1p(prediction / scale) - np.log1p(target / scale)
    cells['scaled_log_mae'] = np.abs(signed)
    cells['log_mae'] = np.abs(np.log1p(prediction) - np.log1p(target))
    cells['raw_mae'] = np.abs(prediction - target)
    cells['positive_under'] = np.maximum(-signed, 0) * (target > 0)
    cells['positive_over'] = np.maximum(signed, 0) * (target > 0)
    cells['explicit_zero'] = np.abs(signed) * (target == 0)
    details = cells.groupby(CELL_KEYS)[ERRORS + CONTRIBUTIONS].mean().reset_index()
    np.testing.assert_allclose(details[CONTRIBUTIONS].sum(axis=1), details.scaled_log_mae,
                               rtol=1e-12, atol=1e-14)
    merged = details.merge(groups[CELL_KEYS + ['scaled_log_mae']], on=CELL_KEYS,
                           suffixes=('', '_reference'), validate='one_to_one')
    np.testing.assert_array_equal(merged.scaled_log_mae, merged.scaled_log_mae_reference)
    nutrition = data.axes.loc[data.axes.loss_group.eq('nutrition') & data.axes.loss_eligible, 'axis_index']
    sources = []
    for source, subset in cells[cells.axis_index.isin(nutrition)].groupby('source_key'):
        by_axis = subset.groupby('axis_index')[ERRORS + CONTRIBUTIONS].mean()
        sources.append({'source_key': source, 'supported_nutrition_axes': len(by_axis),
            'candidate_groups': subset.exact_name_group_id.nunique(), **by_axis.mean().to_dict()})
    for label, condition in [('positive', target > 0), ('zero', target == 0)]:
        support = cells.loc[condition].groupby('axis_index').exact_name_group_id.nunique()
        axes[f'{label}_candidate_support'] = axes.axis_index.map(support).fillna(0).astype(int)
    return score, axes, details, pd.DataFrame(sources)
