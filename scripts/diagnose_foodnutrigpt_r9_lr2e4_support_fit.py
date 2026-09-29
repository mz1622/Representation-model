"""Describe support-stratified fit from completed aggregate metrics; no new inference."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from foodcomp.research_r0 import digest, write_json


def main():
    out = ROOT/'reports/v9_r9_lr2e4_support_fit_v1'
    if out.exists():
        raise FileExistsError(out)
    fit_dir = ROOT/'reports/v9_r9_lr2e4_fit_v1'
    paths = [fit_dir/'summary.json', fit_dir/'per_axis_fit.csv',
             ROOT/'reports/v9_r9_lr2e4_axis_changes_v1/axis_changes.csv', Path(__file__)]
    hashes = {p.relative_to(ROOT).as_posix():digest(p) for p in paths}
    fit = json.loads(paths[0].read_text(encoding='utf-8'))
    if fit['status'] != 'complete' or fit['complete_test_opened']:
        raise ValueError('Require completed, test-closed fit')
    frame = pd.read_csv(paths[1],float_precision='round_trip')
    frame = frame[frame.loss_group.eq('nutrition')]
    axis = pd.read_csv(paths[2],float_precision='round_trip')
    if len(frame) != 4*142 or frame.duplicated(['axis_index','method','partition']).any():
        raise ValueError('Expected four complete nutrition axis panels')
    keys = ['axis_index','canonical_name']
    panel = axis[keys+['training_candidates','candidate_support','macro_weight']].copy()
    panel = panel.rename(columns={'candidate_support':'validation_groups'})
    for method in ['mae','lr2e4']:
        for partition in ['train','validation']:
            rows = frame[frame.method.eq(method)&frame.partition.eq(partition)]
            if len(rows) != 142 or not np.isfinite(rows.scaled_log_mae).all():
                raise ValueError('Missing/nonfinite metric panel')
            score = fit['source_free_training_fit' if partition=='train' else 'source_free_validation'][method]['nutrition']['scaled_log_mae']
            np.testing.assert_allclose(rows.scaled_log_mae.mean(),score,rtol=0,atol=1e-12)
            col = method+'_'+partition
            panel = panel.merge(rows[keys+['candidate_support','scaled_log_mae']].rename(
                columns={'candidate_support':col+'_groups','scaled_log_mae':col}),on=keys,validate='one_to_one')
            np.testing.assert_array_equal(panel[col+'_groups'], panel['training_candidates' if partition=='train' else 'validation_groups'])
    if len(panel)!=142:
        raise ValueError('Changed axis intersection')
    for partition in ['train','validation']:
        panel[partition+'_delta'] = panel['lr2e4_'+partition]-panel['mae_'+partition]
    np.testing.assert_allclose(panel.validation_delta*panel.macro_weight,axis.lr2e4_minus_mae_mae,rtol=0,atol=1e-12)
    panel['support_bin']=np.where(panel.training_candidates<100,'below100',
        np.where(panel.training_candidates<1000,'100to999','at_least1000'))
    bins=[]
    for name,rows in panel.groupby('support_bin'):
        bins.append({'support_bin':name,'axes':len(rows),
            'training_delta_contribution':float((rows.train_delta*rows.macro_weight).sum()),
            'validation_delta_contribution':float((rows.validation_delta*rows.macro_weight).sum()),
            'training_axes_improved':int(rows.train_delta.lt(0).sum()),
            'validation_axes_improved':int(rows.validation_delta.lt(0).sum())})
    sparse=panel[panel.training_candidates<100]
    summary={'status':'complete_aggregate_support_fit_diagnosis','input_hashes':hashes,
        'support_bins':bins,'sparse_axes':sparse[['canonical_name','training_candidates','validation_groups',
            'mae_train','lr2e4_train','mae_validation','lr2e4_validation','train_delta','validation_delta']].to_dict('records'),
        'training_performed':False,'model_forward_performed':False,'data_modified':False,
        'baseline_refit':False,'complete_test_opened':False,
        'scope':'Exploratory aggregation of existing selected-checkpoint metrics. Training and validation foods/support differ. Low training errors and large validation gaps do not prove overfitting, scarcity as a unique cause, or that additional sampling or regularization will help.'}
    for relative,expected in hashes.items():
        if digest(ROOT/relative)!=expected:
            raise ValueError('Changed input: '+relative)
    out.mkdir()
    panel.to_csv(out/'axis_support_fit.csv',index=False)
    summary['axis_output_sha256']=digest(out/'axis_support_fit.csv')
    write_json(out/'summary.json',summary)
    print(json.dumps({'status':summary['status'],'support_bins':bins,'sparse_axes':summary['sparse_axes']}))


if __name__=='__main__':
    main()
