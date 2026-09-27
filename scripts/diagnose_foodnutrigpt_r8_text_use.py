"""Registered fixed-checkpoint name interventions; original benchmark results stay intact."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import digest, write_json, score_predictions
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_inference import NutritionModel
from foodcomp.research_neural import evaluate
from foodcomp.research_completion_input import execution_contract
from foodcomp.research_name_intervention import intervene_name_features


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def local_cases(data, baseline, candidate, path):
    keys = ['profile_index', 'axis_index']
    joined = data.jobs.merge(baseline[keys+['prediction']].rename(columns={'prediction':'baseline_prediction'}),
        on=keys, validate='one_to_one').merge(candidate[keys+['prediction']].rename(columns={'prediction':'candidate_prediction'}),
        on=keys, validate='one_to_one')
    axes = data.axes.loc[data.axes.loss_group.eq('nutrition') & data.axes.loss_eligible, 'axis_index']
    joined = joined[joined.axis_index.isin(axes)].copy()
    scale = data.scale[joined.axis_index.to_numpy()]
    for role in ['baseline','candidate']:
        joined[role+'_error'] = np.abs(np.log1p(joined[role+'_prediction']/scale)-np.log1p(joined.target/scale))
    joined['error_difference'] = joined.candidate_error-joined.baseline_error
    cases = pd.concat([joined.nsmallest(20,'error_difference').assign(selection='largest_error_reduction'),
        joined.nlargest(20,'error_difference').assign(selection='largest_error_increase')])
    cases = cases.merge(data.profiles[['profile_index','original_name','source_key','exact_name_group_id']],
        on='profile_index',validate='many_to_one').merge(data.axes[['axis_index','canonical_name']],on='axis_index',validate='many_to_one')
    cases.to_csv(path,index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(); out = args.output_dir.resolve()
    if not out.is_relative_to((ROOT/'data/local/research_diagnostics').resolve()):
        raise ValueError('Original names, values and predictions must stay under the ignored local data directory.')
    if out.exists(): raise FileExistsError(out)
    contract, contract_path = execution_contract(ROOT)
    assert read(ROOT/'reports/v9_r8_mlp_completed_audit_v2/verification.json')['status']=='complete'
    out.mkdir(parents=True); torch.set_num_threads(4); started=time.monotonic()
    files=[Path(__file__),ROOT/'src/foodcomp/research_name_intervention.py',ROOT/'scripts/compare_foodnutrigpt_research_predictions.py']
    snapshot=out/'code_snapshot';snapshot.mkdir()
    for file in files:shutil.copyfile(file,snapshot/file.name)
    receipt={'status':'running','version':'V9-R8-text-diagnostic-v1','complete_test_opened':False,
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'code_hashes':{str(f.relative_to(ROOT)):digest(f) for f in files},
        'plan_sha256':digest(ROOT/'experiments/foodnutrigpt_v9_research/r8/TEXT_DIAGNOSTIC_PLAN.md'),
        'execution_contract_sha256':digest(contract_path),'fitted_model_code_commit':contract['code_commit'],
        'scope':'Post-fit fixed-input interventions; no training/selection or new benchmark candidate. OOD perturbations are not natural causal contributions.',
        'records':[]}
    write_json(out/'manifest.json',receipt)
    try:
        for dims in [32,128]:
            run=ROOT/f'output/v9_r8/mlp512_name{dims}';manifest=read(run/'run_manifest.json')
            assert manifest['status']=='complete' and manifest['code_commit']==contract['code_commit']
            assert digest(run/'best_model.pt')==manifest['checkpoint_hash']
            wrapper=NutritionModel(run/'best_model.pt');data=wrapper.data
            names=data.profiles.original_name.astype(str).to_numpy()[data.validation]
            assert not (set(names)&set(data.profiles.original_name.astype(str).to_numpy()[data.train]))
            unchanged={key:fingerprint_array(getattr(data,key)) for key in ['values','observed','weights']}
            original={task:pd.read_parquet(run/f'{task}_predictions.parquet') for task in ['completion','name_only']}
            for task in original:
                replay=evaluate(wrapper.model,data,wrapper._cached_text,wrapper.device,task)
                pd.testing.assert_frame_equal(replay,original[task],check_exact=True)
                assert score_predictions(data,replay)[0]==read(run/'metrics.json')[task]
                if dims==128:
                    control=pd.read_parquet(ROOT/f'output/v9_r8/mlp512_name32/{task}_predictions.parquet')
                    local_cases(data,control,original[task],out/f'matched_mlp128_vs32_{task}_local_cases.csv')
            modes=['zero_name','permute_name']+(['drop_extra96'] if dims==128 else [])
            for mode in modes:
                altered,replacements=intervene_name_features(wrapper._cached_text[data.validation],names,mode)
                text=wrapper._cached_text.copy();text[data.validation]=altered
                np.testing.assert_array_equal(text[data.train],wrapper._cached_text[data.train])
                if mode=='drop_extra96':
                    control_cache=ROOT/read(ROOT/'reports/v9_r7_name_cache_v1/verification.json')['paths']['32']
                    np.testing.assert_array_equal(altered,np.load(control_cache/'features.npy')[data.validation])
                sub=out/f'name{dims}_{mode}';sub.mkdir()
                if replacements is not None:
                    pd.DataFrame({'profile_index':data.validation,'original_name':names,'replacement_name':replacements}).to_csv(sub/'local_name_permutation.csv',index=False)
                    reverse,reverse_names=intervene_name_features(wrapper._cached_text[data.validation][::-1],names[::-1],mode)
                    np.testing.assert_array_equal(reverse[::-1],altered);np.testing.assert_array_equal(reverse_names[::-1],replacements)
                for task in ['completion','name_only']:
                    predictions=evaluate(wrapper.model,data,text,wrapper.device,task)
                    path=sub/f'{task}_predictions.parquet';predictions.to_parquet(path,index=False)
                    local_cases(data,original[task],predictions,sub/f'{task}_local_cases.csv')
                    compare=sub/f'{task}_comparison'
                    subprocess.run([sys.executable,str(ROOT/'scripts/compare_foodnutrigpt_research_predictions.py'),
                        '--baseline',str(run/f'{task}_predictions.parquet'),'--candidate',str(path),'--task',task,
                        '--output-dir',str(compare)],cwd=ROOT,check=True,capture_output=True,text=True)
                    result=read(compare/'summary.json')
                    axis=pd.read_csv(compare/'axis_paired_intervals.csv');axis=axis[axis.loss_group.eq('nutrition')]
                    source=pd.read_csv(compare/'source_metrics.csv').pivot(index='source',columns='role',values='scaled_log_mae')
                    record={'dimensions':dims,'intervention':mode,'task':task,'checkpoint_sha256':manifest['checkpoint_hash'],
                        'changed_name_features_sha256':fingerprint_array(altered),'prediction_sha256':digest(path),
                        'comparison_sha256':digest(compare/'summary.json'),'comparison':result,
                        'axes_point_worse':int((axis.candidate_minus_baseline>0).sum()),
                        'axes_interval_worse':int((axis.difference_95_low>0).sum()),
                        'axes_interval_better':int((axis.difference_95_high<0).sum()),
                        'sources_point_worse':int((source.candidate>source.baseline).sum()),'source_count':len(source)}
                    receipt['records'].append(record)
                    receipt.update(completed_conditions=len(receipt['records']),elapsed_seconds=time.monotonic()-started)
                    write_json(out/'manifest.json',receipt)
                    ci=result['paired_intervals']['scaled_log_mae']
                    print(dims,mode,task,'relative_improvement',ci['relative_improvement'],ci['relative_improvement_95_interval'],flush=True)
                for key,expected in unchanged.items():assert fingerprint_array(getattr(data,key))==expected
            assert digest(run/'best_model.pt')==manifest['checkpoint_hash']
        assert len(receipt['records'])==10
        receipt.update(status='complete',elapsed_seconds=time.monotonic()-started,
            original_predictions_replayed_exact=True,labels_weights_observations_and_checkpoints_unchanged=True,
            permutation_row_order_and_duplicate_name_invariance=True)
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error),elapsed_seconds=time.monotonic()-started)
        write_json(out/'manifest.json',receipt);raise
    write_json(out/'manifest.json',receipt)
    print('Completed all10 diagnostic conditions; no fitting or selection.',flush=True)


if __name__=='__main__':main()
