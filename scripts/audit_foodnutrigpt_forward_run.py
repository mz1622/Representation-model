"""Verify complete forward exposure, public loading, modality isolation and rank replay."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import digest,write_json,score_predictions
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_inference import NutritionModel
from foodcomp.research_neural import evaluate,training_batch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--candidate',type=Path,default=ROOT/'output/v9_r5/name_mlp60_legacy_loss')
    p.add_argument('--reference',type=Path,default=ROOT/'output/v9_r5/name_mlp8_replay_v1')
    p.add_argument('--retrieval',type=Path,default=ROOT/'output/v9_r5/retrieval_name_mlp60_legacy_loss')
    p.add_argument('--loss-ablation',action='store_true');args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    run=args.candidate;old=args.reference
    receipt={'status':'incomplete','script_sha256':digest(Path(__file__)),'complete_test_opened':False}
    try:
        m=json.loads((run/'run_manifest.json').read_text());a=json.loads((old/'run_manifest.json').read_text())
        assert m['status']=='complete' and m['epoch_completed']==60 and not m['complete_test_opened']
        for key in ['data_hash','protocol_hash','name_cache_hash','initial_state_sha256','parameter_count','training_profiles',
            'observed_target_cells','training_rows_sha256','training_target_values_sha256','training_target_mask_sha256','training_weights_sha256']:
            assert m[key]==a[key],key
        if args.loss_ablation:
            assert m['args']['objective']=='mae' and a['args'].get('objective','smooth_l1')=='smooth_l1'
            assert m['args']['epochs']==a['args']['epochs']==60
            for path,expected in a['code_hashes'].items():
                if Path(path).name!='train_foodnutrigpt_v9_r5_forward.py':assert m['code_hashes'][path]==expected
        else:assert m['code_hashes']==a['code_hashes']
        for path,expected in m['code_hashes'].items():assert digest(run/'code_snapshot'/Path(path).name)==expected
        model=NutritionModel(run/'best_model.pt');data=model.data;text=model._cached_text
        active=data.train[data.observed[data.train][:,data.targets].any(1)]
        arrays={'training_rows':active,'training_target_values':data.values[active][:,data.targets],
            'training_target_mask':data.observed[active][:,data.targets],'training_weights':data.weights[active][:,data.targets]}
        for key,x in arrays.items():assert m[key+'_sha256']==fingerprint_array(x)
        h=pd.read_csv(run/'history.csv',float_precision='round_trip');np.testing.assert_array_equal(h.epoch,np.arange(1,61))
        assert np.isfinite(h.select_dtypes('number')).all().all()
        opt=torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))],lr=.001)
        schedule=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60,eta_min=.00001)
        for row in h.itertuples():
            assert row.training_profiles==len(active) and row.observed_target_cells==arrays['training_target_mask'].sum()
            assert row.training_order_sha256==fingerprint_array(np.random.default_rng(m['seed']+row.epoch).permutation(active))
            assert row.learning_rate==opt.param_groups[0]['lr'];opt.step();schedule.step()
        assert m['best_epoch']==int(h.loc[h.validation_primary.idxmin(),'epoch'])
        assert digest(run/'best_model.pt')==m['checkpoint_hash']
        saved=json.loads((run/'metrics.json').read_text());frames=[]
        for mode in ['completion','name_only']:
            f=evaluate(model.model,data,text,model.device,mode)
            pd.testing.assert_frame_equal(f,pd.read_parquet(run/f'{mode}_predictions.parquet'),check_exact=True)
            assert score_predictions(data,f)[0]==saved[mode];frames.append(f)
        b=training_batch(data,text,active[:32],model.device,np.random.default_rng(11),'name_mlp')
        with torch.no_grad():
            before=model.model(b)['amount_normalized'];b['value'].fill_(123.);b['masked'].fill_(False)
            assert torch.equal(before,model.model(b)['amount_normalized'])
        name=str(data.profiles.iloc[active[0]].original_name);axis=int(data.targets[0]);other=int(data.targets[1])
        assert model.predict(name,{},[axis])==model.predict(name,{other:1.},[axis])
        try:model.encode(food_name=name,observed_profile={},modality='nutrition')
        except ValueError:pass
        else:raise AssertionError('Name-only baseline claimed a nutrition encoder.')
        replay=args.output_dir/'retrieval_replay'
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_foodnutrigpt_v9_r0_retrieval.py'),
            '--checkpoint',str(run/'best_model.pt'),'--output-dir',str(replay.resolve())],cwd=ROOT,check=True,capture_output=True,text=True)
        reference=args.retrieval
        pd.testing.assert_frame_equal(pd.read_parquet(replay/'ranks.parquet'),pd.read_parquet(reference/'ranks.parquet'),check_exact=True)
        rmeta=json.loads((replay/'metrics.json').read_text());original=json.loads((reference/'metrics.json').read_text())
        for key in ['metrics','candidate_count','candidate_sha256','query_profiles','checkpoint_sha256','data_sha256','scoring','correct_answers']:assert rmeta[key]==original[key]
        assert original['candidate_sha256']=='e5f4910d9eebe3feb0f9ed0395420596ff7e2b2bbee5f41cb9df5d71d91df87f'
        joined=frames[0].merge(frames[1],on=['profile_index','axis_index'],suffixes=('_completion','_name'),validate='one_to_one')
        maxdiff=float(np.abs(joined.prediction_completion-joined.prediction_name).max())
        receipt.update(status='complete',checkpoint_sha256=m['checkpoint_hash'],manifest_sha256=digest(run/'run_manifest.json'),
            all60_orders_exposures_and_learning_rates_verified=True,same_initialization_labels_weights_as_reference=True,
            pointwise_loss_changed=args.loss_ablation,run=str(run),reference=str(old),
            both_prediction_tables_and_scores_reload_exact=True,all19089_retrieval_ranks_and_metrics_reload_exact=True,
            trained_name_model_predictions_independent_of_numeric_context=True,nutrition_representation_claim_rejected=True,
            completion_vs_name_max_raw_prediction_difference=maxdiff,
            completion_minus_name_primary=saved['completion']['nutrition']['scaled_log_mae']-saved['name_only']['nutrition']['scaled_log_mae'],
            scope='Complete60-epoch single-seed forward baseline; pointwise-loss intervention explicitly flagged. Both evaluation tables individually replay exactly despite batch-shape roundoff across modes.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt)


if __name__=='__main__':main()
