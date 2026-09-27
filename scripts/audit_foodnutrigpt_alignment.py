"""Actual train-data branch isolation, objective, weighting and loading checks."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_alignment import AlignmentPanel,NameSpace,NumericNameMapper,numeric_features,mapping_loss,state_fingerprint
from foodcomp.research_alignment_inference import AlignmentModel


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(1)
    result={'status':'incomplete','complete_test_opened':False,'script_sha256':digest(Path(__file__))}
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION);panel=AlignmentPanel(data,ROOT,'cpu')
        # No nutrition attributes even exist on this proxy: constructing the real
        # candidate bank must succeed from only names, IDs and frozen cache inputs.
        proxy=SimpleNamespace(profiles=data.profiles.copy(),train=data.train.copy())
        names_only=NameSpace(proxy,ROOT)
        np.testing.assert_array_equal(names_only.features,panel.namespace.features)
        train_groups=set(data.profiles.iloc[panel.rows].exact_name_group_id)
        assert not train_groups.intersection(data.profiles.iloc[data.validation].exact_name_group_id)
        pp=data.profiles.iloc[panel.rows][['exact_name_group_id','source_key']].copy();pp['weight']=panel.weights
        group_sums=pp.groupby('exact_name_group_id').weight.sum()
        np.testing.assert_allclose(group_sums,group_sums.iloc[0],rtol=1e-12,atol=1e-12)
        by_source=pp.groupby(['exact_name_group_id','source_key']).weight.sum().reset_index()
        spread=by_source.groupby('exact_name_group_id').weight.agg(['min','max'])
        np.testing.assert_allclose(spread['min'],spread['max'],rtol=1e-12,atol=1e-12)
        tasks=np.random.default_rng(20260923).choice(len(panel.rows),768,replace=False)
        selected_rows=panel.rows[tasks];values=data.values[selected_rows][:,panel.axes].copy();visible=data.observed[selected_rows][:,panel.axes]
        expected=numeric_features(values,visible);values[~visible]=np.nan
        np.testing.assert_array_equal(expected,numeric_features(values,visible))
        np.testing.assert_array_equal(expected,panel.features[tasks])
        initial=[];records=[]
        for objective in ['mse','contrastive']:
            torch.manual_seed(20260922);model=NumericNameMapper();initial.append(state_fingerprint(model))
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            steps=[]
            for start in range(0,768,256):
                ids=tasks[start:start+256];features=torch.as_tensor(panel.features[ids]);targets=torch.as_tensor(panel.targets[ids])
                name_ids=torch.as_tensor(panel.name_ids[ids],dtype=torch.long);weight=torch.as_tensor(panel.weights[ids],dtype=torch.float32)
                group_ids=torch.as_tensor(panel.group_ids[ids],dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                loss=mapping_loss(model(features),targets,name_ids,weight,objective=objective,population_size=len(panel.rows),weight_sum=float(panel.weights.sum()),group_ids=group_ids)
                loss.backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                steps.append({'loss':float(loss.detach()),'gradient_norm':float(norm),'unique_names':int(torch.unique(name_ids).numel())})
            model.eval();fixture=args.output_dir/f'{objective}_three_step_fixture_not_candidate';fixture.mkdir()
            torch.save({'model_state':model.state_dict(),'args':{'width':512}},fixture/'best_model.pt')
            write_json(fixture/'run_manifest.json',{'status':'complete','kind':'mlp','data_hash':panel.manifest['data_sha256'],
                'name_cache_hash':panel.manifest['name_cache_sha256'],'checkpoint_hash':digest(fixture/'best_model.pt'),'fixture_only_not_candidate':True})
            loaded=AlignmentModel(fixture,device='cpu')
            with torch.no_grad():torch.testing.assert_close(model(torch.as_tensor(expected[:64])),loaded.predict_features(expected[:64]),rtol=0,atol=0)
            names=panel.namespace.names[:3].tolist();context={int(panel.axes[0]):0.}
            np.testing.assert_array_equal(loaded.encode(names[0],context),loaded.encode(names[1],context))
            np.testing.assert_array_equal(loaded.encode(food_name=names[0],modality='name'),panel.namespace.encode([names[0]])[0])
            ranking=loaded.retrieve_names(context,names,top_k=3)
            assert len(ranking)==3 and all(np.isfinite(x['score']) and 'not probability' in x['score_type'] for x in ranking)
            try:loaded.encode(observed_profile=context,modality='fused')
            except ValueError:pass
            else:raise AssertionError('Independent specialist claimed fused capability.')
            records.append({'objective':objective,'cpu_steps':steps,'parameters':sum(p.numel() for p in model.parameters()),'fixture_sha256':digest(fixture/'best_model.pt')})
        assert initial[0]==initial[1]
        for q in panel.queries.values():assert not np.isin(q['rows'],panel.rows).any()
        result.update(status='complete',panel_manifest=panel.manifest,candidate_bank_constructed_without_any_nutrition_attributes=True,
            training_only_rows_groups_targets=True,group_and_source_equal_training_weights=True,
            hidden_values_do_not_change_features=True,query_name_does_not_enter_nutrition_mapping=True,
            identical_objective_arm_initialization=True,initial_state_sha256=initial[0],save_reload_outputs_bitwise_equal=True,
            training_candidate_group_ids_provided_for_contrastive_negative_exclusion=True,
            two_public_apis_checked=True,independent_model_rejects_fused_capability=True,task_ids_sha256=fingerprint_array(tasks),records=records,
            code_sha256={f:digest(ROOT/f) for f in ['src/foodcomp/research_alignment.py','src/foodcomp/research_alignment_inference.py','scripts/train_foodnutrigpt_v9_r5_mapping.py','scripts/run_foodnutrigpt_v9_r5_ridge.py']},
            scope='Actual768 training profiles, threeCPU steps for each objective; not fullCUDA training or evidence of retrieval performance. Namespace proxy contains no nutrition attributes. Exact-name false negatives addressed; unknown aliases remain unverified.')
    except Exception as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',result);raise
    write_json(args.output_dir/'verification.json',result)
    print({k:result[k] for k in ['status','initial_state_sha256']});print({k:panel.manifest[k] for k in ['training_rows','training_names','candidate_count']})


if __name__=='__main__':main()
