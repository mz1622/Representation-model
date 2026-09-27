"""Actual default replay and subset-view isolation before R5 candidate8."""
import argparse
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_r1 import fingerprint_array
from foodcomp.research_alignment import AlignmentPanel,NumericNameMapper,mapping_loss,state_fingerprint
from foodcomp.research_alignment_views import partial_view_features
from foodcomp.research_alignment_inference import AlignmentModel
from audit_foodnutrigpt_v9_r3_control import load_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(1)
    receipt={'status':'incomplete','complete_test_opened':False,'script_sha256':digest(Path(__file__))}
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION);panel=AlignmentPanel(data,ROOT,'cpu')
        parent=ROOT/'output/v9_r5/mapper_contrastive60'
        old=load_snapshot('foodcomp.alignment_full_reference',parent/'code_snapshot/research_alignment.py')
        torch.manual_seed(20260922);reference=old.NumericNameMapper();initial=old.state_fingerprint(reference);rng=torch.get_rng_state()
        torch.manual_seed(20260922);model=NumericNameMapper();assert initial==state_fingerprint(model)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        tasks=np.random.default_rng(20260923).choice(len(panel.rows),768,replace=False)
        identity,_,_=partial_view_features(panel.features,0.,20260922,1)
        np.testing.assert_array_equal(identity,panel.features)
        optimizers=[torch.optim.AdamW(x.parameters(),lr=.001,weight_decay=.0001) for x in [reference,model]]
        losses=[]
        for start in range(0,768,256):
            ix=tasks[start:start+256]
            target=torch.as_tensor(panel.targets[ix]);names=torch.as_tensor(panel.name_ids[ix],dtype=torch.long)
            groups=torch.as_tensor(panel.group_ids[ix],dtype=torch.long);weights=torch.as_tensor(panel.weights[ix],dtype=torch.float32)
            for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
            a=old.mapping_loss(reference(torch.as_tensor(panel.features[ix])),target,names,weights,objective='contrastive',population_size=len(panel.rows),weight_sum=float(panel.weights.sum()),group_ids=groups)
            b=mapping_loss(model(torch.as_tensor(identity[ix])),target,names,weights,objective='contrastive',population_size=len(panel.rows),weight_sum=float(panel.weights.sum()),group_ids=groups)
            torch.testing.assert_close(a,b,rtol=0,atol=0);a.backward();b.backward()
            for x,y in zip(reference.parameters(),model.parameters()):torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
            for m,optimizer in zip([reference,model],optimizers):torch.nn.utils.clip_grad_norm_(m.parameters(),1.);optimizer.step()
            assert old.state_fingerprint(reference)==state_fingerprint(model)
            torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0);losses.append([float(a.detach()),float(b.detach())])
        augmented,mask,assigned=partial_view_features(panel.features,.5,20260922,1)
        observed=panel.features[:,142:].astype(bool)
        assert not (mask&~observed).any() and (mask.sum(1)>=3).all()
        np.testing.assert_array_equal(augmented[~assigned],panel.features[~assigned])
        # Targets, weights and names are not arguments to the view function.
        changed=panel.features.copy();changed[:,:142]+=777
        _,again,assign_again=partial_view_features(changed,.5,20260922,1)
        np.testing.assert_array_equal(mask,again);np.testing.assert_array_equal(assigned,assign_again)
        ix=tasks[:256];model.zero_grad(set_to_none=True)
        loss=mapping_loss(model(torch.as_tensor(augmented[ix])),torch.as_tensor(panel.targets[ix]),torch.as_tensor(panel.name_ids[ix],dtype=torch.long),torch.as_tensor(panel.weights[ix],dtype=torch.float32),
            objective='contrastive',population_size=len(panel.rows),weight_sum=float(panel.weights.sum()),group_ids=torch.as_tensor(panel.group_ids[ix],dtype=torch.long))
        loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        fixture=args.output_dir/'three_step_default_fixture_not_candidate';fixture.mkdir()
        torch.save({'model_state':model.state_dict(),'args':{'width':512,'partial_view_probability':.5}},fixture/'best_model.pt')
        write_json(fixture/'run_manifest.json',{'status':'complete','kind':'mlp','data_hash':panel.manifest['data_sha256'],
            'name_cache_hash':panel.manifest['name_cache_sha256'],'checkpoint_hash':digest(fixture/'best_model.pt'),'fixture_only_not_candidate':True})
        restored=AlignmentModel(fixture,device='cpu');model.eval()
        with torch.no_grad():torch.testing.assert_close(model(torch.as_tensor(augmented[ix])),restored.predict_features(augmented[ix]),rtol=0,atol=0)
        receipt.update(status='complete',tasks=768,default_optimizer_steps=3,initial_state_sha256=initial,
            default_parameters_rng_losses_gradients_updates_match_frozen_parent=True,paired_cpu_losses=losses,
            full_training_mask_is_original_subset=True,minimum_visible_three=True,unassigned_inputs_unchanged=True,
            mask_independent_of_numeric_values=True,partial_input_gradients_finite=True,save_reload_bitwise=True,
            assigned_profiles=int(assigned.sum()),changed_profiles=int(np.any(mask!=observed,axis=1).sum()),
            original_visible_cells=int(observed.sum()),partial_training_visible_cells=int(mask.sum()),
            mask_sha256=fingerprint_array(mask),assignment_sha256=fingerprint_array(assigned),
            tasks_sha256=fingerprint_array(tasks),parent_snapshot_sha256=digest(parent/'code_snapshot/research_alignment.py'),
            code_sha256={f:digest(ROOT/f) for f in ['src/foodcomp/research_alignment_views.py','src/foodcomp/research_alignment.py','src/foodcomp/research_alignment_inference.py','scripts/train_foodnutrigpt_v9_r5_mapping.py']},
            scope='Real768profiles,3CPU default updates exactly replay frozen parent; full60721-profile view checked. This is not a new trained candidate or fullCUDA trajectory replay. Query protocol is unchanged.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print({k:receipt[k] for k in ['status','assigned_profiles','changed_profiles','original_visible_cells','partial_training_visible_cells']})


if __name__=='__main__':main()
