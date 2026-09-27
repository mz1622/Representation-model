"""Train-only real-task default compatibility and context/label isolation gate."""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_text import prepare_names
from foodcomp.research_r1 import FamilyPanel,PANEL_VERSION,model_loss,fingerprint_array
from foodcomp.research_neural import make_model
from foodcomp.research_context_dropout import context_dropout_masks,drop_context
from foodcomp.research_inference import NutritionModel
from audit_foodnutrigpt_v9_r3_control import load_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(1)
    receipt={'status':'incomplete','complete_test_opened':False,'script_sha256':digest(Path(__file__))}
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION);text,cache=prepare_names(data,ROOT)
        panel=FamilyPanel(data,text,ROOT/'data/processed'/PANEL_VERSION,'cpu');parent=ROOT/'output/v9_r2/mlp60_mae_width512'
        old_neural=load_snapshot('context_old_neural',parent/'code_snapshot/research_neural.py')
        old_loss=load_snapshot('foodcomp.context_old_loss',parent/'code_snapshot/research_r1.py')
        torch.manual_seed(20260922);old,old_config=old_neural.make_model(data,text.shape[1],'mlp',mlp_width=512);rng=torch.get_rng_state()
        torch.manual_seed(20260922);model,config=make_model(data,text.shape[1],'mlp',mlp_width=512)
        assert torch.equal(rng,torch.get_rng_state())
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in old.state_dict().items())
        opts=[torch.optim.AdamW(x.parameters(),lr=.001,weight_decay=.0001) for x in [old,model]]
        tasks=np.random.default_rng(20260923).choice(len(panel.rows),768,replace=False);losses=[]
        none,_=context_dropout_masks(len(panel.rows),len(data.axes),'none',20260922,1)
        for start in range(0,768,256):
            b=panel.batch(tasks[start:start+256]);changed=drop_context(b,none);assert changed is b
            for opt in opts:opt.zero_grad(set_to_none=True)
            left=old_loss.model_loss(old,b,'mlp',old_config,'mae');right=model_loss(model,changed,'mlp',config,'mae')
            assert torch.equal(left,right);left.backward();right.backward()
            assert all(torch.equal(a.grad,b.grad) for a,b in zip(old.parameters(),model.parameters()))
            for net,opt in zip([old,model],opts):torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            assert all(torch.equal(v,model.state_dict()[k]) for k,v in old.state_dict().items())
            losses.append(float(left.detach()))
        assert torch.equal(rng,torch.get_rng_state())
        extra,ids=context_dropout_masks(len(panel.rows),len(data.axes),'mix_30_60_90',20260922,1)
        fixed,_=context_dropout_masks(len(panel.rows),len(data.axes),'fixed_30',20260922,1);assert not (fixed&~extra).any()
        b=panel.batch(tasks);changed=drop_context(b,extra[tasks]);assert not ((~changed['masked'])&b['masked']).any()
        assert changed['masked'][panel.masks[panel.family_ids[tasks]]].all()
        for key in b:
            if key!='masked':assert changed[key] is b[key]
        model.eval();output=model(changed)['amount_normalized'];assert torch.isfinite(output).all()
        altered=copy.deepcopy(changed);altered['value'][changed['masked']]+=999;altered['target']=~changed['target'];altered['source']+=99;altered['positive']=~changed['positive']
        assert torch.equal(output,model(altered)['amount_normalized'])
        missing=copy.deepcopy(changed);missing['value'][~changed['target']]=1234
        # Loss output fixed: altering unobserved/non-target labels cannot add supervision.
        from foodcomp.research_r1 import panel_loss
        assert torch.equal(panel_loss({'amount_normalized':output},changed,objective='mae'),panel_loss({'amount_normalized':output},missing,objective='mae'))
        for bad in [float('nan'),float('inf')]:
            invalid=output.detach().clone();invalid[0,0]=bad
            try:panel_loss({'amount_normalized':invalid},changed,objective='mae')
            except FloatingPointError:pass
            else:raise AssertionError('Nonfinite output did not fail.')
        fixture=args.output_dir/'default_three_step_fixture_not_candidate.pt'
        saved={'model_state':model.state_dict(),'kind':'mlp','config':asdict(config),'text_dim':text.shape[1],
            'data_root':str(data.root),'view':data.view,'name_cache':str(cache),'data_hash':digest(data.root/'manifest.json'),
            'name_cache_hash':digest(cache/'manifest.json'),'args':{'mlp_width':512,'context_dropout':'mix_30_60_90'}}
        torch.save(saved,fixture);loaded=NutritionModel(fixture,device='cpu')
        assert torch.equal(output,loaded.model(changed)['amount_normalized'])
        axes=data.axes.loc[data.axes.loss_eligible&data.axes.loss_group.eq('nutrition'),'axis_index'].to_numpy()
        name=data.profiles.iloc[data.train[0]].original_name
        for context in [{},{int(axes[2]):0.}]:assert np.isfinite(list(loaded.predict(name,context,axes[:2]).values())).all()
        receipt.update(status='complete',actual_train_tasks=768,default_steps=3,default_rng_initialization_loss_gradients_updates_exact=True,
            target_family_hidden=True,only_visibility_changes=True,all_task_fixed30_masks_nested_in_mixture=True,
            hidden_values_labels_sources_do_not_change_predictions=True,non_target_values_do_not_enter_loss=True,
            nan_inf_outputs_fail=True,save_reload_exact=True,caller_unobserved_axis_prediction=True,
            parameter_count=sum(p.numel() for p in model.parameters()),paired_default_losses=losses,
            train_tasks_sha256=fingerprint_array(tasks),full_epoch1_mask_sha256=fingerprint_array(extra),full_epoch1_rates_sha256=fingerprint_array(ids),
            data_sha256=saved['data_hash'],name_cache_sha256=saved['name_cache_hash'],fixture_sha256=digest(fixture),
            scope='768 actual train tasks/3 default CPU updates against frozen R2 source; no full GPU training replay. New dropout mask never changes supervision.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt['status'],receipt['parameter_count'])

if __name__=='__main__':main()
