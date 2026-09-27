"""Train-only default compatibility, loss partition and inference isolation audit."""
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
from foodcomp.research_r1 import FamilyPanel,PANEL_VERSION,model_loss,panel_loss,fingerprint_array
from foodcomp.research_neural import make_model
from foodcomp.research_auxiliary import metabolome_axis_scale
from foodcomp.research_inference import NutritionModel
from audit_foodnutrigpt_v9_r3_control import load_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();out=args.output_dir
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    receipt={'status':'incomplete','complete_test_opened':False,'script_sha256':digest(Path(__file__))}
    try:
        data=ResearchData(ROOT/'data/processed'/VERSION);text,cache=prepare_names(data,ROOT)
        panel=FamilyPanel(data,text,ROOT/'data/processed'/PANEL_VERSION,'cpu')
        parent=ROOT/'output/v9_r2/mlp60_mae_width512'
        old_neural=load_snapshot('aux_old_neural',parent/'code_snapshot/research_neural.py')
        old_loss=load_snapshot('foodcomp.aux_old_loss',parent/'code_snapshot/research_r1.py')
        torch.manual_seed(20260922);old,old_config=old_neural.make_model(data,text.shape[1],'mlp',mlp_width=512)
        rng=torch.get_rng_state()
        torch.manual_seed(20260922);model,config=make_model(data,text.shape[1],'mlp',mlp_width=512)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key,v in old.state_dict().items():torch.testing.assert_close(v,model.state_dict()[key],rtol=0,atol=0)
        default=metabolome_axis_scale(data,1.,'cpu');assert default is None
        half=metabolome_axis_scale(data,.5,'cpu')
        tasks=np.random.default_rng(20260923).choice(len(panel.rows),768,replace=False)
        assert np.isin(panel.rows[tasks],data.train).all()
        opts=[torch.optim.AdamW(x.parameters(),lr=.001,weight_decay=.0001) for x in [old,model]]
        losses=[]
        for start in range(0,768,256):
            ids=tasks[start:start+256];b=panel.batch(ids)
            assert b['masked'][panel.masks[panel.family_ids[ids]]].all()
            for opt in opts:opt.zero_grad(set_to_none=True)
            a=old_loss.model_loss(old,b,'mlp',old_config,'mae')
            c=model_loss(model,b,'mlp',config,'mae',axis_loss_scale=default)
            torch.testing.assert_close(a,c,rtol=0,atol=0);a.backward();c.backward()
            for x,y in zip(old.parameters(),model.parameters()):torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
            for x,opt in zip([old,model],opts):torch.nn.utils.clip_grad_norm_(x.parameters(),1.);opt.step()
            for key,v in old.state_dict().items():torch.testing.assert_close(v,model.state_dict()[key],rtol=0,atol=0)
            torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
            losses.append([float(a.detach()),float(c.detach())])
        b=panel.batch(tasks[:256]);model.eval();outputs=model(b)
        parameters=list(model.parameters());parts=[];gradients=[]
        for group in ['nutrition','food_metabolome']:
            masked={**b,'target':b['target']&torch.as_tensor(data.axes.loss_group.eq(group).to_numpy())}
            value=panel_loss(outputs,masked,objective='mae')
            assert value>0
            parts.append(value);gradients.append(torch.autograd.grad(value,parameters,retain_graph=True))
        weighted=panel_loss(outputs,b,objective='mae',axis_loss_scale=half)
        actual=torch.autograd.grad(weighted,parameters)
        torch.testing.assert_close(weighted,parts[0]+.5*parts[1],rtol=2e-6,atol=1e-7)
        for a,n,m in zip(actual,*gradients):torch.testing.assert_close(a,n+.5*m,rtol=1e-4,atol=2e-7)
        altered=copy.deepcopy(b);altered['value'][b['masked']]+=999;altered['source']+=99
        altered['target']=~altered['target'];altered['positive']=~altered['positive']
        with torch.no_grad():
            torch.testing.assert_close(model(b)['amount_normalized'],model(altered)['amount_normalized'],rtol=0,atol=0)
        model.zero_grad(set_to_none=True)
        weighted=model_loss(model,b,'mlp',config,'mae',axis_loss_scale=half);weighted.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
        count=sum(p.numel() for p in parameters);assert count==667900
        fixture=out/'three_step_default_fixture_not_candidate.pt'
        saved={'model_state':model.state_dict(),'kind':'mlp','config':asdict(config),'text_dim':text.shape[1],
               'data_root':str(data.root),'view':data.view,'name_cache':str(cache),
               'data_hash':digest(data.root/'manifest.json'),'name_cache_hash':digest(cache/'manifest.json'),
               'args':{'mlp_width':512,'metabolome_loss_weight':.5}}
        torch.save(saved,fixture);loaded=NutritionModel(fixture,device='cpu')
        with torch.no_grad():torch.testing.assert_close(model(b)['amount_normalized'],loaded.model(b)['amount_normalized'],rtol=0,atol=0)
        axes=data.axes.loc[data.axes.loss_eligible & data.axes.loss_group.eq('nutrition'),'axis_index'].to_numpy()
        names=data.profiles.iloc[data.train].original_name.drop_duplicates().iloc[:3].tolist()
        context={int(axes[2]):0.}
        for observed in [{},context]:
            predictions=loaded.predict(names[0],observed,axes[:2]);assert len(predictions)==2 and np.isfinite(list(predictions.values())).all()
        np.testing.assert_array_equal(loaded.encode(names[0],context,'nutrition'),loaded.encode(names[1],context,'nutrition'))
        ranking=loaded.retrieve_names(context,names,top_k=2);assert len(ranking)==2 and all(np.isfinite(x['score']) for x in ranking)
        receipt.update(status='complete',training_tasks_checked=768,optimizer_steps=3,parameter_count=count,
            default_initial_rng_parameters_loss_gradients_updates_exact=True,all_target_families_hidden=True,
            weighted_loss_and_parameter_gradients_equal_nutrition_plus_half_metabolome=True,
            original_187_denominator_retained=True,all187_positive_supervision=True,
            hidden_values_labels_sources_do_not_change_predictions=True,finite_weighted_gradients=True,
            save_reload_bitwise=True,three_api_methods_checked=True,paired_cpu_losses=losses,
            partition_losses=[float(x.detach()) for x in parts],weighted_loss=float(weighted.detach()),
            task_ids_sha256=fingerprint_array(tasks),axis_scale_sha256=fingerprint_array(half.numpy()),
            data_sha256=saved['data_hash'],name_cache_sha256=saved['name_cache_hash'],fixture_sha256=digest(fixture),
            frozen_parent_neural_sha256=digest(parent/'code_snapshot/research_neural.py'),
            frozen_parent_loss_sha256=digest(parent/'code_snapshot/research_r1.py'),
            code_sha256={file:digest(ROOT/file) for file in ['src/foodcomp/research_auxiliary.py','src/foodcomp/research_r1.py','src/foodcomp/research_neural.py','src/foodcomp/research_inference.py','scripts/train_foodnutrigpt_v9_r1.py','scripts/train_foodnutrigpt_v9_r4_auxiliary.py']},
            scope='768 actual train tasks/three default CPU steps, not full CUDA replay. Half-weight partition tests parameter gradients before clipping; effective optimizer trajectories need not match. Fixture is not a candidate.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(out/'verification.json',receipt);raise
    write_json(out/'verification.json',receipt);print({k:receipt[k] for k in ['status','parameter_count','partition_losses','weighted_loss']})


if __name__=='__main__':main()
