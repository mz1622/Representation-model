"""Check a completed shared-head control's frozen MLP/loss against current code."""
import argparse
import json
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_text import prepare_names
from foodcomp.research_r1 import FamilyPanel,PANEL_VERSION,model_loss
from foodcomp.research_neural import make_model
from foodcomp.research_task_mix import name_only_tasks,remove_numeric_context
from foodcomp.research_inference import NutritionModel


def load_snapshot(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control",type=Path,default=ROOT/"output/v9_r2/mlp60_mae_width512")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"reports/v9_r3_control_audit_v1")
    parser.add_argument("--check-separate-loader",action="store_true")
    args=parser.parse_args();out=args.output_dir
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    control=args.control
    receipt=json.loads((control/"run_manifest.json").read_text())
    if receipt["status"]!="complete" or receipt["args"]["kind"]!="mlp" or receipt["args"].get("mlp_task_heads","shared")!="shared":
        raise ValueError("Completed shared-head MLP control required.")
    if receipt["args"].get("mlp_width")!=512 or receipt["args"].get("mlp_normalization","layer_norm")!="layer_norm" or receipt["args"]["objective"]!="mae":
        raise ValueError("This compatibility audit is registered for the512/LayerNorm/MAE controls.")
    probability=receipt["args"].get("name_only_probability",0.)
    old_neural=load_snapshot("r3_old_neural",control/"code_snapshot/research_neural.py")
    old_loss=load_snapshot("foodcomp.r3_old_loss",control/"code_snapshot/research_r1.py")
    data=ResearchData(ROOT/"data/processed"/VERSION);text,cache=prepare_names(data,ROOT)
    panel=FamilyPanel(data,text,ROOT/"data/processed"/PANEL_VERSION,"cpu")
    torch.manual_seed(20260922);old,c1=old_neural.make_model(data,text.shape[1],"mlp",mlp_width=512);rng=torch.get_rng_state()
    torch.manual_seed(20260922);new,c2=make_model(data,text.shape[1],"mlp",mlp_width=512)
    torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
    for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in [old,new]]
    rows=np.random.default_rng(20260923).choice(len(panel.rows),768,replace=False)
    assignment=name_only_tasks(len(panel.rows),probability,20260922,1)
    losses=[]
    for start in range(0,len(rows),256):
        chosen=rows[start:start+256];batch=panel.batch(chosen);masked=remove_numeric_context(batch,assignment[chosen])
        pair=[]
        for model,config,opt,fn,b in [(old,c1,optimizers[0],old_loss.model_loss,masked),(new,c2,optimizers[1],model_loss,masked)]:
            model.train();opt.zero_grad(set_to_none=True);loss=fn(model,b,"mlp",config,"mae");loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();pair.append(float(loss.detach()))
        np.testing.assert_array_equal(pair[0],pair[1]);losses.append(pair)
        for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
    loader=False
    if args.check_separate_loader:
        # Fixture only: clone a completed shared head into both branches to test loader/API parity.
        saved=torch.load(control/"best_model.pt",map_location="cpu",weights_only=True)
        saved["args"]["mlp_task_heads"]="separate"
        for key in ["weight","bias"]:saved["model_state"]["name_head."+key]=saved["model_state"]["head."+key].clone()
        fixture=out/"loader_fixture_not_trained.pt";torch.save(saved,fixture)
        shared_wrapper=NutritionModel(control/"best_model.pt",device="cpu")
        separate_wrapper=NutritionModel(fixture,device="cpu")
        original=panel.batch(rows[:16])
        with torch.no_grad():
            for b in [original,remove_numeric_context(original,np.ones(16,bool))]:
                torch.testing.assert_close(shared_wrapper.model(b)["amount_normalized"],separate_wrapper.model(b)["amount_normalized"],rtol=0,atol=0)
        name=data.profiles.iloc[data.train[0]].original_name
        for observed in [{},{int(data.targets[2]):0.}]:
            assert shared_wrapper.predict(name,observed,data.targets[:2])==separate_wrapper.predict(name,observed,data.targets[:2])
        loader=True
    write_json(out/"verification.json",{"completed_control":str(control),"control_checkpoint_sha256":digest(control/"best_model.pt"),
        "frozen_neural_code_sha256":digest(control/"code_snapshot/research_neural.py"),
        "frozen_loss_code_sha256":digest(control/"code_snapshot/research_r1.py"),"data_hash":digest(data.root/"manifest.json"),
        "name_cache_hash":digest(cache/"manifest.json"),"training_tasks_checked":len(rows),"optimizer_steps_checked":3,
        "name_only_probability":probability,"name_only_assigned_tasks":int(assignment[rows].sum()),
        "identical_initialization_and_parameters_after_each_step":True,"paired_losses":losses,"complete_test_opened":False,
        "separate_head_loader_and_predict_api_checked":loader,"code_sha256":digest(Path(__file__)),
        "scope":"Shared-head compatibility audit on768 real train tasks and three CPU steps; not an independently rerun60-epoch trajectory. Optional separate-head fixture duplicates a trained shared head only to verify loading, routing and inference; it is not an experimental candidate."})
    print("Frozen control initialization, three real-task losses and optimizer updates match exactly.")


if __name__=="__main__":main()
