"""Check the completed R2 control's frozen MLP/loss against current zero-mixture code."""
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


def load_snapshot(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def main():
    out=ROOT/"reports/v9_r3_control_audit_v1"
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    control=ROOT/"output/v9_r2/mlp60_mae_width512"
    old_neural=load_snapshot("r3_old_neural",control/"code_snapshot/research_neural.py")
    old_loss=load_snapshot("foodcomp.r3_old_loss",control/"code_snapshot/research_r1.py")
    data=ResearchData(ROOT/"data/processed"/VERSION);text,cache=prepare_names(data,ROOT)
    panel=FamilyPanel(data,text,ROOT/"data/processed"/PANEL_VERSION,"cpu")
    torch.manual_seed(20260922);old,c1=old_neural.make_model(data,text.shape[1],"mlp",mlp_width=512)
    torch.manual_seed(20260922);new,c2=make_model(data,text.shape[1],"mlp",mlp_width=512)
    for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
    optimizers=[torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for m in [old,new]]
    rows=np.random.default_rng(20260923).choice(len(panel.rows),768,replace=False)
    losses=[]
    for start in range(0,len(rows),256):
        batch=panel.batch(rows[start:start+256]);masked=remove_numeric_context(batch,name_only_tasks(256,0.,20260922,1))
        pair=[]
        for model,config,opt,fn,b in [(old,c1,optimizers[0],old_loss.model_loss,batch),(new,c2,optimizers[1],model_loss,masked)]:
            model.train();opt.zero_grad(set_to_none=True);loss=fn(model,b,"mlp",config,"mae");loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();pair.append(float(loss.detach()))
        np.testing.assert_array_equal(pair[0],pair[1]);losses.append(pair)
        for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
    write_json(out/"verification.json",{"completed_control":str(control),"control_checkpoint_sha256":digest(control/"best_model.pt"),
        "frozen_neural_code_sha256":digest(control/"code_snapshot/research_neural.py"),
        "frozen_loss_code_sha256":digest(control/"code_snapshot/research_r1.py"),"data_hash":digest(data.root/"manifest.json"),
        "name_cache_hash":digest(cache/"manifest.json"),"training_tasks_checked":len(rows),"optimizer_steps_checked":3,
        "identical_initialization_and_parameters_after_each_step":True,"paired_losses":losses,"complete_test_opened":False,
        "scope":"Default zero-mixture compatibility audit on 768 real train tasks and three CPU steps; not a claim of an independently rerun full 60-epoch trajectory."})
    print("Frozen control initialization, three real-task losses and optimizer updates match exactly.")


if __name__=="__main__":main()
