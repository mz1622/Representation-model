"""Verify default direct-V9 math against its frozen completed control after unrelated MLP extensions."""
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


def main():
    out=ROOT/"reports/v9_r2_direct_default_compatibility_v1"
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);torch.set_num_threads(1)
    control=ROOT/"output/v9_r2/v9_direct_smoothl1_20"
    source=control/"code_snapshot/research_neural.py"
    spec=importlib.util.spec_from_file_location("old_direct_v9_neural",source)
    old_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old_module;spec.loader.exec_module(old_module)
    data=ResearchData(ROOT/"data/processed"/VERSION);text,cache=prepare_names(data,ROOT)
    panel=FamilyPanel(data,text,ROOT/"data/processed"/PANEL_VERSION,"cpu")
    indices=np.random.default_rng(20260923).choice(len(panel.rows),8,replace=False)
    if not np.isin(panel.rows[indices],data.train).all():raise AssertionError("Non-training compatibility task.")
    records=[]
    for objective in ["smooth_l1","mae"]:
        torch.manual_seed(20260922);old,c1=old_module.make_model(data,text.shape[1],"v9_direct");rng=torch.get_rng_state()
        torch.manual_seed(20260922);new,c2=make_model(data,text.shape[1],"v9_direct")
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        assert {k:p.requires_grad for k,p in old.named_parameters()}=={k:p.requires_grad for k,p in new.named_parameters()}
        for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
        optimizers=[torch.optim.AdamW(m.parameters(),lr=.0001,weight_decay=.0001) for m in [old,new]]
        for step in range(2):
            batch=panel.batch(indices[step*4:(step+1)*4]);pair=[];states=[]
            step_rng=torch.get_rng_state()
            for model,config,opt in zip([old,new],[c1,c2],optimizers):
                torch.set_rng_state(step_rng);model.train();opt.zero_grad(set_to_none=True)
                value=model_loss(model,batch,"v9_direct",config,objective);value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                pair.append(float(value.detach()));states.append(torch.get_rng_state())
            np.testing.assert_array_equal(pair[0],pair[1]);torch.testing.assert_close(states[0],states[1],rtol=0,atol=0)
            for key,value in old.state_dict().items():torch.testing.assert_close(value,new.state_dict()[key],rtol=0,atol=0)
            records.append({"objective":objective,"step":step+1,"paired_losses":pair,"parameters_and_rng_match":True})
        del old,new,optimizers
    write_json(out/"verification.json",{"records":records,"initial_state_and_requires_grad_match":True,
        "training_task_indices":indices.tolist(),"frozen_neural_sha256":digest(source),
        "current_neural_sha256":digest(ROOT/"src/foodcomp/research_neural.py"),
        "loss_module_sha256":digest(ROOT/"src/foodcomp/research_r1.py"),"script_sha256":digest(Path(__file__)),
        "data_sha256":digest(data.root/"manifest.json"),"panel_sha256":digest(panel.root/"manifest.json"),
        "name_cache_sha256":digest(cache/"manifest.json"),"complete_test_opened":False,
        "scope":"Default direct-V9 model and loss compatibility on8 real training tasks, two CPU AdamW steps separately for each objective, with matched dropout RNG. Not a full CUDA trajectory replay or evidence about objective superiority."})
    print("Direct V9 initial state, losses, optimizer updates and dropout RNG match frozen control for both objectives.")


if __name__=="__main__":main()
