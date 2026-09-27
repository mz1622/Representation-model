"""Real768-profile,3-step default replay plus finite MAE functional fixture."""
import argparse
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_text import prepare_names
from foodcomp.research_neural import make_model,training_batch
from foodcomp.research_forward_loss import forward_loss
from foodcomp.research_alignment import state_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True);torch.set_num_threads(4)
    receipt={'status':'incomplete','complete_test_opened':False,'script_sha256':digest(Path(__file__))}
    try:
        frozen=ROOT/'output/v9_r5/name_mlp60_legacy_loss/code_snapshot/research_neural.py'
        spec=importlib.util.spec_from_file_location('frozen_forward_neural',frozen);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        data=ResearchData(ROOT/'data/processed'/VERSION);text,_=prepare_names(data,ROOT)
        rows=data.train[data.observed[data.train][:,data.targets].any(1)][:768]
        torch.manual_seed(20260922);parent,config=old.make_model(data,text.shape[1],'name_mlp')
        torch.manual_seed(20260922);model,_=make_model(data,text.shape[1],'name_mlp')
        assert state_fingerprint(parent)==state_fingerprint(model);initial=state_fingerprint(model)
        optimizers=[torch.optim.AdamW(x.parameters(),lr=.001,weight_decay=.0001) for x in [parent,model]];pairs=[]
        for start in range(0,768,256):
            b=training_batch(data,text,rows[start:start+256],'cpu',np.random.default_rng(22),'name_mlp')
            for opt in optimizers:opt.zero_grad(set_to_none=True)
            x=old.loss(parent,b,'name_mlp',config);y=forward_loss(model,b,config);assert torch.equal(x,y)
            x.backward();y.backward()
            for a,z in zip(parent.parameters(),model.parameters()):assert torch.equal(a.grad,z.grad)
            for net,opt in zip([parent,model],optimizers):torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            assert state_fingerprint(parent)==state_fingerprint(model);pairs.append([float(x.detach()),float(y.detach())])
        model.zero_grad(set_to_none=True);value=forward_loss(model,b,config,'mae');value.backward()
        assert all(torch.isfinite(x.grad).all() for x in model.parameters())
        hidden=copy.deepcopy(b);hidden['value'][~hidden['target']]=123456.
        assert torch.equal(value,forward_loss(model,hidden,config,'mae'))
        torch.save(model.state_dict(),args.output_dir/'fixture.pt');clone,_=make_model(data,text.shape[1],'name_mlp')
        clone.load_state_dict(torch.load(args.output_dir/'fixture.pt',weights_only=True));assert torch.equal(model(b)['amount_normalized'],clone(b)['amount_normalized'])
        files=['scripts/train_foodnutrigpt_v9_r5_forward.py','src/foodcomp/research_forward_loss.py','src/foodcomp/research_neural.py']
        receipt.update(status='complete',tasks=768,optimizer_steps=3,initial_state_sha256=initial,
            default_losses_gradients_postclip_updates_exact=True,paired_losses=pairs,mae_finite_gradients=True,
            mae_hidden_target_mutation_exact=True,save_reload_prediction_exact=True,
            code_sha256={f:digest(ROOT/f) for f in files},frozen_parent_sha256=digest(frozen),
            scope='Real3CPUstep default compatibility, not a new full training candidate; same initialization and labels. Full parent60 trajectory already archived.')
    except Exception as error:
        receipt.update(status='failed',error_type=type(error).__name__,error=str(error));write_json(args.output_dir/'verification.json',receipt);raise
    write_json(args.output_dir/'verification.json',receipt);print(receipt)


if __name__=='__main__':main()
