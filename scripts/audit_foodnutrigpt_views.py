"""Real-training-input isolation, matched control and loading audit for R4 views."""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_text import prepare_names
from foodcomp.research_r1 import FamilyPanel, PANEL_VERSION, fingerprint_array
from foodcomp.research_neural import make_model
from foodcomp.research_views import extra_view_masks, subset_view, representation_distance, two_view_loss
from foodcomp.research_inference import NutritionModel
from audit_foodnutrigpt_v9_r3_control import load_snapshot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True);args=p.parse_args()
    if args.output_dir.exists():raise FileExistsError(args.output_dir)
    out=args.output_dir;out.mkdir(parents=True);torch.set_num_threads(1)
    result={"status":"incomplete","complete_test_opened":False,"script_sha256":digest(Path(__file__))}
    try:
        data=ResearchData(ROOT/"data/processed"/VERSION);text,cache=prepare_names(data,ROOT)
        panel=FamilyPanel(data,text,ROOT/"data/processed"/PANEL_VERSION,"cpu")
        parent=ROOT/"output/v9_r2/mlp60_mae_width512"
        old_neural=load_snapshot("views_old_neural",parent/"code_snapshot/research_neural.py")
        old_loss=load_snapshot("foodcomp.views_old_loss",parent/"code_snapshot/research_r1.py")
        torch.manual_seed(20260922);old,old_config=old_neural.make_model(data,text.shape[1],"mlp",mlp_width=512)
        rng=torch.get_rng_state()
        torch.manual_seed(20260922);model,config=make_model(data,text.shape[1],"mlp",mlp_width=512)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        for key,v in old.state_dict().items():torch.testing.assert_close(v,model.state_dict()[key],rtol=0,atol=0)
        tasks=np.random.default_rng(20260923).choice(len(panel.rows),768,replace=False)
        if not np.isin(panel.rows[tasks],data.train).all():raise ValueError("Nontraining audit task.")
        masks=extra_view_masks(len(panel.rows),len(data.axes),.3,20260922,1)
        opt_old=torch.optim.AdamW(old.parameters(),lr=.001,weight_decay=.0001)
        opt_new=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        losses=[]
        for start in range(0,len(tasks),256):
            ids=tasks[start:start+256];b=panel.batch(ids);second=subset_view(b,masks[ids])
            all_family=panel.masks[panel.family_ids[ids]]
            if not second["masked"][all_family].all():raise AssertionError("Target-family leakage.")
            if not second["masked"][b["masked"]].all():raise AssertionError("B revealed unavailable input.")
            opt_old.zero_grad(set_to_none=True);opt_new.zero_grad(set_to_none=True)
            a=old_loss.model_loss(old,b,"mlp",old_config,"mae")
            c,_=two_view_loss(model,b,masks[ids],0.)
            torch.testing.assert_close(a,c,rtol=0,atol=0)
            a.backward();c.backward()
            for x,y in zip(old.parameters(),model.parameters()):torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
            torch.nn.utils.clip_grad_norm_(old.parameters(),1.);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt_old.step();opt_new.step()
            for key,v in old.state_dict().items():torch.testing.assert_close(v,model.state_dict()[key],rtol=0,atol=0)
            torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
            losses.append([float(a.detach()),float(c.detach())])
        b=panel.batch(tasks[:64]);drop=masks[tasks[:64]];other=copy.deepcopy(b)
        other["value"][b["masked"]]+=999;other["source"]+=99
        other["target"]=~other["target"];other["positive"]=~other["positive"]
        model.eval()
        with torch.no_grad():
            for view in [None,drop]:
                a=b if view is None else subset_view(b,view)
                c=other if view is None else subset_view(other,view)
                torch.testing.assert_close(model.encode(a),model.encode(c),rtol=0,atol=0)
                torch.testing.assert_close(model(a)["amount_normalized"],model(c)["amount_normalized"],rtol=0,atol=0)
        model.zero_grad(set_to_none=True)
        loss,parts=two_view_loss(model,b,drop,.1);loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise FloatingPointError("Nonfinite consistency-model gradients.")
        if not parts["consistency"]>0:raise AssertionError("Vacuous consistency fixture.")
        count=sum(p.numel() for p in model.parameters())
        if count!=667900:raise AssertionError("Unexpected parameter count.")
        fixture=out/"three_step_loader_fixture_not_candidate.pt"
        saved={"model_state":model.state_dict(),"kind":"mlp","config":asdict(config),"text_dim":text.shape[1],
            "data_root":str(data.root),"view":data.view,"name_cache":str(cache),
            "data_hash":digest(data.root/"manifest.json"),"name_cache_hash":digest(cache/"manifest.json"),
            "args":{"mlp_width":512,"consistency_weight":.1,"view_drop_probability":.3}}
        torch.save(saved,fixture);loaded=NutritionModel(fixture,device="cpu")
        with torch.no_grad():torch.testing.assert_close(model(b)["amount_normalized"],loaded.model(b)["amount_normalized"],rtol=0,atol=0)
        axes=data.axes.loc[data.axes.loss_eligible & data.axes.loss_group.eq("nutrition"),"axis_index"].to_numpy()
        names=data.profiles.iloc[data.train].original_name.drop_duplicates().iloc[:3].tolist()
        context={int(axes[2]):0.}
        for observed in [{},context]:
            predictions=loaded.predict(names[0],observed,axes[:2])
            if len(predictions)!=2 or not np.isfinite(list(predictions.values())).all():raise AssertionError("Arbitrary unobserved-axis prediction failed.")
        np.testing.assert_array_equal(loaded.encode(names[0],context,"nutrition"),loaded.encode(names[1],context,"nutrition"))
        ranking=loaded.retrieve_names(context,names,top_k=2)
        if len(ranking)!=2 or not all(np.isfinite(x["score"]) for x in ranking):raise AssertionError("Retrieval API failure.")
        result.update(status="complete",training_tasks_checked=768,optimizer_steps=3,parameter_count=count,
            initial_state_rng_losses_gradients_updates_equal_to_frozen_parent=True,
            full_target_family_hidden_in_both_views=True,hidden_values_targets_and_sources_do_not_change_encoding=True,
            positive_consistency_loss_and_finite_gradients=True,save_reload_bitwise_equal=True,three_api_methods_checked=True,
            paired_cpu_losses=losses,probe_consistency=float(parts["consistency"]),
            task_ids_sha256=fingerprint_array(tasks),mask_sha256=fingerprint_array(masks),
            data_sha256=saved["data_hash"],name_cache_sha256=saved["name_cache_hash"],fixture_sha256=digest(fixture),
            frozen_parent_neural_sha256=digest(parent/"code_snapshot/research_neural.py"),
            frozen_parent_loss_sha256=digest(parent/"code_snapshot/research_r1.py"),
            code_sha256={file:digest(ROOT/file) for file in ["src/foodcomp/research_views.py","src/foodcomp/research_neural.py","src/foodcomp/research_inference.py","src/foodcomp/research_r1.py","scripts/train_foodnutrigpt_v9_r4_views.py"]},
            scope="Real train-only768-task/three-stepCPU functional compatibility, not full CUDA trajectory or a trained experimental model. Zero-weight control matches original; nonzero-weight gradients are not asserted equal. Fixture is only for loading/APIs.")
    except Exception as error:
        result.update(status="failed",error_type=type(error).__name__,error=str(error));write_json(out/"verification.json",result);raise
    write_json(out/"verification.json",result)
    print({k:result[k] for k in ["status","parameter_count","probe_consistency"]})


if __name__=="__main__":main()
