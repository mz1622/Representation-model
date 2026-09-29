"""Training-row checks of the PDF capacity group; no candidate validation scoring."""
import argparse
import ast
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData,VERSION,digest,write_json
from foodcomp.research_r1 import FamilyPanel
from foodcomp.research_neural import DirectSourceCalibratedModel
from foodcomp.research_alignment import state_fingerprint
from foodcomp.research_completion_input import completion_view
from foodcomp.research_transformer_r9 import frozen_inputs,make_transformer,transformer_loss,TransformerNutritionModel
from foodcomp.research_transformer_r9_capacity256 import load_method,verify_bindings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    local=ROOT/'data/local/research_diagnostics/v9_r9_capacity256_functional_v1'
    for path in [args.output_dir,local]:
        if path.exists():raise FileExistsError(path)
    frozen,freeze_path=frozen_inputs(ROOT)
    config_path=ROOT/'experiments/foodnutrigpt_v9_research/r9/capacity256_v1/config.json'
    plan,parent,spec,bindings=load_method(ROOT,config_path,frozen,freeze_path)
    def loop(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        found=[node for node in ast.walk(tree) if isinstance(node,ast.For)
               and isinstance(node.target,ast.Name) and node.target.id=='offset']
        assert len(found)==1
        return ast.dump(found[0],include_attributes=False)
    assert loop(ROOT/'scripts/train_foodnutrigpt_v9_r9_transformer.py')==loop(ROOT/'scripts/train_foodnutrigpt_v9_r9_capacity256.py')
    torch.set_num_threads(4)
    data=ResearchData(ROOT/'data/processed'/VERSION)
    text,cache,panel_root=completion_view(data,ROOT,32)
    assert text.shape[1]==128 and (text[:,32:]==0).all()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    panel=FamilyPanel(data,text,panel_root,device)
    assert data.profiles.iloc[panel.rows[:32]].partition.eq('train').all()
    torch.manual_seed(spec['seed'])
    control,parent_config=make_transformer(data,128,parent['spec'])
    assert state_fingerprint(control)==parent['initial_state_sha256']
    # Explicit configuration replacement is an independent check on the factory mapping.
    reference_config=replace(parent_config,d_model=256,n_heads=8,feedforward_dim=1024)
    sources=np.unique(data.profiles.iloc[data.train].source_index)
    torch.manual_seed(spec['seed'])
    reference=DirectSourceCalibratedModel(128,len(data.axes),int(data.profiles.source_index.max())+1,sources,reference_config)
    reference_rng=torch.get_rng_state().clone()
    torch.manual_seed(spec['seed'])
    model,config=make_transformer(data,128,spec)
    assert asdict(config)==asdict(reference_config)
    assert {key for key in asdict(config) if asdict(config)[key]!=asdict(parent_config)[key]}=={'d_model','n_heads','feedforward_dim'}
    torch.testing.assert_close(reference_rng,torch.get_rng_state(),atol=0,rtol=0)
    for key,value in reference.state_dict().items():
        torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)
    initial_fingerprint=state_fingerprint(model)
    assert initial_fingerprint!=parent['initial_state_sha256']
    parameters=sum(p.numel() for p in model.parameters())
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert parameters>parent['parameter_count'] and trainable>parent['trainable_parameter_count']
    reference.to(device).eval();model.to(device).eval()
    batch=panel.batch(np.arange(32))
    with torch.no_grad():
        before=model(batch)['amount_normalized']
        torch.testing.assert_close(reference(batch)['amount_normalized'],before,atol=0,rtol=0)
        torch.testing.assert_close(transformer_loss(reference,batch,reference_config,'mae'),transformer_loss(model,batch,config,'mae'),atol=0,rtol=0)
        changed={key:value.clone() if isinstance(value,torch.Tensor) else value for key,value in batch.items()}
        changed['value'][batch['masked']]=123456
        changed['target']=~changed['target'];changed['positive']=~changed['positive'];changed['source'][:]=0
        torch.testing.assert_close(model(changed)['amount_normalized'],before,atol=0,rtol=0)
    del reference,control
    args.output_dir.mkdir(parents=True);local.mkdir(parents=True)
    checkpoint={'version':'V9-R9','kind':'transformer_direct','model_state':model.state_dict(),
        'spec':spec,'config':asdict(config),'text_dim':128,'seed':spec['seed'],
        'data_root':data.root.relative_to(ROOT).as_posix(),'view':data.view,
        'name_cache':cache.relative_to(ROOT).as_posix(),'data_hash':frozen['data_hash'],
        'name_cache_hash':frozen['name_cache_hash']}
    initial_path=local/'functional_checkpoint.pt';torch.save(checkpoint,initial_path)
    loaded=TransformerNutritionModel(initial_path,ROOT,str(device))
    with torch.no_grad():torch.testing.assert_close(loaded.model(batch)['amount_normalized'],before,atol=0,rtol=0)
    row=int(data.train[0]);unobserved=data.targets[~data.observed[row,data.targets]][:3]
    assert len(unobserved)
    output=loaded.predict(str(data.profiles.original_name.iloc[row]),{},unobserved.tolist())
    assert len(output)==len(unobserved) and np.isfinite(list(output.values())).all()
    names=data.profiles.original_name.iloc[data.train[:8]].astype(str).tolist()
    raw=loaded.candidate_profiles(names,target_axes=data.targets)
    assert raw.shape==(8,187) and np.isfinite(raw).all()
    first=float(transformer_loss(model,batch,config,'mae').detach())
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    model.train()
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss=transformer_loss(model,batch,config,'mae');loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),1.)):
            raise FloatingPointError('Nonfinite functional gradient')
        optimizer.step()
    model.eval();last=float(transformer_loss(model,batch,config,'mae').detach())
    assert last<first and torch.count_nonzero(model.source_amount_residual.weight)>0
    learned_path=local/'learned_functional_checkpoint.pt'
    checkpoint['model_state']=model.state_dict();torch.save(checkpoint,learned_path)
    learned=TransformerNutritionModel(learned_path,ROOT,str(device))
    with torch.no_grad():
        torch.testing.assert_close(learned.model(batch)['amount_normalized'],model(batch)['amount_normalized'],atol=0,rtol=0)
        changed_source=dict(batch,source=torch.zeros_like(batch['source']))
        torch.testing.assert_close(model(changed_source)['amount_normalized'],model(batch)['amount_normalized'],atol=0,rtol=0)
    frozen_inputs(ROOT);verify_bindings(ROOT,bindings)
    files=['src/foodcomp/research_transformer_r9.py','scripts/train_foodnutrigpt_v9_r9_transformer.py',
        'src/foodcomp/research_transformer_r9_capacity256.py','scripts/train_foodnutrigpt_v9_r9_capacity256.py',
        'scripts/verify_foodnutrigpt_r9_capacity256_functional.py']
    record={'status':'complete','freeze_sha256':digest(freeze_path),'plan_sha256':digest(config_path),
        'implementation_hashes':{p:digest(ROOT/p) for p in files},'initial_state_sha256':initial_fingerprint,
        'parameter_count':parameters,'trainable_parameter_count':trainable,
        'parent_initialization_reproduced_exact':True,'new_initialization_matches_explicit_reference_exact':True,
        'reference_constructor_rng_exact':True,'only_capacity_configuration_triplet_changed':True,
        'same_seed_parent_initialization_exact':False,'inner_training_loop_ast_unchanged':True,
        'hidden_label_target_positive_and_source_forward_exact':True,
        'unobserved_axis_queries_and8_train_names187_axes_finite':True,
        'initial_and_learned_public_loader_reload_exact':True,'source_calibration_table_learned_nonzero':True,
        'source_not_in_learned_base_forward':True,'small_batch_overfit':{'before':first,'after':last,'steps':50},
        'functional_checkpoint_sha256':digest(initial_path),'learned_checkpoint_sha256':digest(learned_path),
        'current_candidate_validation_evaluated':False,'data_modified':False,'baseline_refit':False,'complete_test_opened':False,
        'scope':'Training-row implementation and learnability checks only. Same seed does not match initial weights, constructor RNG or dropout streams across widths. Formal training constructs a fresh model and restores its own smoke weights/RNG.'}
    write_json(args.output_dir/'verification.json',record)
    print(json.dumps({key:record[key] for key in ['status','initial_state_sha256','parameter_count','trainable_parameter_count','small_batch_overfit']}))


if __name__=='__main__':main()
