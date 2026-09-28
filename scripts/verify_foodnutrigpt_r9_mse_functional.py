"""Training-only MSE checks with exact parent initialization and unchanged numerical loop."""
import ast
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_r1 import FamilyPanel, model_loss
from foodcomp.research_neural import make_model
from foodcomp.research_alignment import state_fingerprint
from foodcomp.research_transformer_r9_methods import load_method, verify_bindings
from foodcomp.research_completion_input import completion_view, execution_contract
from foodcomp.research_transformer_r9 import frozen_inputs, make_transformer, transformer_loss, TransformerNutritionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frozen, freeze_path = frozen_inputs(ROOT)
    config_path = ROOT / 'experiments/foodnutrigpt_v9_research/r9/mse_v1/config.json'
    plan, parent, spec, bindings = load_method(ROOT, config_path, frozen, freeze_path)
    def training_loop(path):
        tree = ast.parse(path.read_text(encoding='utf-8'))
        loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
                 and isinstance(node.target, ast.Name) and node.target.id == 'offset']
        assert len(loops) == 1
        return ast.dump(loops[0], include_attributes=False)
    assert training_loop(ROOT / 'scripts/train_foodnutrigpt_v9_r9_transformer.py') == training_loop(ROOT / 'scripts/train_foodnutrigpt_v9_r9_mse.py')
    data = ResearchData(ROOT / 'data/processed' / VERSION)
    text, cache, panel_root = completion_view(data, ROOT, 32)
    assert text.shape[1] == 128 and (text[:, 32:] == 0).all()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(4)
    panel = FamilyPanel(data, text, panel_root, device)
    torch.manual_seed(spec['seed'])
    old, old_config = make_model(data, 128, 'v9_direct')
    torch.manual_seed(spec['seed'])
    new, config = make_transformer(data, 128, spec)
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
    assert state_fingerprint(new) == parent['initial_state_sha256']
    old.to(device).eval(); new.to(device).eval()
    batch = panel.batch(np.arange(32))
    with torch.no_grad():
        before = new(batch)['amount_normalized']
        torch.testing.assert_close(old(batch)['amount_normalized'], before, rtol=0, atol=0)
        torch.testing.assert_close(transformer_loss(old, batch, old_config, 'mse'),
            transformer_loss(new, batch, config, 'mse'), rtol=0, atol=0)
        modified = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        modified['value'][batch['masked']] = 123456
        modified['target'] = ~modified['target']
        modified['source'][:] = 0
        torch.testing.assert_close(new(modified)['amount_normalized'], before, rtol=0, atol=0)
    args.output_dir.mkdir(parents=True)
    local_dir = ROOT / 'data/local/research_diagnostics/v9_r9_mse_functional_v1'
    if local_dir.exists():
        raise FileExistsError(local_dir)
    local_dir.mkdir(parents=True)
    path = local_dir / 'functional_checkpoint.pt'
    torch.save({'version': 'V9-R9', 'kind': 'transformer_direct', 'model_state': new.state_dict(),
        'spec': spec, 'config': asdict(config), 'text_dim': 128, 'seed': spec['seed'],
        'data_root': data.root.relative_to(ROOT).as_posix(), 'view': data.view,
        'name_cache': cache.relative_to(ROOT).as_posix(), 'data_hash': frozen['data_hash'],
        'name_cache_hash': frozen['name_cache_hash']}, path)
    loaded = TransformerNutritionModel(path, ROOT, str(device))
    with torch.no_grad():
        torch.testing.assert_close(loaded.model(batch)['amount_normalized'], before, rtol=0, atol=0)
    row = int(data.train[0])
    unobserved = data.targets[~data.observed[row, data.targets]][:3]
    assert len(unobserved)
    output = loaded.predict(str(data.profiles.original_name.iloc[row]), {}, unobserved.tolist())
    assert len(output) == len(unobserved) and np.isfinite(list(output.values())).all()
    names = data.profiles.original_name.iloc[data.train[:8]].astype(str).tolist()
    candidates = loaded.candidate_profiles(names, target_axes=data.targets)
    assert candidates.shape == (8, 187) and np.isfinite(candidates).all()
    first = float(transformer_loss(new, batch, config, 'mse').detach())
    opt = torch.optim.AdamW(new.parameters(), lr=.001)
    new.train()
    for _ in range(50):
        opt.zero_grad(set_to_none=True)
        loss = transformer_loss(new, batch, config, 'mse')
        loss.backward()
        if not torch.isfinite(torch.nn.utils.clip_grad_norm_(new.parameters(), 1.)):
            raise FloatingPointError('Functional overfit gradient is nonfinite.')
        opt.step()
    new.eval()
    last = float(transformer_loss(new, batch, config, 'mse').detach())
    assert last < first
    frozen_inputs(ROOT)
    verify_bindings(ROOT, bindings)
    old_contract, _ = execution_contract(ROOT)
    files = ['src/foodcomp/research_transformer_r9.py', 'scripts/train_foodnutrigpt_v9_r9_transformer.py',
             'scripts/train_foodnutrigpt_v9_r9_mse.py', 'src/foodcomp/research_transformer_r9_methods.py',
             'scripts/verify_foodnutrigpt_r9_mse_functional.py']
    write_json(args.output_dir / 'verification.json', {'status': 'complete', 'freeze_sha256': digest(freeze_path),
        'implementation_hashes': {p: digest(ROOT / p) for p in files},
        'plan_sha256': digest(config_path), 'same_seed_parent_initialization_exact': True,
        'inner_training_loop_ast_unchanged': True, 'functional_checkpoint_sha256': digest(path),
        'legacy_control_initialization_forward_and_mse_exact': True,
        'hidden_label_target_and_source_changes_forward_exact': True,
        'public_loader_reload_forward_exact': True, 'caller_requested_unobserved_axes_predicted': True,
        'train_names_all187_queries_finite': True, 'small_batch_overfit': {'before': first, 'after': last, 'steps': 50},
        'parameter_count': sum(p.numel() for p in new.parameters()),
        'trainable_parameter_count': sum(p.numel() for p in new.parameters() if p.requires_grad),
        'old_frozen18_sources_unchanged': True, 'complete_test_opened': False,
        'validation_metrics_read': False, 'scope': 'Training-row functional checks only, not performance evidence.'})
    print(json.dumps({'status': 'complete', 'overfit_before': first, 'overfit_after': last}))


if __name__ == '__main__':
    main()
