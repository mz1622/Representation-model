"""Validate saved model/optimizer/RNG recovery using training rows only."""
import argparse
import ast
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_r1 import FamilyPanel
from foodcomp.research_completion_input import completion_view
from foodcomp.research_transformer_r9 import frozen_inputs, make_transformer, transformer_loss
from foodcomp.research_transformer_r9_capacity256 import load_method, verify_bindings
from foodcomp.research_transformer_r9_capacity256_recovery import load_recovery, restore, assert_state_equal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    path = ROOT/'experiments/foodnutrigpt_v9_research/r9/capacity256_v1/recovery1/config.json'
    plan, original, history, state, hashes = load_recovery(ROOT, path)
    frozen, freeze_path = frozen_inputs(ROOT)
    _, _, spec, bindings = load_method(ROOT,
        ROOT/'experiments/foodnutrigpt_v9_research/r9/capacity256_v1/config.json', frozen, freeze_path)
    assert original['spec'] == spec
    def block(file, predicate):
        tree = ast.parse((ROOT/file).read_text(encoding='utf-8'))
        nodes = [node for node in ast.walk(tree) if predicate(node)]
        assert len(nodes) == 1
        return ast.dump(nodes[0], include_attributes=False)
    source = 'scripts/train_foodnutrigpt_v9_r9_capacity256.py'
    resume = 'scripts/resume_foodnutrigpt_v9_r9_capacity256.py'
    assert block(source, lambda n: isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'offset') == block(resume, lambda n: isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == 'offset')
    assert block(source, lambda n: isinstance(n, ast.FunctionDef) and n.name == 'checkpoint') == block(resume, lambda n: isinstance(n, ast.FunctionDef) and n.name == 'checkpoint')
    # All numeric statements from order construction through the end-of-epoch print are unchanged.
    def epoch_body(file):
        code = (ROOT/file).read_text(encoding='utf-8')
        text = code[code.index('            order = '):code.index('        saved = torch.load')]
        import textwrap
        return ast.dump(ast.parse(textwrap.dedent(text)), include_attributes=False)
    assert epoch_body(source) == epoch_body(resume)
    torch.set_num_threads(4)
    if not torch.cuda.is_available():
        raise ValueError('Original GPU execution backend required')
    device = torch.device('cuda')
    data = ResearchData(ROOT/'data/processed'/VERSION)
    text, _, panel_root = completion_view(data, ROOT, 32)
    panel = FamilyPanel(data, text, panel_root, device)
    assert data.profiles.iloc[panel.rows[:32]].partition.eq('train').all()
    def build():
        model, config = make_transformer(data, 128, spec)
        assert asdict(config) == state['config']
        model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=spec['learning_rate']*.01)
        restore(model, optimizer, scheduler, state, device)
        model.eval()
        return model, optimizer, scheduler, config
    a, ao, ar, config = build()
    cpu_a, gpu_a = torch.rand(32), torch.rand(32, device=device)
    b, bo, br, _ = build()
    cpu_b, gpu_b = torch.rand(32), torch.rand(32, device=device)
    assert_state_equal(cpu_a, cpu_b)
    assert_state_equal(gpu_a, gpu_b)
    batch = panel.batch(np.arange(32))
    with torch.no_grad():
        assert_state_equal(a(batch)['amount_normalized'], b(batch)['amount_normalized'])
        assert_state_equal(transformer_loss(a, batch, config, 'mae'), transformer_loss(b, batch, config, 'mae'))
    assert_state_equal(ao.state_dict(), bo.state_dict())
    assert_state_equal(ar.state_dict(), br.state_dict())
    for candidate in [a, b]:
        assert_state_equal(candidate.state_dict(), state['model_state'])
    verify_bindings(ROOT, hashes)
    verify_bindings(ROOT, bindings)
    frozen_inputs(ROOT)
    record = {'status': 'complete_saved_boundary_recovery_preflight',
        'recovery_config_sha256': digest(path), 'input_hashes': dict(bindings, **hashes),
        'original_completed_epoch': 48, 'resumed_first_epoch': 49, 'planned_final_epoch': 60,
        'model_optimizer_scheduler_and_rng_reload_exact': True,
        'same_next_cpu_and_cuda_random_draws': True,
        'actual_training_row_forward_and_loss_exact': True,
        'saved_tensors_and_optimizer_moments_finite': True,
        'complete_epoch_numerical_body_ast_unchanged': True,
        'checkpoint_function_ast_unchanged': True, 'optimizer_steps_in_this_preflight': 0,
        'cpu_synthetic_trajectory_test_separate': True,
        'production_gpu_backward_bitwise_equivalence_claimed': False,
        'formal_cuda_backend_unchanged': not torch.are_deterministic_algorithms_enabled(),
        'source_files_overwritten': False, 'candidate_validation_evaluated': False,
        'data_modified': False, 'baseline_refit': False, 'complete_test_opened': False,
        'script_sha256': digest(Path(__file__)),
        'scope': 'Exact available saved-state recovery and training-row forward checks. Original GPU backward remains nondeterministic; an uninterrupted counterfactual trajectory is unavailable. Resume starts at the next full epoch with the original60-epoch schedule; interruption and unsaved work remain disclosed.'}
    if not record['formal_cuda_backend_unchanged']:
        raise ValueError('Do not change CUDA determinism settings for formal recovery')
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir/'verification.json', record)
    print(json.dumps({key: record[key] for key in ['status', 'original_completed_epoch',
        'resumed_first_epoch', 'complete_epoch_numerical_body_ast_unchanged', 'optimizer_steps_in_this_preflight']}))


if __name__ == '__main__':
    main()
