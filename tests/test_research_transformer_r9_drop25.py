"""Dropout intervention preserves parameters/evaluation but changes stochastic training."""
import ast
import json
from pathlib import Path

import pytest
import torch

from foodcomp.research_transformer_r9 import make_transformer, transformer_loss
from foodcomp.research_transformer_r9_drop25 import load_method, dropout_sites
from test_research_transformer_r9 import example


@pytest.mark.parametrize('training', [False, True])
def test_dropout_control_has_same_weights_but_only_training_function_changes(training):
    data, spec, batch = example()
    torch.manual_seed(20260922)
    control, ca = make_transformer(data, 128, dict(spec, dropout=.15))
    constructor_rng = torch.get_rng_state()
    torch.manual_seed(20260922)
    candidate, cb = make_transformer(data, 128, dict(spec, dropout=.25))
    torch.testing.assert_close(constructor_rng, torch.get_rng_state(), rtol=0, atol=0)
    left_sites, right_sites = dropout_sites(control), dropout_sites(candidate)
    assert left_sites.keys() == right_sites.keys()
    assert set(left_sites.values()) == {.15} and set(right_sites.values()) == {.25}
    assert any('self_attn' in name for name in left_sites)
    assert any('amount_head' in name for name in left_sites)
    for key, value in control.state_dict().items():
        torch.testing.assert_close(value, candidate.state_dict()[key], rtol=0, atol=0)
    control.train(training); candidate.train(training)
    rng = torch.get_rng_state()
    left = transformer_loss(control, batch, ca, 'mae')
    left.backward()
    after = torch.get_rng_state()
    torch.set_rng_state(rng)
    right = transformer_loss(candidate, batch, cb, 'mae')
    right.backward()
    assert torch.isfinite(left) and torch.isfinite(right)
    torch.testing.assert_close(after, torch.get_rng_state(), rtol=0, atol=0)
    differences = []
    for (a,x),(b,y) in zip(control.named_parameters(),candidate.named_parameters()):
        assert a == b
        if x.grad is None:
            assert y.grad is None
        else:
            assert y.grad is not None and torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
            differences.append(not torch.equal(x.grad,y.grad))
    if training:
        assert not torch.equal(left,right) and any(differences)
    else:
        torch.testing.assert_close(left,right,rtol=0,atol=0)
        assert not any(differences)


def test_numeric_batch_loop_and_checkpoint_payload_unchanged():
    root = Path(__file__).resolve().parents[1]
    def part(name, cls, key, value):
        tree = ast.parse((root / 'scripts' / name).read_text(encoding='utf-8'))
        nodes = [n for n in ast.walk(tree) if isinstance(n, cls) and key(n) == value]
        assert len(nodes) == 1
        return ast.dump(nodes[0], include_attributes=False)
    old = 'train_foodnutrigpt_v9_r9_transformer.py'
    new = 'train_foodnutrigpt_v9_r9_drop25.py'
    for cls, key, value in [(ast.For, lambda n: getattr(n.target, 'id', None), 'offset'),
                            (ast.FunctionDef, lambda n: n.name, 'checkpoint')]:
        assert part(old, cls, key, value) == part(new, cls, key, value)


@pytest.mark.parametrize('changes', [{'dropout': .15},
    {'dropout': .25, 'source_weight': 0.}, {'objective': 'mse'}])
def test_registration_rejects_other_changes(tmp_path, changes):
    plan = {'status': 'registered', 'purpose': 'r9_single_factor_dropout_contrast',
            'changes': changes, 'seed': 20260922, 'candidate': 'tf192_mae_drop25_lr3e4_60',
            'candidate_index': 9, 'max_candidates': 12}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(plan), encoding='utf-8')
    with pytest.raises(ValueError, match='registered same-seed'):
        load_method(tmp_path, path, {}, tmp_path / 'unused')
