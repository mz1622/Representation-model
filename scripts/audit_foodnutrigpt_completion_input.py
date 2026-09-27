"""Train-only R8 matched model, label isolation and save/reload preflight."""
import argparse
import copy
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_r1 import FamilyPanel, panel_loss, model_loss, fingerprint_array
from foodcomp.research_neural import make_model
from foodcomp.research_completion_input import completion_view
from foodcomp.research_alignment import state_fingerprint
from foodcomp.research_inference import NutritionModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(1)
    receipt = {'status': 'incomplete', 'script_sha256': digest(Path(__file__)), 'complete_test_opened': False}
    try:
        data = ResearchData(ROOT / 'data/processed' / VERSION)
        records, batches, states = [], [], []
        for dims in [32, 128]:
            text, cache, root = completion_view(data, ROOT, dims)
            panel = FamilyPanel(data, text, root, 'cpu')
            tasks = np.random.default_rng(20260923).choice(len(panel.rows), 768, replace=False)
            batch = panel.batch(tasks)
            batches.append(batch)
            torch.manual_seed(20260922)
            model, config = make_model(data, 128, 'mlp', mlp_width=512)
            states.append(state_fingerprint(model))
            assert sum(p.numel() for p in model.parameters()) == 717052
            output = model(batch)['amount_normalized']
            changed = copy.deepcopy(batch)
            changed['value'][changed['masked']] += 999
            changed['target'] = ~changed['target']; changed['positive'] = ~changed['positive']; changed['source'] += 99
            assert torch.equal(output, model(changed)['amount_normalized'])
            assert batch['masked'][panel.masks[panel.family_ids[tasks]]].all()
            changed = copy.deepcopy(batch); changed['value'][~changed['target']] = 1234
            assert torch.equal(panel_loss({'amount_normalized': output}, batch, objective='mae'),
                panel_loss({'amount_normalized': output}, changed, objective='mae'))
            for bad in [float('nan'), float('inf')]:
                invalid = output.detach().clone(); invalid[0, 0] = bad
                try:
                    panel_loss({'amount_normalized': invalid}, batch, objective='mae')
                except FloatingPointError:
                    pass
                else:
                    raise AssertionError('Nonfinite output did not fail.')
            model.zero_grad(set_to_none=True)
            model_loss(model, batch, 'mlp', config, 'mae').backward()
            extra_grad = model.encoder[0].weight.grad[:, 32:128]
            assert bool(extra_grad.any()) == (dims == 128)
            # A diagnostic-only tiny training fit, never evaluated or selected on validation.
            tiny = panel.batch(tasks[:32]); opt = torch.optim.AdamW(model.parameters(), lr=.001)
            before = float(model_loss(model, tiny, 'mlp', config, 'mae').detach())
            for _ in range(50):
                opt.zero_grad(set_to_none=True)
                loss = model_loss(model, tiny, 'mlp', config, 'mae'); loss.backward(); opt.step()
            after = float(model_loss(model, tiny, 'mlp', config, 'mae').detach())
            assert after < before
            model.eval()
            with torch.no_grad():
                output = model(batch)['amount_normalized']
                without = copy.deepcopy(batch); without['masked'].fill_(True)
                assert not torch.equal(output, model(without)['amount_normalized'])
            path = args.output_dir / f'name{dims}_train_only_fixture_not_candidate.pt'
            torch.save({'model_state': model.state_dict(), 'kind': 'mlp', 'config': asdict(config),
                'text_dim': 128, 'data_root': str(data.root), 'view': data.view, 'name_cache': str(cache),
                'data_hash': digest(data.root / 'manifest.json'), 'name_cache_hash': digest(cache / 'manifest.json'),
                'args': {'mlp_width': 512}}, path)
            loaded = NutritionModel(path, device='cpu')
            with torch.no_grad():
                assert torch.equal(output, loaded.model(batch)['amount_normalized'])
            axes = data.axes.loc[data.axes.loss_eligible & data.axes.loss_group.eq('nutrition'), 'axis_index'].to_numpy()
            name = data.profiles.original_name.iloc[data.train[0]]
            for context in [{}, {int(axes[2]): 0.}]:
                assert np.isfinite(list(loaded.predict(name, context, axes[:2]).values())).all()
            records.append({'dimensions': dims, 'parameters': 717052, 'initial_state_sha256': states[-1],
                'tasks_sha256': fingerprint_array(tasks), 'panel_sha256': digest(root / 'manifest.json'),
                'name_cache_sha256': digest(cache / 'manifest.json'), 'overfit_before': before,
                'overfit_after': after, 'overfit_steps': 50, 'save_reload_exact': True,
                'fixture_sha256': digest(path), 'inactive_input_gradients_zero': dims == 32})
        assert states[0] == states[1]
        for key in batches[0]:
            if key != 'text':
                left, right = batches[0][key], batches[1][key]
                assert (torch.equal(left, right) if isinstance(left, torch.Tensor) else left == right), key
        assert torch.equal(batches[0]['text'][:, :32], batches[1]['text'][:, :32])
        assert not batches[0]['text'][:, 32:].any()
        receipt.update(status='complete', records=records, actual_train_tasks=768,
            matched_initialization=True, all_nontext_batch_fields_equal=True,
            hidden_values_labels_sources_do_not_change_predictions=True, target_family_hidden=True,
            non_target_values_do_not_enter_loss=True, nan_inf_outputs_fail=True,
            numeric_context_affects_predictions=True, caller_unobserved_axes_predictable=True,
            scope='Train-only diagnostic fixtures, not formal candidates. R7 gate and complete trainer/tree implementation still required before R8 launches.')
    except Exception as error:
        receipt.update(status='failed', error_type=type(error).__name__, error=str(error))
        write_json(args.output_dir / 'verification.json', receipt)
        raise
    write_json(args.output_dir / 'verification.json', receipt)
    print(receipt['status'], records)


if __name__ == '__main__':
    main()
