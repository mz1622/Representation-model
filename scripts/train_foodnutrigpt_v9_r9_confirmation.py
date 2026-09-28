"""Repeat an explicitly selected, audited R9 Transformer with only a new seed."""
import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from foodcomp.research_r0 import ResearchData, VERSION, digest, score_predictions, write_json
from foodcomp.research_r1 import FamilyPanel, fingerprint_array
from foodcomp.research_neural import evaluate
from foodcomp.research_completion_input import completion_view, execution_contract
from foodcomp.research_alignment import state_fingerprint, SCORING, CORRECT
from foodcomp.research_transformer_r9 import frozen_inputs, make_transformer, transformer_loss, TransformerNutritionModel
from evaluate_foodnutrigpt_name_neighbors import evaluate_candidates



def verify_bindings(repo, hashes):
    for relative, expected in hashes.items():
        path = (Path(repo) / relative).resolve()
        if not path.is_relative_to(Path(repo).resolve()) or digest(path) != expected:
            raise ValueError('Confirmation evidence changed: ' + relative)


def load_confirmation(repo, plan_path, seed, frozen, freeze_path):
    """Derive a recipe from completed evidence; expose no hyperparameter overrides."""
    repo = Path(repo).resolve()
    if seed not in (20260923, 20260924):
        raise ValueError('Reuse the audited seed22 parent; only seeds23/24 may be trained.')
    plan_path = Path(plan_path).resolve()
    if not plan_path.is_relative_to(repo):
        raise ValueError('Register the confirmation plan within the repository.')
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes.decode('utf-8-sig'))
    required = {'schema_version', 'purpose', 'seeds', 'freeze_sha256', 'parent_run',
                'parent_manifest_sha256', 'parent_audit', 'parent_audit_sha256',
                'decision_record', 'decision_record_sha256'}
    if set(plan) != required:
        raise ValueError('Unexpected confirmation plan fields; recipe overrides are prohibited.')
    if (plan['schema_version'] != 1 or plan['purpose'] != 'fixed_r9_transformer_seed_confirmation'
            or plan['seeds'] != [20260922, 20260923, 20260924]
            or plan['freeze_sha256'] != digest(freeze_path)):
        raise ValueError('Expected the registered confirmation purpose, seeds and freeze.')
    hashes = {plan_path.relative_to(repo).as_posix(): hashlib.sha256(plan_bytes).hexdigest()}
    def bound(relative, expected):
        path = (repo / relative).resolve()
        if not path.is_relative_to(repo):
            raise ValueError('Evidence must remain within the repository.')
        hashes[path.relative_to(repo).as_posix()] = expected
        verify_bindings(repo, {relative: expected})
        return path
    parent_dir = (repo / plan['parent_run']).resolve()
    if not parent_dir.is_relative_to(repo):
        raise ValueError('Parent must remain within the repository.')
    parent_path = bound((parent_dir / 'run_manifest.json').relative_to(repo).as_posix(),
                        plan['parent_manifest_sha256'])
    parent = json.loads(parent_path.read_text(encoding='utf-8-sig'))
    if (parent['status'] != 'complete' or parent['version'] != 'V9-R9'
            or parent['kind'] != 'transformer_direct' or parent['seed'] != 20260922
            or parent['spec']['seed'] != 20260922
            or any(parent[k] for k in ['complete_test_opened', 'data_modified', 'baseline_refit'])):
        raise ValueError('A complete, unchanged, seed22 R9 Transformer parent is required.')
    if parent['epoch_completed'] != parent['spec']['epochs']:
        raise ValueError('The full registered parent budget must be complete.')
    for key in ['data_hash', 'name_cache_hash', 'panel_hash']:
        if parent[key] != frozen[key]:
            raise ValueError('Parent frozen input mismatch: ' + key)
    if parent['freeze_sha256'] != plan['freeze_sha256']:
        raise ValueError('Parent freeze mismatch.')
    audit_path = bound(plan['parent_audit'], plan['parent_audit_sha256'])
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    if (audit['status'] != 'complete' or audit['complete_test_opened']
            or audit['manifest_sha256'] != plan['parent_manifest_sha256']
            or audit['freeze_sha256'] != plan['freeze_sha256']
            or audit['checkpoint_sha256'] != parent['checkpoint_sha256']
            or not audit['both323809_predictions_public_loader_replayed_exact']
            or not audit['all_candidate_vectors_and19089_ranks_replayed_exact']
            or not audit['all_epoch_orders_exposures_and_schedule_verified']):
        raise ValueError('A matching complete independent parent replay is required.')
    bound((parent_dir / 'best_model.pt').relative_to(repo).as_posix(), parent['checkpoint_sha256'])
    decision_path = bound(plan['decision_record'], plan['decision_record_sha256'])
    decision = json.loads(decision_path.read_text(encoding='utf-8-sig'))
    if (decision['status'] != 'fixed_recipe_for_seed_confirmation'
            or decision['parent_run'] != plan['parent_run']
            or decision['parent_manifest_sha256'] != plan['parent_manifest_sha256']
            or decision['freeze_sha256'] != plan['freeze_sha256']
            or not str(decision['rationale']).strip()
            or len(decision['supporting_evidence']) < 2):
        raise ValueError('An explicit recipe decision with supporting evidence is required.')
    for evidence in decision['supporting_evidence']:
        bound(evidence['path'], evidence['sha256'])
    if len({(repo / item['path']).resolve() for item in decision['supporting_evidence']}) < 2:
        raise ValueError('Distinct supporting evidence records are required.')
    for relative, expected in parent['code_hashes'].items():
        bound(relative, expected)
    spec = copy.deepcopy(parent['spec'])
    spec['seed'] = seed
    if {k: v for k, v in spec.items() if k != 'seed'} != {
            k: v for k, v in parent['spec'].items() if k != 'seed'}:
        raise AssertionError('The fixed recipe changed.')
    return plan, parent, spec, hashes


def verify_environment(expected):
    import sklearn
    import xgboost
    actual = {'python': platform.python_version(), 'platform': platform.platform(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'torch': torch.__version__,
        'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__,
        'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if actual != expected:
        raise ValueError('Confirmation environment differs from the selected parent: ' + repr(actual))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[20260923, 20260924], required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Do not overwrite an earlier confirmation attempt.')
    config_path = args.plan.resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError('Register the confirmation plan within the repository.')
    expected_plan_hash = digest(config_path)
    frozen, freeze_path = frozen_inputs(ROOT)
    plan, parent_manifest, spec, confirmation_hashes = load_confirmation(
        ROOT, config_path, args.seed, frozen, freeze_path)
    if digest(config_path) != expected_plan_hash:
        raise ValueError('Confirmation plan changed during loading.')
    verify_environment(parent_manifest['environment'])
    args.candidate = parent_manifest['candidate']
    torch.set_num_threads(4)
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = ResearchData(ROOT / 'data/processed' / VERSION)
    text, cache, panel_root = completion_view(data, ROOT, spec['active_name_dimensions'])
    panel = FamilyPanel(data, text, panel_root, device)
    model, config = make_transformer(data, text.shape[1], spec)
    model.to(device)
    args.output_dir.mkdir(parents=True)
    started = time.monotonic()
    old, _ = execution_contract(ROOT)
    files = list(dict.fromkeys(list(parent_manifest['code_hashes']) +
        ['scripts/train_foodnutrigpt_v9_r9_confirmation.py']))
    snapshot = args.output_dir / 'code_snapshot'
    for name in files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    manifest = {'status': 'running', 'version': 'V9-R9', 'kind': 'transformer_direct', 'candidate': args.candidate,
        'spec': spec, 'seed': spec['seed'], 'code_hashes': {p: digest(ROOT / p) for p in files},
        'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'freeze_sha256': digest(freeze_path), 'functional_sha256': parent_manifest['functional_sha256'],
        'config_sha256': expected_plan_hash, 'data_hash': frozen['data_hash'], 'panel_hash': frozen['panel_hash'],
        'name_cache_hash': frozen['name_cache_hash'], 'initial_state_sha256': state_fingerprint(model),
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'training_tasks': len(panel.rows), 'observed_target_cells': panel.manifest['observed_target_cells'],
        'environment': old['environment'], 'complete_test_opened': False,
        'baseline_refit': False, 'data_modified': False, 'scientific_confirmation': False}
    manifest.update(purpose='fixed_recipe_seed_reproducibility',
        confirmation_plan=str(config_path.relative_to(ROOT)),
        confirmation_input_hashes=confirmation_hashes,
        seed22_parent=plan['parent_run'], seed22_manifest_sha256=plan['parent_manifest_sha256'],
        parent_audit_sha256=plan['parent_audit_sha256'], decision_record_sha256=plan['decision_record_sha256'],
        recipe_changes=['seed'])
    status_path = args.output_dir / 'run_manifest.json'
    write_json(status_path, manifest)
    def checkpoint(epoch):
        return {'version': 'V9-R9', 'kind': 'transformer_direct', 'spec': spec, 'config': asdict(config),
            'model_state': model.state_dict(), 'text_dim': text.shape[1], 'seed': spec['seed'], 'best_epoch': epoch,
            'data_root': data.root.relative_to(ROOT).as_posix(), 'view': data.view,
            'name_cache': cache.relative_to(ROOT).as_posix(), 'data_hash': frozen['data_hash'],
            'name_cache_hash': frozen['name_cache_hash'], 'panel_hash': frozen['panel_hash']}
    try:
        initial = copy.deepcopy(model.state_dict())
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state_all() if device.type == 'cuda' else None
        batch = panel.batch(np.random.default_rng(spec['seed']).choice(len(panel.rows), 32, replace=False))
        smoke = torch.optim.AdamW(model.parameters(), lr=.001)
        model.eval()
        before = float(transformer_loss(model, batch, config, spec['objective']).detach())
        model.train()
        for _ in range(50):
            smoke.zero_grad(set_to_none=True)
            loss = transformer_loss(model, batch, config, spec['objective'])
            loss.backward()
            if not torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)):
                raise FloatingPointError('Nonfinite smoke gradient.')
            smoke.step()
        model.eval()
        after = float(transformer_loss(model, batch, config, spec['objective']).detach())
        if not after < before:
            raise AssertionError('Small training batch did not overfit.')
        model.load_state_dict(initial)
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state_all(gpu_rng)
        del initial, smoke, batch
        manifest['overfit'] = {'before': before, 'after': after, 'steps': 50, 'weights_and_rng_restored': True}
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
            T_max=spec['schedule_epochs'], eta_min=spec['learning_rate'] * .01)
        best, history = float('inf'), []
        for epoch in range(1, spec['epochs'] + 1):
            if digest(config_path) != manifest['config_sha256']:
                raise ValueError('Registration changed during training.')
            frozen_inputs(ROOT)
            verify_bindings(ROOT, confirmation_hashes)
            for name, expected in manifest['code_hashes'].items():
                if digest(ROOT / name) != expected:
                    raise ValueError('Executable source changed during training: ' + name)
            order = np.random.default_rng(spec['seed'] + epoch).permutation(len(panel.rows))
            model.train()
            total, seen, targets, norm_sum, clipped, steps = 0., 0, 0, 0., 0, 0
            for offset in range(0, len(order), spec['batch_size']):
                indices = order[offset:offset + spec['batch_size']]
                batch = panel.batch(indices)
                optimizer.zero_grad(set_to_none=True)
                loss = transformer_loss(model, batch, config, spec['objective'])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), spec['gradient_clip'])
                if not torch.isfinite(norm):
                    raise FloatingPointError('Nonfinite training gradient.')
                optimizer.step()
                total += float(loss.detach()) * len(indices)
                seen += len(indices)
                targets += int(batch['target'].sum())
                norm_sum += float(norm)
                clipped += int(norm > spec['gradient_clip'])
                steps += 1
            if seen != len(panel.rows) or targets != manifest['observed_target_cells']:
                raise ValueError('Training target exposure changed.')
            prediction = evaluate(model, data, text, device)
            metric, _, _ = score_predictions(data, prediction)
            primary = metric['nutrition']['scaled_log_mae']
            history.append({'epoch': epoch, 'train_loss': total / seen, 'validation_primary': primary,
                'validation_legacy_log_mae': metric['nutrition']['log_mae'],
                'validation_positive_mae': metric['nutrition']['positive_scaled_log_mae'],
                'validation_zero_mae': metric['nutrition']['zero_scaled_log_mae'],
                'learning_rate': optimizer.param_groups[0]['lr'], 'mean_preclip_gradient_norm': norm_sum / steps,
                'gradient_clip_fraction': clipped / steps, 'training_tasks': seen, 'observed_target_cells': targets,
                'training_order_sha256': fingerprint_array(order), 'elapsed_seconds': time.monotonic() - started})
            pd.DataFrame(history).to_csv(args.output_dir / 'history.csv', index=False)
            if primary < best:
                best = primary
                torch.save(checkpoint(epoch), args.output_dir / 'best_model.pt')
            if epoch in [20, spec['epochs']]:
                (args.output_dir / f'best_through_epoch_{epoch:03d}.pt').write_bytes((args.output_dir / 'best_model.pt').read_bytes())
            scheduler.step()
            state = checkpoint(epoch)
            state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
            torch.save(state, args.output_dir / 'latest_training_state.pt')
            manifest.update(epoch_completed=epoch, best_primary=best, elapsed_seconds=time.monotonic() - started)
            write_json(status_path, manifest)
            print(f'{args.candidate} epoch {epoch}/{spec["epochs"]}: loss {total/seen:.6f}; primary {primary:.6f}; elapsed {time.monotonic()-started:.1f}s', flush=True)
        saved = torch.load(args.output_dir / 'best_model.pt', map_location=device, weights_only=True)
        model.load_state_dict(saved['model_state'])
        loaded = TransformerNutritionModel(args.output_dir / 'best_model.pt', ROOT, str(device))
        scores = {}
        for task in ['completion', 'name_only']:
            prediction = evaluate(model, data, text, device, task)
            replay = evaluate(loaded.model, loaded.data, loaded._cached_text, device, task)
            pd.testing.assert_frame_equal(prediction, replay, check_exact=True)
            score, axes, groups = score_predictions(data, prediction)
            prediction.to_parquet(args.output_dir / f'{task}_predictions.parquet', index=False)
            axes.to_csv(args.output_dir / f'{task}_axis_metrics.csv', index=False)
            groups.to_parquet(args.output_dir / f'{task}_candidate_errors.parquet', index=False)
            scores[task] = score
        write_json(args.output_dir / 'metrics.json', scores)
        names = sorted(set(data.profiles.original_name.astype(str)))
        axes = np.flatnonzero(data.axes.loss_group.eq('nutrition') & data.axes.loss_eligible)
        raw = loaded.candidate_profiles(names, target_axes=axes)
        scaled = np.log1p(raw / data.scale[axes]).astype(np.float32)
        np.save(args.output_dir / 'candidate_scaled.npy', scaled)
        ret = args.output_dir / 'retrieval'
        ret.mkdir()
        write_json(ret / 'candidate_names.json', names)
        ranks, metrics = evaluate_candidates(data, names, scaled, axes, str(device))
        ranks.to_parquet(ret / 'ranks.parquet', index=False)
        replay, other = evaluate_candidates(data, names, np.load(args.output_dir / 'candidate_scaled.npy'), axes, str(device))
        pd.testing.assert_frame_equal(ranks, replay, check_exact=True)
        assert metrics == other
        write_json(ret / 'metrics.json', {'metrics': metrics, 'candidate_count': len(names),
            'query_profiles': ranks.groupby('visible_fraction').size().to_dict(),
            'candidate_sha256': digest(ret / 'candidate_names.json'), 'data_sha256': manifest['data_hash'],
            'name_cache_sha256': manifest['name_cache_hash'], 'checkpoint_sha256': digest(args.output_dir / 'best_model.pt'),
            'scoring': SCORING, 'correct_answers': CORRECT, 'method': 'R9_transformer_name_predicted_profiles',
            'complete_test_opened': False})
        frozen_inputs(ROOT)
        verify_bindings(ROOT, confirmation_hashes)
        manifest.update(status='complete', elapsed_seconds=time.monotonic() - started,
            best_epoch=saved['best_epoch'], checkpoint_sha256=digest(args.output_dir / 'best_model.pt'),
            both_complete_prediction_tables_reloaded_exact=True, saved_matrix_retrieval_reloaded_exact=True)
        write_json(status_path, manifest)
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic() - started)
        write_json(status_path, manifest)
        raise


if __name__ == '__main__':
    main()
