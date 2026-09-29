"""Resume the saved epoch48 boundary into a new directory without recipe changes."""
import argparse
import copy
from datetime import datetime, timezone
from dataclasses import asdict
import json
from pathlib import Path
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
from foodcomp.research_transformer_r9_capacity256 import load_method, verify_bindings
from foodcomp.research_transformer_r9_capacity256_recovery import load_recovery, restore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('Do not overwrite an earlier trial.')
    recovery_path = ROOT / 'experiments/foodnutrigpt_v9_research/r9/capacity256_v1/recovery1/config.json'
    recovery, original_manifest, old_history, training_state, recovery_hashes = load_recovery(ROOT, recovery_path)
    if args.output_dir.resolve() != (ROOT / recovery['output_run']).resolve():
        raise ValueError('Resume output differs from operational registration')
    verification_path = ROOT / 'reports/v9_r9_capacity256_recovery_functional_v1/verification.json'
    recovery_check = json.loads(verification_path.read_text(encoding='utf-8'))
    if (recovery_check['status'] != 'complete_saved_boundary_recovery_preflight'
            or recovery_check['recovery_config_sha256'] != digest(recovery_path)):
        raise ValueError('Matching completed recovery verification required')
    verify_bindings(ROOT, recovery_check['input_hashes'])
    recovery_hashes[verification_path.relative_to(ROOT).as_posix()] = digest(verification_path)
    config_path = ROOT / 'experiments/foodnutrigpt_v9_research/r9/capacity256_v1/config.json'
    frozen, freeze_path = frozen_inputs(ROOT)
    plan, parent, spec, bindings = load_method(ROOT, config_path, frozen, freeze_path)
    if args.candidate != plan['candidate']:
        raise ValueError('Candidate not registered in the method plan')
    functional_path = ROOT / 'reports/v9_r9_capacity256_functional_v1/verification.json'
    functional = json.loads(functional_path.read_text(encoding='utf-8'))
    if (functional['status'] != 'complete' or functional['freeze_sha256'] != digest(freeze_path)
            or functional['plan_sha256'] != digest(config_path)):
        raise ValueError('Current capacity functional verification is required')
    verify_bindings(ROOT, functional['implementation_hashes'])
    bindings[functional_path.relative_to(ROOT).as_posix()] = digest(functional_path)
    torch.set_num_threads(4)
    torch.manual_seed(spec['seed'])
    np.random.seed(spec['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = ResearchData(ROOT / 'data/processed' / VERSION)
    text, cache, panel_root = completion_view(data, ROOT, spec['active_name_dimensions'])
    panel = FamilyPanel(data, text, panel_root, device)
    model, config = make_transformer(data, text.shape[1], spec)
    model.to(device)
    if (state_fingerprint(model) != functional['initial_state_sha256']
            or sum(p.numel() for p in model.parameters()) != functional['parameter_count']
            or sum(p.numel() for p in model.parameters() if p.requires_grad) != functional['trainable_parameter_count']):
        raise ValueError('Initial capacity model differs from its verified functional construction')
    args.output_dir.mkdir(parents=True)
    segment_started = time.monotonic()
    started = segment_started - original_manifest['elapsed_seconds']
    manifest = copy.deepcopy(original_manifest)
    original = ROOT / recovery['source_run']
    files = list(manifest['code_hashes']) + [
        'src/foodcomp/research_transformer_r9_capacity256_recovery.py',
        'scripts/resume_foodnutrigpt_v9_r9_capacity256.py']
    snapshot = args.output_dir / 'code_snapshot'
    for name in files:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((ROOT / name).read_bytes())
    for name in ['history.csv', 'best_model.pt', 'best_through_epoch_020.pt', 'latest_training_state.pt']:
        (args.output_dir / name).write_bytes((original / name).read_bytes())
    manifest['code_hashes'] = {name: digest(ROOT / name) for name in files}
    manifest.update(recovery_config_sha256=digest(recovery_path), recovery_input_hashes=recovery_hashes,
        original_run=recovery['source_run'], original_manifest_sha256=digest(original / 'run_manifest.json'),
        original_completed_epochs=48, resumed_first_epoch=49, resume_count=1,
        recovery_code_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        recovery_script_sha256=digest(Path(__file__)), recovery_started_utc=datetime.now(timezone.utc).isoformat(),
        original_saved_elapsed_seconds=original_manifest['elapsed_seconds'],
        elapsed_seconds_scope='Saved original elapsed plus resumed segment. Unsaved partial work and downtime excluded.',
        interruption_cause='Unknown: original PIDs and session missing; no terminal error in log.',
        numerical_recipe_changed_by_recovery=False)
    status_path = args.output_dir / 'run_manifest.json'
    write_json(status_path, manifest)
    def checkpoint(epoch):
        return {'version': 'V9-R9', 'kind': 'transformer_direct', 'spec': spec, 'config': asdict(config),
            'model_state': model.state_dict(), 'text_dim': text.shape[1], 'seed': spec['seed'], 'best_epoch': epoch,
            'data_root': data.root.relative_to(ROOT).as_posix(), 'view': data.view,
            'name_cache': cache.relative_to(ROOT).as_posix(), 'data_hash': frozen['data_hash'],
            'name_cache_hash': frozen['name_cache_hash'], 'panel_hash': frozen['panel_hash']}
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'], weight_decay=spec['weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
            T_max=spec['schedule_epochs'], eta_min=spec['learning_rate'] * .01)
        restore(model, optimizer, scheduler, training_state, device)
        manifest['recovery_model_optimizer_scheduler_and_rng_loaded_exact'] = True
        write_json(status_path, manifest)
        best, history = original_manifest['best_primary'], old_history.to_dict('records')
        del training_state
        for epoch in range(49, spec['epochs'] + 1):
            if digest(config_path) != manifest['config_sha256']:
                raise ValueError('Registration changed during training.')
            frozen_inputs(ROOT)
            verify_bindings(ROOT, bindings)
            verify_bindings(ROOT, recovery_hashes)
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
        verify_bindings(ROOT, bindings)
        verify_bindings(ROOT, recovery_hashes)
        manifest.update(status='complete', elapsed_seconds=time.monotonic() - started,
            recovery_segment_elapsed_seconds=time.monotonic() - segment_started,
            recovery_finished_utc=datetime.now(timezone.utc).isoformat(),
            best_epoch=saved['best_epoch'], checkpoint_sha256=digest(args.output_dir / 'best_model.pt'),
            both_complete_prediction_tables_reloaded_exact=True, saved_matrix_retrieval_reloaded_exact=True)
        write_json(status_path, manifest)
    except Exception as error:
        manifest.update(status='failed', error_type=type(error).__name__, error=str(error), elapsed_seconds=time.monotonic() - started)
        write_json(status_path, manifest)
        raise


if __name__ == '__main__':
    main()
