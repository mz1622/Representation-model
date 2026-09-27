"""Quantify source-residual/global-clipping coupling at a common V9 initialization."""
import json
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_text import prepare_names
from foodcomp.research_r1 import FamilyPanel, PANEL_VERSION, panel_loss, fingerprint_array
from foodcomp.research_neural import make_model


def norm(values):
    squared = sum(float(value.double().square().sum()) for value in values)
    if not np.isfinite(squared):
        raise FloatingPointError("Nonfinite gradient norm.")
    return squared ** .5


def main():
    out = ROOT / "reports/v9_r1_source_clipping_audit_v1"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    started = time.monotonic()
    result = {"status": "incomplete", "complete_test_opened": False, "script_sha256": digest(Path(__file__))}
    try:
        torch.set_num_threads(1)
        data = ResearchData(ROOT / "data/processed" / VERSION)
        text, cache = prepare_names(data, ROOT)
        panel = FamilyPanel(data, text, ROOT / "data/processed" / PANEL_VERSION, "cpu")
        torch.manual_seed(20260922)
        model, config = make_model(data, text.shape[1], "v9", amount_weight=1., source_weight=1.)
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        initial_rng = torch.get_rng_state()
        named = list(model.named_parameters())
        parameters = [parameter for _, parameter in named]
        source = [i for i, (name, _) in enumerate(named) if name.startswith(("source_amount_residual.", "source_presence_residual."))]
        shared = [i for i in range(len(parameters)) if i not in source]
        if len(source) != 2 or any(torch.count_nonzero(parameters[i]).item() for i in source):
            raise ValueError("Require exactly two zero-initialized source residual parameter arrays.")
        indices = np.random.default_rng(20260923).choice(len(panel.rows), 512, replace=False)
        if not np.isin(panel.rows[indices], data.train).all():
            raise ValueError("Non-training task in clipping audit.")
        rows = []; model.train()
        for start in range(0, 512, 64):
            batch = panel.batch(indices[start:start + 64])
            outputs = model(batch)
            free = panel_loss(outputs, batch, config.amount_loss_weight, "hurdle")
            calibrated = panel_loss(model.calibrated_outputs(outputs, batch), batch, config.amount_loss_weight, "hurdle")
            torch.testing.assert_close(free, calibrated, rtol=0, atol=0)
            penalty = config.source_residual_l2 * model.source_residual_penalty()
            gradients = {}; stats = {}
            for weight in [0, 1]:
                objective = (free + weight * calibrated) / (1 + weight) + penalty
                values = torch.autograd.grad(objective, parameters, retain_graph=(weight == 0), allow_unused=True)
                values = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, values)]
                if not all(torch.isfinite(g).all() for g in values):
                    raise FloatingPointError("Nonfinite gradient.")
                gradients[weight] = values
                before = norm([values[i] for i in shared])
                if before <= 0:
                    raise ValueError("Undefined shared-gradient ratio at zero norm.")
                residual = norm([values[i] for i in source])
                for parameter, gradient in zip(parameters, values):
                    parameter.grad = gradient.detach().clone()
                total = float(torch.nn.utils.clip_grad_norm_(parameters, 1.))
                after = norm([parameters[i].grad for i in shared])
                stats[weight] = {"loss": float(objective.detach()), "shared_norm_before": before,
                                 "source_norm_before": residual, "global_norm_before": total,
                                 "shared_norm_after": after, "clipped": total > 1.,
                                 "shared_post_pre_ratio": after / before}
            max_difference = max(float((gradients[0][i] - gradients[1][i]).abs().max()) for i in shared)
            for index in shared:
                torch.testing.assert_close(gradients[0][index], gradients[1][index], rtol=1e-6, atol=1e-7)
            ratio = stats[1]["shared_norm_after"] / stats[0]["shared_norm_after"]
            rows.append({"batch": start // 64, "tasks": 64, "free_calibrated_losses_equal": True,
                         "shared_preclip_max_difference": max_difference,
                         "on_off_shared_postclip_norm_ratio": ratio,
                         **{f"w{weight}_{key}": value for weight, part in stats.items() for key, value in part.items()}})
            model.zero_grad(set_to_none=True)
            print({"batch": start // 64, "shared_preclip_max_difference": max_difference,
                   "postclip_shared_on_off_ratio": ratio}, flush=True)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
        torch.set_rng_state(initial_rng)
        frame = pd.DataFrame(rows)
        frame.to_csv(out / "batch_gradient_clipping.csv", index=False)
        ratios = frame.on_off_shared_postclip_norm_ratio.to_numpy()
        result.update(status="complete", seed=20260922, task_sample_seed=20260923,
            tasks=512, batch_size=64, batches=8, task_indices_sha256=fingerprint_array(indices),
            shared_preclip_max_absolute_difference=float(frame.shared_preclip_max_difference.max()),
            all_shared_preclip_gradients_bitwise_equal=bool(frame.shared_preclip_max_difference.eq(0).all()),
            on_off_postclip_shared_norm_ratio={"min": float(ratios.min()), "median": float(np.median(ratios)), "max": float(ratios.max())},
            clipped_batches={str(w): int(frame[f"w{w}_clipped"].sum()) for w in [0, 1]},
            source_parameters_initially_zero=True, model_parameters_unchanged=True,
            diagnostic_rng_restored=True, optimizer_updates=0,
            data_sha256=digest(data.root / "manifest.json"), panel_sha256=digest(panel.root / "manifest.json"),
            name_cache_sha256=digest(cache / "manifest.json"),
            code_hashes={str(path): digest(ROOT / path) for path in ["src/foodcomp/research_r1.py", "src/foodcomp/research_neural.py", "scripts/train_global_foodnutrigpt_v9_source_calibrated.py"]},
            elapsed_seconds=time.monotonic() - started,
            scope="Eight training batches at the common zero-residual initialization; one shared train-mode forward/dropout per paired loss. Measures gradient clipping, not AdamW updates, learned-residual behavior or validation causality. Diagnostic restores its own RNG and does not modify any training run.")
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(out / "verification.json", result)
        raise
    write_json(out / "verification.json", result)
    print(json.dumps({key: result[key] for key in ["status", "shared_preclip_max_absolute_difference", "clipped_batches", "on_off_postclip_shared_norm_ratio", "elapsed_seconds"]}))


if __name__ == "__main__":
    main()
