"""Train-only nutrition/metabolome gradient diagnostic; never updates model parameters."""
import argparse
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.research_inference import NutritionModel
from foodcomp.research_r0 import digest, write_json
from foodcomp.research_r1 import FamilyPanel, PANEL_VERSION, panel_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--device",choices=["cpu","cuda"])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    if args.batches < 1 or args.batch_size < 1:
        raise ValueError("Positive diagnostic sample sizes required.")
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(2)
    wrapper = NutritionModel(args.checkpoint,device=args.device)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model, data = wrapper.model, wrapper.data
    if saved["kind"] not in {"mlp", "v9"}:
        raise ValueError("This diagnostic requires an R1/R2 completion model.")
    config = SimpleNamespace(**saved["config"])
    objective = saved["args"]["objective"]
    panel = FamilyPanel(data, wrapper._cached_text, ROOT / "data/processed" / PANEL_VERSION, wrapper.device)
    if digest(panel.root / "manifest.json") != saved["panel_hash"]:
        raise ValueError("Training panel differs from checkpoint.")
    # Shared representation parameters only: prediction heads and source tables are excluded.
    parameter_items = [(name, p) for name, p in model.named_parameters()
                       if p.requires_grad and "head" not in name and not name.startswith("source_")]
    parameters = [p for _, p in parameter_items]
    if not parameters:
        raise ValueError("No shared representation parameters.")
    masks = {group: torch.as_tensor(data.axes.loss_group.eq(group).to_numpy(), device=wrapper.device)
             for group in ["nutrition", "food_metabolome"]}
    count = min(args.batches * args.batch_size, len(panel.rows))
    indices = np.random.default_rng(args.seed).choice(len(panel.rows), count, replace=False)
    if not np.isin(panel.rows[indices], data.train).all():
        raise AssertionError("Gradient diagnostic accessed non-training tasks.")
    np.save(args.output_dir / "training_task_indices.npy", indices)
    records = []
    model.eval()  # Deterministic local gradients, dropout disabled; not a training intervention.
    initial={key:value.clone() for key,value in model.state_dict().items()}
    cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all() if wrapper.device.type=="cuda" else []
    totals={group:torch.zeros(sum(p.numel() for p in parameters),dtype=torch.float64,device=wrapper.device) for group in masks}
    for start in range(0, count, args.batch_size):
        batch = panel.batch(indices[start:start + args.batch_size])
        outputs = model(batch)
        calibrated = model.calibrated_outputs(outputs, batch) if saved["kind"] == "v9" else None
        vectors, row = {}, {"batch": start // args.batch_size, "tasks": len(indices[start:start + args.batch_size])}
        for group, mask in masks.items():
            group_batch = dict(batch, target=batch["target"] & mask)
            value = panel_loss(outputs, group_batch, config.amount_loss_weight, objective)
            if calibrated is not None:
                weight = config.source_calibrated_loss_weight
                value = (value + weight * panel_loss(calibrated, group_batch, config.amount_loss_weight, objective)) / (1 + weight)
            grads = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
            vector = torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p, g in zip(parameters, grads)])
            if not torch.isfinite(vector).all():
                raise FloatingPointError("Nonfinite task gradient.")
            vectors[group] = vector
            totals[group].add_(vector.double(),alpha=row["tasks"]/count)
            row[group + "_loss"] = float(value.detach())
            row[group + "_norm"] = float(vector.norm())
            row[group + "_targets"] = int(group_batch["target"].sum())
        a, b = vectors["nutrition"], vectors["food_metabolome"]
        if start==0:
            full=panel_loss(outputs,batch,config.amount_loss_weight,objective)
            if calibrated is not None:
                full=(full+weight*panel_loss(calibrated,batch,config.amount_loss_weight,objective))/(1+weight)
            torch.testing.assert_close(full,torch.as_tensor(row["nutrition_loss"]+row["food_metabolome_loss"],device=full.device,dtype=full.dtype),rtol=1e-5,atol=1e-7)
            grads=torch.autograd.grad(full,parameters,retain_graph=True,allow_unused=True)
            vector=torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(parameters,grads)])
            torch.testing.assert_close(a+b,vector,rtol=1e-4,atol=1e-6)
        denominator = a.norm() * b.norm()
        row["both_nonzero"] = bool(denominator > 0)
        row["cosine"] = float(torch.dot(a, b) / denominator) if row["both_nonzero"] else None
        records.append(row)
    frame = pd.DataFrame(records)
    frame.to_csv(args.output_dir / "gradient_batches.csv", index=False)
    valid = frame[frame.both_nonzero]
    if not len(valid):
        raise ValueError("No batches contain both nonzero task gradients.")
    for key,value in initial.items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
    if any(p.grad is not None for p in parameters):raise AssertionError("Diagnostic populated gradient buffers.")
    torch.testing.assert_close(cpu_rng,torch.get_rng_state(),rtol=0,atol=0)
    for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all() if cuda_rng else []):torch.testing.assert_close(a,b,rtol=0,atol=0)
    from foodcomp.research_gradient_geometry import gradient_geometry
    geometry=gradient_geometry([totals["nutrition"]],[totals["food_metabolome"]])
    summary = {
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": digest(args.checkpoint),
        "data_sha256": saved["data_hash"], "panel_sha256": saved["panel_hash"],
        "code_sha256": digest(Path(__file__)), "task_indices_sha256": digest(args.output_dir / "training_task_indices.npy"),
        "seed": args.seed, "objective": objective, "tasks": count,
        "batches": len(frame), "valid_both_task_batches": len(valid),
        "negative_cosine_fraction": float(valid.cosine.lt(0).mean()),
        "mean_cosine": float(valid.cosine.mean()), "median_cosine": float(valid.cosine.median()),
        "median_metabolome_to_nutrition_gradient_norm": float((valid.food_metabolome_norm / valid.nutrition_norm).median()),
        "shared_parameter_names": [name for name, _ in parameter_items],
        "shared_parameter_count": sum(p.numel() for p in parameters),
        "device":str(wrapper.device),"parameters_grad_buffers_and_rng_unchanged":True,
        "first_batch_loss_and_gradient_partition_check_passed":True,
        "undefined_cosine_batches":len(frame)-len(valid),
        "geometry_of_mean_gradients":{"nutrition_norm":geometry["completion_norm"],"metabolome_norm":geometry["name_only_norm"],"dot":geometry["dot"],"cosine":geometry["cosine"],"metabolome_to_nutrition_norm":geometry["name_to_completion_norm"],"both_nonzero":geometry["both_nonzero"]},
        "normalization": "Original 187-axis training normalization retained for both component losses.",
        "model_mode": "eval; dropout disabled; no optimizer or parameter update",
        "interpretation": "Local train-only gradient diagnostic at one selected checkpoint. Negative cosine indicates conflicting infinitesimal directions in these shared parameters, not causal evidence of validation harm. A controlled loss-weight ablation is required before attribution.",
        "complete_test_opened": False,
    }
    write_json(args.output_dir / "summary.json", summary)
    print({k: summary[k] for k in ["tasks", "negative_cosine_fraction", "mean_cosine", "median_metabolome_to_nutrition_gradient_norm"]})


if __name__ == "__main__":
    main()
