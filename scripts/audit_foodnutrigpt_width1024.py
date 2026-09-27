"""Exercise the registered wider model on real training inputs before long training."""
from dataclasses import asdict
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from foodcomp.research_r0 import ResearchData, VERSION, digest, write_json
from foodcomp.research_text import prepare_names
from foodcomp.research_r1 import FamilyPanel, PANEL_VERSION, model_loss
from foodcomp.research_neural import make_model
from foodcomp.research_inference import NutritionModel


def main():
    out = ROOT / "reports/v9_r2_width1024_functional_audit_v1"
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True); torch.set_num_threads(1)
    result = {"status": "incomplete", "complete_test_opened": False, "script_sha256": digest(Path(__file__))}
    try:
        data = ResearchData(ROOT / "data/processed" / VERSION)
        text, cache = prepare_names(data, ROOT)
        panel = FamilyPanel(data, text, ROOT / "data/processed" / PANEL_VERSION, "cpu")
        indices = np.random.default_rng(20260923).choice(len(panel.rows), 64, replace=False)
        if not np.isin(panel.rows[indices], data.train).all():
            raise ValueError("Non-training audit task.")
        batch = panel.batch(indices)
        torch.manual_seed(20260922)
        model, config = make_model(data, text.shape[1], "mlp", mlp_width=1024)
        count = sum(parameter.numel() for parameter in model.parameters())
        if count != 1859836:
            raise ValueError("Unregistered capacity.")
        model.eval()
        with torch.no_grad():
            reference = model(batch)["amount_normalized"]
            changed = dict(batch)
            changed["value"] = torch.where(batch["masked"], batch["value"] + 999, batch["value"])
            changed["target"] = ~batch["target"]
            changed["positive"] = ~batch["positive"]
            changed["source"] = batch["source"] + 99
            torch.testing.assert_close(reference, model(changed)["amount_normalized"], rtol=0, atol=0)
            if not torch.isfinite(reference).all():
                raise FloatingPointError("Nonfinite initial outputs.")
        before = {key: value.clone() for key, value in model.state_dict().items()}
        loss = model_loss(model, batch, "mlp", config, "mae")
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        if any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Absent or nonfinite wider-model gradient.")
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        fixture = out / "untrained_loader_fixture.pt"
        saved = {"model_state": model.state_dict(), "kind": "mlp", "config": asdict(config), "text_dim": text.shape[1],
                 "data_root": str(data.root), "view": data.view, "name_cache": str(cache),
                 "data_hash": digest(data.root / "manifest.json"), "name_cache_hash": digest(cache / "manifest.json"),
                 "args": {"mlp_width": 1024, "mlp_normalization": "layer_norm", "mlp_task_heads": "shared"}}
        torch.save(saved, fixture)
        loaded = NutritionModel(fixture, device="cpu")
        with torch.no_grad():
            torch.testing.assert_close(reference, loaded.model(batch)["amount_normalized"], rtol=0, atol=0)
        name = str(data.profiles.iloc[data.train[0]].original_name)
        for observed in [{}, {int(data.targets[2]): 0.}]:
            prediction = loaded.predict(name, observed, data.targets[:2])
            if len(prediction) != 2 or not np.isfinite(list(prediction.values())).all():
                raise AssertionError("Arbitrary-query API failed.")
        if loaded.encode(name, {}, modality="name").shape != (1024,):
            raise AssertionError("Incorrect representation width.")
        result.update(status="complete", parameter_count=count, real_training_tasks=64,
            hidden_values_labels_and_source_do_not_change_outputs=True, all_gradients_present_and_finite=True,
            save_reload_bitwise_equal=True, arbitrary_query_and_explicit_zero_api_checked=True, initial_loss=float(loss.detach()),
            fixture_sha256=digest(fixture), neural_module_sha256=digest(ROOT / "src/foodcomp/research_neural.py"),
            data_sha256=saved["data_hash"], name_cache_sha256=saved["name_cache_hash"],
            scope="Functional audit of an untrained model on real training inputs. Fixture is not a trained candidate, selected checkpoint or prediction-quality result; no optimizer update occurred.")
    except Exception as error:
        result.update(status="failed", error_type=type(error).__name__, error=str(error))
        write_json(out / "verification.json", result)
        raise
    write_json(out / "verification.json", result)
    print({k: result[k] for k in ["status", "parameter_count", "initial_loss", "save_reload_bitwise_equal"]})


if __name__ == "__main__":
    main()
