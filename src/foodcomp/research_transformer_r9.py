"""R9 Transformer-only models using immutable prior data and baseline predictions."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from .research_neural import DirectSourceCalibratedModel, legacy_v9, OUTPUT_QUERY_POLICY
from .research_r0 import ResearchData, digest
from .research_r1 import panel_loss
from .research_inference import NutritionModel


def frozen_inputs(repo):
    repo = Path(repo)
    path = repo / 'reports/v9_r9_freeze_v1/manifest.json'
    frozen = json.loads(path.read_text(encoding='utf-8'))
    if frozen['status'] != 'frozen' or frozen['complete_test_opened']:
        raise ValueError('Expected frozen, test-closed R9 inputs.')
    for name, expected in frozen['input_hashes'].items():
        if digest(repo / name) != expected:
            raise ValueError('Frozen data or baseline changed: ' + name)
    return frozen, path


def make_transformer(data, text_dim, spec):
    config = legacy_v9.Config(d_model=spec['d_model'], n_layers=spec['n_layers'],
        n_heads=spec['n_heads'], feedforward_dim=spec['feedforward_dim'], dropout=spec['dropout'],
        axis_residual_rank=spec['axis_residual_rank'], source_calibrated_loss_weight=spec['source_weight'],
        source_residual_l2=spec['source_residual_l2'])
    if config.d_model % config.n_heads or spec['objective'] not in {'mae', 'smooth_l1', 'mse'}:
        raise ValueError('Invalid Transformer architecture or objective.')
    sources = np.unique(data.profiles.iloc[data.train].source_index)
    model = DirectSourceCalibratedModel(text_dim, len(data.axes),
        int(data.profiles.source_index.max()) + 1, sources, config)
    return model, config


def transformer_loss(model, batch, config, objective):
    def value_loss(output):
        if objective != 'mse':
            return panel_loss(output, batch, objective=objective)
        # Same global axis/source normalization, targets and denominator as MAE.
        error = (output['amount_normalized'] - batch['value']).square()
        weight = batch['cell_weight'] * batch['target'] / batch['axis_total'].clamp_min(1e-12)
        result = (weight * error).sum() * batch['objective_multiplier']
        if not torch.isfinite(result):
            raise FloatingPointError('Nonfinite masked MSE.')
        return result
    output = model(batch)
    base = value_loss(output)
    calibrated = value_loss(model.calibrated_outputs(output, batch))
    weight = config.source_calibrated_loss_weight
    result = (base + weight * calibrated) / (1 + weight) + config.source_residual_l2 * model.source_residual_penalty()
    if not torch.isfinite(result):
        raise FloatingPointError('Nonfinite Transformer loss.')
    return result


class TransformerNutritionModel(NutritionModel):
    """Reuse the validated inference contract with the explicitly saved R9 architecture."""
    def __init__(self, checkpoint, repo=None, device=None):
        self.repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.output_query_policy = OUTPUT_QUERY_POLICY
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if saved['version'] != 'V9-R9' or saved['kind'] != 'transformer_direct':
            raise ValueError('Expected an R9 Transformer checkpoint.')
        self.data = ResearchData(self.repo / saved['data_root'], saved['view'])
        self.cache = self.repo / saved['name_cache']
        if digest(self.data.root / 'manifest.json') != saved['data_hash'] or digest(self.cache / 'manifest.json') != saved['name_cache_hash']:
            raise ValueError('Checkpoint data/name identity changed.')
        manifest = json.loads((self.cache / 'manifest.json').read_text())
        for name, expected in manifest['hashes'].items():
            if digest(self.cache / name) != expected:
                raise ValueError('Changed name cache: ' + name)
        self.model, config = make_transformer(self.data, saved['text_dim'], saved['spec'])
        if asdict(config) != saved['config']:
            raise ValueError('Checkpoint architecture/configuration changed.')
        self.model.load_state_dict(saved['model_state'])
        self.model.to(self.device).eval()
        self.kind = 'v9_direct'
        self.text_dim = saved['text_dim']
        self._encoder = None
        self._pca = dict(np.load(self.cache / 'pca.npz', allow_pickle=False))
        self._cached_text = np.load(self.cache / 'features.npy')
        self._name_index = {name: i for i, name in enumerate(self.data.profiles.original_name.astype(str))}
        self._axes = {str(name): int(i) for i, name in zip(self.data.axes.axis_index, self.data.axes.canonical_name)}
