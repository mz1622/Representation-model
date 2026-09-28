"""Versioned R9 axis-by-value residual, preserving every frozen predecessor."""
from dataclasses import asdict
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .research_neural import DirectSourceCalibratedModel, legacy_v9, OUTPUT_QUERY_POLICY
from .research_r0 import ResearchData, digest
from .research_r1 import fingerprint_array
from .research_inference import NutritionModel
from .research_transformer_r9_methods import verify_bindings

KIND = 'transformer_axisvalue_direct_v1'
METHOD = 'zero_init_linear_v1'


class AxisValueTransformer(DirectSourceCalibratedModel):
    def __init__(self, text_dim, axis_count, source_count, train_sources, config):
        super().__init__(text_dim, axis_count, source_count, train_sources, config)
        # Construct after the parent, with no random draw and no changed old tensor.
        self.axis_value_residual = nn.Parameter(torch.zeros(axis_count, config.d_model))

    def _hidden(self, batch):
        value_token = self.value_encoder(batch['value'].unsqueeze(-1))
        visible_value = torch.where(batch['masked'], 0., batch['value'])
        value_token = value_token + visible_value.unsqueeze(-1) * self.axis_value_residual[batch['axis']]
        value_token = torch.where(batch['masked'].unsqueeze(-1), self.mask_value.view(1, 1, -1), value_token)
        axis_token = self.axis_embedding(batch['axis']) + value_token
        sequence = torch.cat([self.cls.expand(len(batch['axis']), -1, -1),
            self.text_projection(batch['text']).unsqueeze(1), axis_token], dim=1)
        padding = torch.cat([torch.zeros((len(batch['axis']), 2), dtype=torch.bool,
            device=batch['axis'].device), ~batch['valid']], dim=1)
        return self.encoder(sequence, src_key_padding_mask=padding)[:, 2:]

    def forward(self, batch):
        hidden = self._hidden(batch)
        # Preserve the inactive parent's presence-head dropout RNG consumption.
        self.presence_head(hidden, batch['axis'])
        return {'amount_normalized': self.amount_head(hidden, batch['axis'])}

    def encode(self, batch):
        # Same explicit probe as before, now through the actual residual token path.
        return self._hidden(batch).mean(1)


def base_state_fingerprint(model):
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        if key != 'axis_value_residual':
            h.update(key.encode())
            h.update(fingerprint_array(tensor.detach().cpu().numpy()).encode())
    return h.hexdigest()


def make_axisvalue_transformer(data, text_dim, spec):
    if spec.get('axis_value_residual') != METHOD or spec['objective'] != 'mae':
        raise ValueError('Expected registered axis-value MAE method.')
    config = legacy_v9.Config(d_model=spec['d_model'], n_layers=spec['n_layers'],
        n_heads=spec['n_heads'], feedforward_dim=spec['feedforward_dim'], dropout=spec['dropout'],
        axis_residual_rank=spec['axis_residual_rank'], source_calibrated_loss_weight=spec['source_weight'],
        source_residual_l2=spec['source_residual_l2'])
    if config.d_model % config.n_heads:
        raise ValueError('Invalid head dimension.')
    sources = np.unique(data.profiles.iloc[data.train].source_index)
    model = AxisValueTransformer(text_dim, len(data.axes),
        int(data.profiles.source_index.max()) + 1, sources, config)
    return model, config


def load_method(repo, path, frozen, freeze_path):
    root = Path(repo).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Registration must be inside repository.')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if (plan['status'] != 'registered' or plan['purpose'] != 'r9_single_factor_axisvalue_contrast'
            or plan['changes'] != {'axis_value_residual': METHOD} or plan['seed'] != 20260922
            or plan['candidate'] != 'tf192_mae_axisvalue_lr3e4_60'
            or plan['candidate_index'] != 4 or plan['freeze_sha256'] != digest(freeze_path)):
        raise ValueError('Expected registered same-seed axis-by-value contrast.')
    hashes = dict(plan['input_hashes'])
    hashes[path.relative_to(root).as_posix()] = digest(path)
    verify_bindings(root, hashes)
    parent_path = root / plan['parent_run'] / 'run_manifest.json'
    if parent_path.relative_to(root).as_posix() not in hashes:
        raise ValueError('Parent manifest must be bound.')
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    audit = json.loads((root / plan['parent_audit']).read_text(encoding='utf-8'))
    if (parent['status'] != 'complete' or parent['seed'] != plan['seed']
            or parent['spec']['objective'] != 'mae' or parent['epoch_completed'] != 60
            or audit['status'] != 'complete' or audit['manifest_sha256'] != digest(parent_path)
            or audit['checkpoint_sha256'] != parent['checkpoint_sha256']
            or not audit['both323809_predictions_public_loader_replayed_exact']):
        raise ValueError('Complete audited MAE parent required.')
    for key in ['data_hash', 'panel_hash', 'name_cache_hash']:
        if parent[key] != frozen[key]:
            raise ValueError('Frozen parent mismatch: ' + key)
    for key in ['data_modified', 'baseline_refit', 'complete_test_opened']:
        if parent[key] or plan[key]:
            raise ValueError('Method-only scope violated.')
    spec = copy.deepcopy(parent['spec'])
    spec.update(plan['changes'], name=plan['candidate'])
    difference = {key for key in spec if spec[key] != parent['spec'].get(key)}
    if difference != {'name', 'axis_value_residual'}:
        raise ValueError('More than the declared factor changed.')
    hashes.update(parent['code_hashes'])
    verify_bindings(root, hashes)
    return plan, parent, spec, hashes


class AxisValueNutritionModel(NutritionModel):
    """Same public inputs, retrieval and scaling, with an explicit new checkpoint kind."""
    def __init__(self, checkpoint, repo=None, device=None):
        self.repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.output_query_policy = OUTPUT_QUERY_POLICY
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if saved['version'] != 'V9-R9' or saved['kind'] != KIND:
            raise ValueError('Expected an axis-value R9 checkpoint.')
        self.data = ResearchData(self.repo / saved['data_root'], saved['view'])
        self.cache = self.repo / saved['name_cache']
        if digest(self.data.root / 'manifest.json') != saved['data_hash'] or digest(self.cache / 'manifest.json') != saved['name_cache_hash']:
            raise ValueError('Checkpoint data/name identity changed.')
        manifest = json.loads((self.cache / 'manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['hashes'].items():
            if digest(self.cache / name) != expected:
                raise ValueError('Changed name cache: ' + name)
        self.model, config = make_axisvalue_transformer(self.data, saved['text_dim'], saved['spec'])
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
