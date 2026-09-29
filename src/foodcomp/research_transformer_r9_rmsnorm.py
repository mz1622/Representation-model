"""Registered encoder RMSNorm contrast; historical models and loaders stay immutable."""
import copy
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .research_r0 import ResearchData, digest
from .research_inference import NutritionModel
from .research_neural import OUTPUT_QUERY_POLICY
from .research_transformer_r9 import make_transformer as make_control
from .research_transformer_r9_drop25 import verify_bindings

ARCHITECTURE = 'encoder_rmsnorm_v1'
CANDIDATE = 'tf192_mae_rmsnorm_lr3e4_60'
PLAN = 'experiments/foodnutrigpt_v9_research/r9/rmsnorm_v1/config_v2.json'


class RMSNorm(nn.Module):
    """Full RMS normalization, learned scale, no centering or additive bias.

    Keep the parent's epsilon and FP32 training precision. Removing seven
    LayerNorm biases is part of this intervention, not a parameter-matched claim.
    """
    def __init__(self, original):
        super().__init__()
        self.eps = original.eps
        self.normalized_shape = original.normalized_shape
        self.weight = nn.Parameter(original.weight.detach().clone())

    def forward(self, value):
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + self.eps) * self.weight


class ExplicitPreNormLayer(nn.TransformerEncoderLayer):
    """Same attention/FFN/residual operations without a LayerNorm-only fused path.

    Reuse constructed modules so attention, FFN, heads and RNG initialization
    remain identical to the control. rms=False is a functional implementation
    bridge for testing, not another training candidate.
    """
    def __init__(self, original, rms=True):
        nn.Module.__init__(self)
        if not original.norm_first or not original.self_attn.batch_first:
            raise ValueError('Expected existing batch-first Pre-LN architecture')
        for name in ['self_attn', 'linear1', 'linear2', 'dropout', 'dropout1', 'dropout2']:
            setattr(self, name, getattr(original, name))
        self.activation = original.activation
        self.activation_relu_or_gelu = 0
        self.norm_first = True
        self.norm1 = RMSNorm(original.norm1) if rms else original.norm1
        self.norm2 = RMSNorm(original.norm2) if rms else original.norm2

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask, mask_name='src_key_padding_mask',
            other_type=F._none_or_dtype(src_mask), other_name='src_mask', target_type=src.dtype)
        src_mask = F._canonical_mask(mask=src_mask, mask_name='src_mask',
            other_type=None, other_name='', target_type=src.dtype, check_other=False)
        value = src + self._sa_block(self.norm1(src), src_mask, src_key_padding_mask, is_causal=is_causal)
        return value + self._ff_block(self.norm2(value))


def replace_encoder_norms(model, rms=True):
    model.encoder.layers = nn.ModuleList([ExplicitPreNormLayer(layer, rms)
                                         for layer in model.encoder.layers])
    if rms:
        model.encoder.norm = RMSNorm(model.encoder.norm)
    return model


def make_transformer(data, text_dim, spec):
    if spec.get('architecture') != ARCHITECTURE:
        raise ValueError('Explicit RMSNorm architecture required')
    model, config = make_control(data, text_dim, spec)
    return replace_encoder_norms(model), config


def norm_sites(model):
    return {name: {'type': type(module).__name__, 'eps': module.eps,
                   'width': list(module.normalized_shape)}
            for name, module in model.named_modules() if isinstance(module, (RMSNorm, nn.LayerNorm))}


def load_method(repo, path, frozen, freeze_path):
    root = Path(repo).resolve()
    path = Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Registration must remain in repository')
    plan = json.loads(path.read_text(encoding='utf-8'))
    if (plan['status'] != 'registered' or plan['purpose'] != 'r9_single_factor_encoder_rmsnorm'
            or plan['changes'] != {'architecture': ARCHITECTURE}
            or plan['seed'] != 20260922 or plan['candidate'] != CANDIDATE
            or plan['candidate_index'] != 10 or plan['max_candidates'] != 12
            or plan['freeze_sha256'] != digest(freeze_path)):
        raise ValueError('Unexpected structural registration')
    bindings = dict(plan['input_hashes'])
    bindings[path.relative_to(root).as_posix()] = digest(path)
    verify_bindings(root, bindings)
    parent_path = root / plan['parent_run'] / 'run_manifest.json'
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    audit = json.loads((root / plan['parent_audit']).read_text(encoding='utf-8'))
    if (parent['status'] != 'complete' or parent['epoch_completed'] != 60
            or audit['status'] != 'complete' or audit['manifest_sha256'] != digest(parent_path)
            or not audit['both323809_predictions_public_loader_replayed_exact']):
        raise ValueError('Completed audited parent required')
    fixed = {'seed': 20260922, 'dropout': .15, 'learning_rate': .0003,
             'objective': 'mae', 'd_model': 192, 'n_layers': 3, 'n_heads': 6,
             'feedforward_dim': 768, 'batch_size': 64, 'epochs': 60,
             'source_weight': 1., 'source_residual_l2': .0001,
             'weight_decay': .0001, 'gradient_clip': 1., 'active_name_dimensions': 32}
    for key, value in fixed.items():
        if parent['spec'][key] != value:
            raise ValueError('Prohibited recipe change: ' + key)
    for key in ['data_hash', 'panel_hash', 'name_cache_hash']:
        if parent[key] != frozen[key]:
            raise ValueError('Frozen input mismatch: ' + key)
    for key in ['data_modified', 'baseline_refit', 'complete_test_opened']:
        if parent[key] or plan[key]:
            raise ValueError('Method-only scope violation')
    spec = copy.deepcopy(parent['spec'])
    spec.update(plan['changes'], name=CANDIDATE)
    # Current user request supersedes the previous automatic seed expansion.
    # Keep the historical spec fields for provenance; no seed repeats are queued.
    bindings.update(parent['code_hashes'])
    verify_bindings(root, bindings)
    return plan, parent, spec, bindings


class RMSNormNutritionModel(NutritionModel):
    """Public inference contract with an architecture-specific fail-closed loader."""
    def __init__(self, checkpoint, repo=None, device=None):
        self.repo = Path(repo) if repo else Path(__file__).resolve().parents[2]
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.output_query_policy = OUTPUT_QUERY_POLICY
        saved = torch.load(checkpoint, map_location=self.device, weights_only=True)
        if (saved['version'] != 'V9-R9' or saved['kind'] != 'transformer_rmsnorm'
                or saved['spec'].get('architecture') != ARCHITECTURE):
            raise ValueError('Expected registered RMSNorm checkpoint')
        self.data = ResearchData(self.repo / saved['data_root'], saved['view'])
        self.cache = self.repo / saved['name_cache']
        if (digest(self.data.root / 'manifest.json') != saved['data_hash']
                or digest(self.cache / 'manifest.json') != saved['name_cache_hash']):
            raise ValueError('Checkpoint data/name identity changed')
        manifest = json.loads((self.cache / 'manifest.json').read_text(encoding='utf-8'))
        verify_bindings(self.repo, {str((self.cache / name).relative_to(self.repo)): value
                                   for name, value in manifest['hashes'].items()})
        self.model, config = make_transformer(self.data, saved['text_dim'], saved['spec'])
        if asdict(config) != saved['config']:
            raise ValueError('Checkpoint configuration changed')
        self.model.load_state_dict(saved['model_state'], strict=True)
        self.model.to(self.device).eval()
        self.kind = 'v9_direct'
        self.text_dim = saved['text_dim']
        self._encoder = None
        self._pca = dict(np.load(self.cache / 'pca.npz', allow_pickle=False))
        self._cached_text = np.load(self.cache / 'features.npy')
        self._name_index = {name: i for i, name in enumerate(self.data.profiles.original_name.astype(str))}
        self._axes = {str(name): int(i) for i, name in zip(self.data.axes.axis_index, self.data.axes.canonical_name)}
