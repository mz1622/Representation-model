"""R6 task-ID-keyed extra context deletion, independent of supervised targets."""
import numpy as np
import torch


def context_dropout_masks(task_count,axis_count,mode,seed,epoch):
    if mode not in {'none','fixed_30','mix_30_60_90'}:raise ValueError('Unregistered context dropout mode.')
    if any(not isinstance(v,int) or v<1 for v in [task_count,axis_count,epoch]):raise ValueError('Positive dimensions and epoch required.')
    if not isinstance(seed,int) or seed<0:raise ValueError('Nonnegative integer seed required.')
    if mode=='none':return None,None
    rng=np.random.default_rng(np.random.SeedSequence([seed,epoch,6106]))
    rate_ids=rng.integers(0,3,size=task_count)
    uniforms=rng.random((task_count,axis_count),dtype=np.float32)
    if mode=='fixed_30':rate_ids[:]=0
    rates=np.array([.3,.6,.9],dtype=np.float32)[rate_ids]
    return uniforms<rates[:,None],rate_ids


def drop_context(batch,extra_hidden):
    if extra_hidden is None:return batch
    extra=torch.as_tensor(extra_hidden,device=batch['masked'].device)
    if extra.dtype!=torch.bool or extra.shape!=batch['masked'].shape:raise ValueError('Expected Boolean context mask matching batch.')
    return {**batch,'masked':batch['masked']|extra}
