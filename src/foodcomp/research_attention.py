"""Descriptive attention accounting; no parameter updates or causal attribution."""
import torch


def attention_category_mass(attention, query_weight, categories):
    if attention.ndim!=4 or query_weight.shape!=(attention.shape[0],attention.shape[2]):
        raise ValueError("Attention/query dimensions differ.")
    if not torch.isfinite(attention).all() or not torch.isfinite(query_weight).all() or (query_weight<0).any() or (attention<0).any():
        raise FloatingPointError("Nonfinite or negative diagnostic values.")
    if not torch.allclose(attention.sum(-1),torch.ones_like(attention[...,0]),atol=2e-5,rtol=2e-5):
        raise ValueError("Attention probabilities do not sum to one; disable dropout.")
    coverage=torch.zeros((attention.shape[0],attention.shape[3]),dtype=torch.int64,device=attention.device)
    for mask in categories.values():
        if mask.dtype!=torch.bool or mask.shape!=coverage.shape:raise ValueError("Invalid key category mask.")
        coverage+=mask
    if not (coverage==1).all():raise ValueError("Key categories must partition every key exactly once.")
    denominator=query_weight.double().sum()
    if not denominator>0:raise ValueError("No positive query weight.")
    result={}
    for name,mask in categories.items():
        mass=(attention*mask[:,None,None,:]).sum(-1)
        result[name]=(mass.double()*query_weight[:,None,:].double()).sum((0,2))/denominator
    return result


@torch.no_grad()
def replay_v9_attention(model,batch,collect):
    """Observe attention from preserved layer inputs without installing module hooks.

    Hooks disable TransformerEncoderLayer's fused evaluation path in PyTorch.
    The diagnostic attention call is separate; its outputs never feed the replay.
    Callers must verify replay predictions against the ordinary model forward.
    """
    if model.training or any(module._forward_hooks or module._forward_pre_hooks for module in model.modules()):
        raise ValueError("Diagnostic replay requires eval mode and no forward hooks.")
    if model.encoder.norm is None or not all(layer.norm_first for layer in model.encoder.layers):
        raise ValueError("Unsupported V9 encoder structure.")
    value=model.value_encoder(batch["value"].unsqueeze(-1))
    value=torch.where(batch["masked"].unsqueeze(-1),model.mask_value.view(1,1,-1),value)
    count=len(batch["axis"])
    hidden=torch.cat([model.cls.expand(count,-1,-1),model.text_projection(batch["text"]).unsqueeze(1),
                      model.axis_embedding(batch["axis"])+value],1)
    padding=torch.cat([torch.zeros((count,2),dtype=torch.bool,device=hidden.device),~batch["valid"]],1)
    for layer_index,layer in enumerate(model.encoder.layers):
        query=layer.norm1(hidden)
        _,weights=layer.self_attn(query,query,query,key_padding_mask=padding,need_weights=True,average_attn_weights=False)
        collect(layer_index,weights)
        hidden=layer(hidden,src_key_padding_mask=padding)
    hidden=model.encoder.norm(hidden)[:,2:]
    return {"positive_logit":model.presence_head(hidden,batch["axis"]),
            "amount_normalized":model.amount_head(hidden,batch["axis"])}
