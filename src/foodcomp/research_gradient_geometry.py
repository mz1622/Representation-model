"""Finite, explicitly defined local gradient geometry; no optimization policy."""
import torch


def gradient_geometry(first,second):
    if not first or len(first)!=len(second):raise ValueError("Matching nonempty gradient lists required.")
    dot=norm_a=norm_b=0.
    for a,b in zip(first,second):
        if a.shape!=b.shape:raise ValueError("Gradient shapes differ.")
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise FloatingPointError("Nonfinite task gradient.")
        a=a.double();b=b.double()
        dot+=float((a*b).sum());norm_a+=float(a.square().sum());norm_b+=float(b.square().sum())
    norm_a=norm_a**.5;norm_b=norm_b**.5
    defined=norm_a>0 and norm_b>0
    return {"dot":dot,"completion_norm":norm_a,"name_only_norm":norm_b,"both_nonzero":defined,
        "cosine":dot/(norm_a*norm_b) if defined else None,
        "name_to_completion_norm":norm_b/norm_a if norm_a>0 else None}
