"""
Map paper-style intervention / logging sites to concrete HF submodules.

Sites:
  - ``block``: full decoder/encoder block output (default).
  - ``attn``: self-attention module output (when exposed).
  - ``mlp``: feed-forward / MLP submodule output.
"""

from __future__ import annotations

import torch.nn as nn
def get_transformer_layers(model: nn.Module) -> Optional[nn.ModuleList]:
    """Stack of transformer blocks (shared by extractor + interventions)."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "encoder") and hasattr(model.encoder, "layer"):
        return model.encoder.layer
    if hasattr(model, "layers"):
        return model.layers
    return None


def get_decoder_block(model: nn.Module, layer_idx: int) -> nn.Module:
    layers = get_transformer_layers(model)
    if layers is None or layer_idx < 0 or layer_idx >= len(layers):
        raise ValueError(f"Invalid layer_idx {layer_idx} for this model")
    return layers[layer_idx]


def resolve_hook_module(
    model: nn.Module, layer_idx: int, site: str = "block"
) -> nn.Module:
    """
    Return the module to forward-hook for activation logging / interventions.

    Args:
        model: HF ``AutoModel`` or ``AutoModelForCausalLM`` (root).
        layer_idx: Index in the transformer stack.
        site: ``block`` | ``attn`` | ``mlp``
    """
    block = get_decoder_block(model, layer_idx)
    site = (site or "block").lower().strip()
    if site in ("block", "decoder", "layer", "decoder_layer"):
        return block
    if site in ("attn", "attention", "self_attn"):
        for name in ("self_attn", "attention", "attn"):
            if hasattr(block, name):
                return getattr(block, name)
        raise ValueError(
            f"No attention submodule found on layer {type(block).__name__}"
        )
    if site in ("mlp", "ffn", "feed_forward"):
        for name in ("mlp", "feed_forward", "ffn"):
            if hasattr(block, name):
                return getattr(block, name)
        raise ValueError(f"No MLP submodule found on layer {type(block).__name__}")
    raise ValueError(f"Unknown site {site!r}; use block|attn|mlp")
