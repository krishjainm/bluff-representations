"""
Extract **steering directions** from trained sparse autoencoder checkpoints.

Checkpoints are ``state_dict`` files (as saved by ``DeceptionTrainingPipeline``).
Optional sidecar ``*.meta.json`` (written on save) disambiguates hyperparameters;
otherwise dimensions are **inferred** from tensor shapes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from .sparse_autoencoder import DeceptionSparseAutoencoder, SupervisedDeceptionAutoencoder


def infer_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    """
    Returns (bottleneck_dim, input_dim) from ``encoder.weight`` shape
    ``[bottleneck_dim, input_dim]``.
    """
    w = state_dict["encoder.weight"]
    if w.dim() != 2:
        raise ValueError("encoder.weight must be 2D")
    bottleneck_dim, input_dim = int(w.shape[0]), int(w.shape[1])
    return bottleneck_dim, input_dim


def is_supervised_state_dict(state_dict: Dict[str, torch.Tensor]) -> bool:
    return "classifier.weight" in state_dict


def load_sae_meta(checkpoint_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    p = Path(checkpoint_path)
    meta_path = p.with_suffix(".meta.json")
    if not meta_path.exists():
        meta_path = p.parent / (p.stem + ".meta.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def build_autoencoder_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    device: str = "cpu",
    meta: Optional[Dict[str, Any]] = None,
) -> nn.Module:
    """Reconstruct ``DeceptionSparseAutoencoder`` or ``SupervisedDeceptionAutoencoder``."""
    bottleneck_dim, input_dim = infer_dims_from_state_dict(state_dict)
    supervised = is_supervised_state_dict(state_dict)
    if meta:
        tied_weights = bool(meta.get("tied_weights", True))
        activation_type = str(meta.get("activation_type", "ReLU"))
        topk_percent = int(meta.get("topk_percent", 10))
        dropout = float(meta.get("dropout", 0.0))
        supervised = bool(meta.get("supervised", supervised))
    else:
        tied_weights, activation_type, topk_percent, dropout = True, "ReLU", 10, 0.0

    if supervised:
        model = SupervisedDeceptionAutoencoder(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            tied_weights=tied_weights,
            activation_type=activation_type,
            topk_percent=topk_percent,
            dropout=dropout,
        )
    else:
        model = DeceptionSparseAutoencoder(
            input_dim=input_dim,
            bottleneck_dim=bottleneck_dim,
            tied_weights=tied_weights,
            activation_type=activation_type,
            topk_percent=topk_percent,
            dropout=dropout,
        )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def decoder_column_steering_vector(
    state_dict: Dict[str, torch.Tensor],
    feature_index: int,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Direction in **activation space** for SAE latent dimension ``feature_index``.

    For ``nn.Linear`` decoder, column ``j`` maps latent e_j to ``decoder.weight[j]``
    in the sense: output += h_j * decoder.weight[j, :] — we use **row** ``j`` of
    decoder weight when decoder is ``Linear(bottleneck -> input)``, i.e.
    ``decoder.weight`` is ``[input_dim, bottleneck]``, column ``j`` is
    ``decoder.weight[:, j]``.
    """
    w = state_dict["decoder.weight"]
    if w.dim() != 2:
        raise ValueError("decoder.weight must be 2D")
    input_dim, bottleneck_dim = int(w.shape[0]), int(w.shape[1])
    fi = int(feature_index)
    if fi < 0 or fi >= bottleneck_dim:
        raise IndexError(f"feature_index {fi} out of range [0, {bottleneck_dim})")
    vec = w[:, fi].detach().float().clone()
    if normalize:
        vec = vec / vec.norm().clamp(min=1e-12)
    return vec


def load_checkpoint_steering_vector(
    checkpoint_path: Union[str, Path],
    feature_index: int,
    device: str = "cpu",
    normalize: bool = True,
) -> torch.Tensor:
    """Load ``.pt`` state dict and return unit steering vector for one SAE feature."""
    path = Path(checkpoint_path)
    sd = torch.load(path, map_location=device)
    if not isinstance(sd, dict):
        raise ValueError("Checkpoint must be a state_dict mapping")
    return decoder_column_steering_vector(sd, feature_index, normalize=normalize).to(
        device
    )


@torch.inference_mode()
def rank_features_by_label_correlation(
    model: nn.Module,
    activations: torch.Tensor,
    labels: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Rank SAE latent indices by |Pearson correlation| with binary ``labels``.

    Returns:
        sorted_indices (descending by |corr|), correlation_values[sorted_indices]
    """
    model.eval()
    activations = activations.to(next(model.parameters()).device)
    labels = labels.float().to(activations.device).view(-1)
    if isinstance(model, SupervisedDeceptionAutoencoder):
        _, h_act, _, _ = model(activations)
    else:
        _, h_act, _ = model(activations)
    h = h_act.float()
    y = labels
    h_mean = h.mean(dim=0, keepdim=True)
    y_mean = y.mean()
    hc = h - h_mean
    yc = y - y_mean
    denom_h = hc.std(dim=0).clamp(min=1e-8)
    denom_y = yc.std().clamp(min=1e-8)
    corrs = (hc * yc.unsqueeze(1)).mean(dim=0) / (denom_h * denom_y)
    abs_c = corrs.abs()
    order = torch.argsort(abs_c, descending=True)
    sorted_corrs = corrs[order]
    return order, sorted_corrs


def suggest_deception_feature_indices(
    checkpoint_path: Union[str, Path],
    activations: torch.Tensor,
    labels: torch.Tensor,
    device: str = "cpu",
    top_k: int = 10,
) -> List[Dict[str, float]]:
    """
    Load SAE from checkpoint, rank latents vs labels, return top-``k`` as dicts
    ``{"feature": i, "correlation": float}``.
    """
    path = Path(checkpoint_path)
    sd = torch.load(path, map_location=device)
    meta = load_sae_meta(path)
    model = build_autoencoder_from_state_dict(sd, device=device, meta=meta)
    order, sorted_corrs = rank_features_by_label_correlation(
        model, activations, labels
    )
    out = []
    for i in range(min(top_k, len(order))):
        idx = int(order[i].item())
        out.append({"feature": idx, "correlation": float(sorted_corrs[i].item())})
    return out
