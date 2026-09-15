"""
Generation-time causal interventions for paper experiments (steering / patching).

Uses forward hooks on a HuggingFace causal LM during `generate()`. Interventions
apply at the **last sequence position** each decoding step (standard for
next-token prediction).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from .activation_sites import get_transformer_layers, resolve_hook_module

logger = logging.getLogger(__name__)


@dataclass
class InterventionResult:
    text: str
    prompt: str
    mode: str
    layer_idx: int
    strength: float
    timing_policy: str = "every_decode_step"


def _hidden_from_layer_output(output) -> torch.Tensor:
    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state
    if isinstance(output, tuple):
        return output[0]
    return output


def _set_last_position_hidden(output, new_last: torch.Tensor):
    """Return new output with last position hidden states replaced."""
    h_full = _hidden_from_layer_output(output)
    h = h_full.clone()
    h[:, -1, :] = new_last
    if hasattr(output, "last_hidden_state"):
        # Some HF types are not easily mutable; return tuple for decoder layers
        raise TypeError("Unexpected output type with last_hidden_state in layer hook")
    if isinstance(output, tuple):
        return (h,) + tuple(output[1:])
    return h


class CausalLMInterventionRunner:
    """
    Run steering / patching on a causal LM during `generate()`.

    Controls (paper): use `random_direction_control()` for matched-norm noise.
    Hook ``intervention_site``: ``block`` | ``attn`` | ``mlp``.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        torch_dtype=None,
        intervention_site: str = "block",
    ):
        self.model_name = model_name
        self.device = device
        self.intervention_site = intervention_site
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {}
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _layers(self) -> nn.ModuleList:
        layers = get_transformer_layers(self.model)
        if layers is None:
            raise ValueError(f"No transformer layers found for {self.model_name}")
        return layers

    def _hook_module(self, layer_idx: int) -> nn.Module:
        return resolve_hook_module(self.model, layer_idx, self.intervention_site)

    def _encode(self, prompt: str) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.inference_mode()
    def last_hidden_at_layer(self, prompt: str, layer_idx: int) -> torch.Tensor:
        """
        Single forward: hidden state at last non-pad position for one layer.
        Returns vector [hidden_dim] (batch size 1).
        """
        inputs = self._encode(prompt)
        mask = inputs["attention_mask"]
        captured = {}

        def hook(module, inp, out):
            h = _hidden_from_layer_output(out)
            idx = mask.sum(dim=1) - 1
            b = torch.arange(h.size(0), device=h.device)
            captured["vec"] = h[b, idx, :].squeeze(0).detach()

        handle = self._hook_module(layer_idx).register_forward_hook(hook)
        try:
            self.model(**inputs)
        finally:
            handle.remove()
        return captured["vec"]

    def _make_hook(
        self,
        mode: str,
        vector: torch.Tensor,
        strength: float,
        token_position: int = -1,
        timing_policy: str = "every_decode_step",
    ) -> Callable:
        if timing_policy not in {"prefill_only", "first_decision_token", "every_decode_step"}:
            raise ValueError("timing_policy must be prefill_only, first_decision_token, or every_decode_step")
        v = vector.to(self.device).view(1, -1)
        calls = 0

        def hook(module, inp, out):
            nonlocal calls
            calls += 1
            if timing_policy in {"prefill_only", "first_decision_token"} and calls != 1:
                return out
            h = _hidden_from_layer_output(out)
            new_h = h.clone()
            seq_len = new_h.shape[1]
            pos = token_position if token_position >= 0 else seq_len + token_position
            pos = int(max(0, min(pos, seq_len - 1)))
            cur = new_h[:, pos, :]
            if mode == "add":
                new_pos = cur + strength * v
            elif mode == "replace":
                # Replacement is patching: it replaces, rather than scales or
                # subtracts from, the selected activation.
                new_pos = v.expand_as(cur)
            else:
                raise ValueError(f"Unknown mode {mode}")
            new_h[:, pos, :] = new_pos
            if isinstance(out, tuple):
                return (new_h,) + tuple(out[1:])
            return new_h

        return hook

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        temperature: float = 0.7,
        top_p: float = 0.95,
        layer_idx: Optional[int] = None,
        steering_vector: Optional[torch.Tensor] = None,
        steering_mode: str = "add",
        strength: float = 0.0,
        do_sample: bool = True,
        token_position: int = -1,
        timing_policy: str = "every_decode_step",
    ) -> InterventionResult:
        """
        Generate text; optionally steer at `layer_idx` with `steering_vector`.

        If layer_idx is None or strength == 0, runs plain generation.
        ``token_position``: index into sequence (-1 = last token each step).
        ``timing_policy``: apply to prefill/first decision only, or to every
        decode step. This policy is returned with the artifact record.
        """
        inputs = self._encode(prompt)
        handles = []
        if layer_idx is not None and steering_vector is not None and strength != 0:
            hfn = self._make_hook(
                steering_mode, steering_vector, strength, token_position=token_position
                , timing_policy=timing_policy
            )
            handles.append(self._hook_module(layer_idx).register_forward_hook(hfn))
        try:
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        finally:
            for h in handles:
                h.remove()
        text = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return InterventionResult(
            text=text,
            prompt=prompt,
            mode=steering_mode if handles else "none",
            layer_idx=layer_idx if handles else -1,
            strength=strength,
            timing_policy=timing_policy,
        )

    def random_direction_control(self, dim: int, seed: Optional[int] = None) -> torch.Tensor:
        g = torch.Generator(device=self.device)
        if seed is not None:
            g.manual_seed(seed)
        v = torch.randn(dim, generator=g, device=self.device)
        return v / v.norm().clamp(min=1e-12)

    def random_direction_matched_norm(
        self, reference: torch.Tensor, seed: Optional[int] = None
    ) -> torch.Tensor:
        """Random unit vector (same shape as reference flattened to hidden dim)."""
        ref = reference.flatten().float()
        dim = ref.numel()
        return self.random_direction_control(dim, seed=seed)

    def dose_response_steering_sweep(
        self,
        prompt: str,
        layer_idx: int,
        steering_vector: torch.Tensor,
        strengths: List[float],
        max_new_tokens: int = 48,
    ) -> List[InterventionResult]:
        """Run several strengths for a paper-style dose–response table."""
        results = []
        for a in strengths:
            results.append(
                self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    layer_idx=layer_idx,
                    steering_vector=steering_vector,
                    strength=a,
                    steering_mode="add",
                )
            )
        return results


def paired_patch_vector(
    runner: CausalLMInterventionRunner,
    prompt_truthful: str,
    prompt_deceptive: str,
    layer_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hidden states for patching experiments: returns (h_truth, h_decep) at last
    position. Use `replace` mode with h_decep on a truthful prompt to test injection.
    """
    t = runner.last_hidden_at_layer(prompt_truthful, layer_idx)
    d = runner.last_hidden_at_layer(prompt_deceptive, layer_idx)
    return t, d
