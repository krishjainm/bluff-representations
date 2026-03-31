"""
Paper-style **control** interventions for causal sections.

Compares: baseline, probe steering, random direction, orthogonal-to-probe random,
mismatched activation patch (wrong prompt), optional wrong-layer shuffle (caller).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .causal_generation import CausalLMInterventionRunner, InterventionResult


def orthogonal_unit_to(
    reference: torch.Tensor, device: str, seed: Optional[int] = None
) -> torch.Tensor:
    """Unit vector approximately orthogonal to ``reference`` (high-dim)."""
    r = reference.flatten().float().to(device)
    r = r / r.norm().clamp(min=1e-12)
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)
    u = torch.randn(r.numel(), generator=g, device=device, dtype=r.dtype)
    u = u - u.dot(r) * r
    return u / u.norm().clamp(min=1e-12)


@dataclass
class ControlSuiteResult:
    condition: str
    result: InterventionResult
    meta: Dict[str, Any] = field(default_factory=dict)


def run_causal_control_suite(
    runner: CausalLMInterventionRunner,
    prompt: str,
    layer_idx: int,
    probe_steering_vector: torch.Tensor,
    mismatched_patch_vector: Optional[torch.Tensor] = None,
    strengths: Optional[List[float]] = None,
    max_new_tokens: int = 64,
    random_seed: int = 0,
) -> List[ControlSuiteResult]:
    """
    Run a small grid of conditions at a single (prompt, layer).

    ``mismatched_patch_vector`` should be a hidden state from an **unrelated**
    prompt (e.g. different base item) for the null control.
    """
    strengths = strengths or [0.0, 1.0]
    dim = probe_steering_vector.numel()
    out: List[ControlSuiteResult] = []

    out.append(
        ControlSuiteResult(
            "baseline",
            runner.generate(
                prompt, max_new_tokens=max_new_tokens, layer_idx=None, strength=0.0
            ),
        )
    )

    rnd = runner.random_direction_control(dim, seed=random_seed + 1)
    ortho = orthogonal_unit_to(probe_steering_vector, runner.device, seed=random_seed + 2)

    for name, vec in (
        ("random_direction", rnd),
        ("orthogonal_to_probe", ortho),
    ):
        for a in strengths:
            if a == 0:
                continue
            out.append(
                ControlSuiteResult(
                    f"{name}_strength_{a}",
                    runner.generate(
                        prompt,
                        max_new_tokens=max_new_tokens,
                        layer_idx=layer_idx,
                        steering_vector=vec,
                        strength=a,
                        steering_mode="add",
                    ),
                    meta={"control": name, "strength": a},
                )
            )

    for a in strengths:
        if a == 0:
            continue
        out.append(
            ControlSuiteResult(
                f"probe_steering_strength_{a}",
                runner.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    layer_idx=layer_idx,
                    steering_vector=probe_steering_vector,
                    strength=a,
                    steering_mode="add",
                ),
                meta={"control": "probe", "strength": a},
            )
        )

    for a in strengths:
        if a == 0:
            continue
        out.append(
            ControlSuiteResult(
                f"probe_steering_first_token_pos_strength_{a}",
                runner.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    layer_idx=layer_idx,
                    steering_vector=probe_steering_vector,
                    strength=a,
                    steering_mode="add",
                    token_position=0,
                ),
                meta={"control": "probe_wrong_token_pos", "token_position": 0, "strength": a},
            )
        )

    if mismatched_patch_vector is not None:
        v = mismatched_patch_vector.to(runner.device).flatten()
        out.append(
            ControlSuiteResult(
                "mismatched_patch_replace",
                runner.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    layer_idx=layer_idx,
                    steering_vector=v,
                    strength=1.0,
                    steering_mode="replace",
                ),
                meta={"control": "mismatched_patch"},
            )
        )

    return out


def control_suite_to_jsonable(rows: List[ControlSuiteResult]) -> List[Dict]:
    return [
        {
            "condition": r.condition,
            "text": r.result.text,
            "prompt": r.result.prompt[:500],
            "layer_idx": r.result.layer_idx,
            "strength": r.result.strength,
            "mode": r.result.mode,
            "meta": r.meta,
        }
        for r in rows
    ]
