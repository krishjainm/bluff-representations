"""Behavioral causal intervention suite for the strict paper path.

The rule this module exists to enforce is the brief's ninth non-negotiable: if a
direction comes from a linear probe and the intervention adds that direction,
then measuring the *same probe's* score afterwards is circular. So the primary
endpoint here is always something the model itself emits -- the probability it
assigns to its own answer tokens -- and the probe score is carried alongside as a
clearly labelled diagnostic. :func:`assert_endpoint_independence` refuses to let
a probe-derived quantity be declared primary.

Terminology follows the brief exactly:

- **positive / negative steering** -- adding or subtracting a direction.
- **activation patching** -- *replacing* activations with activations from
  another run or example. Never used for addition or subtraction.

Model access is injected via :class:`InterventionRunner` so the whole suite is
testable without loading weights.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .behavioral_metrics import refusal_heuristic, repetition_score
from .paper import PaperConfig, ResearchIntegrityError

# Quantities derived from the probe, which may never be the primary endpoint.
PROBE_DERIVED_ENDPOINTS = frozenset({"probe_score", "probe_probability", "probe_logit"})

INTERVENTION_KINDS = ("none", "add", "replace")
TIMING_POLICIES = ("prefill_only", "first_decision_token", "every_decode_step")


# --------------------------------------------------------------------------- #
# Interventions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Intervention:
    """A fully specified, loggable intervention.

    ``kind='add'`` is steering; ``kind='replace'`` is activation patching. The
    distinction is enforced in :meth:`__post_init__` so a replacement can never
    be described as steering in the results.
    """

    kind: str = "none"
    vector: np.ndarray | None = None
    layer: int | None = None
    site: str = "block"
    token_index: int = -1
    timing_policy: str = "prefill_only"
    strength: float = 0.0
    # Provenance, carried into every result row.
    direction_name: str = "none"
    direction_source: str = "none"

    def __post_init__(self) -> None:
        if self.kind not in INTERVENTION_KINDS:
            raise ResearchIntegrityError(f"kind must be one of {list(INTERVENTION_KINDS)}")
        if self.timing_policy not in TIMING_POLICIES:
            raise ResearchIntegrityError(f"timing_policy must be one of {list(TIMING_POLICIES)}")
        if self.kind == "none":
            if self.vector is not None or self.strength:
                raise ResearchIntegrityError("A 'none' intervention must carry no vector or strength")
            return
        if self.vector is None:
            raise ResearchIntegrityError(f"kind={self.kind!r} requires a vector")
        if self.layer is None:
            raise ResearchIntegrityError(f"kind={self.kind!r} requires an explicit layer")
        if self.kind == "replace" and self.strength:
            raise ResearchIntegrityError(
                "Activation patching replaces the activation outright; a nonzero strength would "
                "make it a scaled steer. Use kind='add' for steering.")

    @property
    def is_steering(self) -> bool:
        return self.kind == "add"

    @property
    def is_patching(self) -> bool:
        return self.kind == "replace"

    def to_record(self) -> dict[str, Any]:
        return {
            "kind": self.kind, "layer": self.layer, "site": self.site,
            "token_index": self.token_index, "timing_policy": self.timing_policy,
            "strength": float(self.strength), "direction_name": self.direction_name,
            "direction_source": self.direction_source,
            "direction_norm": float(np.linalg.norm(self.vector)) if self.vector is not None else None,
        }


@dataclass(frozen=True)
class Direction:
    """A named unit-norm direction plus how it was obtained."""

    name: str
    vector: np.ndarray
    source: str
    layer: int
    seed: int | None = None
    notes: str = ""

    def to_record(self) -> dict[str, Any]:
        return {"name": self.name, "source": self.source, "layer": self.layer,
                "seed": self.seed, "notes": self.notes,
                "norm": float(np.linalg.norm(self.vector)), "dim": int(self.vector.shape[0])}


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ResearchIntegrityError("Cannot normalise a zero-norm direction")
    return np.asarray(vector, dtype=np.float64) / norm


def orthogonal_direction(reference: np.ndarray, *, seed: int) -> np.ndarray:
    """A unit vector orthogonal to ``reference`` (Gram-Schmidt on a random draw)."""
    reference = _unit(reference)
    rng = np.random.default_rng(seed)
    candidate = rng.normal(size=reference.shape[0])
    candidate = candidate - candidate.dot(reference) * reference
    return _unit(candidate)


def build_direction_set(
    activations: np.ndarray, df: pd.DataFrame, manifest: dict[str, Any],
    config: PaperConfig, *, layer: int, nuisance_column: str | None = None,
) -> dict[str, Direction]:
    """Construct the probe direction and every control direction, from train only.

    Fitting on the training partition alone matters: a direction derived with any
    held-out information would make the causal evaluation leak even if the
    endpoint itself is independent.
    """
    index = {sid: i for i, sid in enumerate(df.sample_id)}
    train_pos = np.array([index[s] for s in manifest["train"]])
    x_train = activations[train_pos, layer]
    y_train = df.label.to_numpy(int)[train_pos]
    if len(np.unique(y_train)) < 2:
        raise ResearchIntegrityError("Direction fitting needs both label classes in train")

    def fit(y: np.ndarray) -> np.ndarray:
        model = LogisticRegression(C=config.probe_c, class_weight="balanced", max_iter=5000,
                                   random_state=config.seed)
        model.fit(x_train, y)
        return model.coef_.ravel()

    raw_probe = fit(y_train)
    if float(np.linalg.norm(raw_probe)) < 1e-12:
        raise ResearchIntegrityError(
            f"The probe learned a zero-norm direction at layer {layer}: these activations carry no "
            "linearly decodable label signal, so there is no direction to steer along. Check the "
            "layer choice and that the activations are real rather than a placeholder.")
    probe_vector = _unit(raw_probe)
    rng = np.random.default_rng(config.seed)
    directions: dict[str, Direction] = {
        "probe": Direction("probe", probe_vector, "logistic probe weights, train partition only",
                           layer, config.seed),
        "random_matched_norm": Direction(
            "random_matched_norm", _unit(rng.normal(size=probe_vector.shape[0])),
            "random gaussian, unit norm (matched to probe by construction)", layer, config.seed,
            "tests whether any direction of this magnitude moves the endpoint"),
        "orthogonal": Direction(
            "orthogonal", orthogonal_direction(probe_vector, seed=config.seed + 1),
            "random vector orthogonalised against the probe direction", layer, config.seed + 1,
            "tests whether the effect is specific to the probe subspace"),
    }

    shuffled = rng.permutation(y_train)
    if len(np.unique(shuffled)) >= 2:
        directions["shuffled_label"] = Direction(
            "shuffled_label", _unit(fit(shuffled)),
            "probe refit on permuted training labels", layer, config.seed,
            "tests whether the fitting procedure alone produces an effective direction")

    if nuisance_column and nuisance_column in df.columns:
        values = df[nuisance_column].iloc[train_pos]
        # Binarise against the most common value so a categorical nuisance yields
        # a single comparable direction.
        if values.notna().any() and values.nunique(dropna=True) >= 2:
            majority = values.mode(dropna=True).iloc[0]
            nuisance_y = (values != majority).to_numpy(int)
            if len(np.unique(nuisance_y)) >= 2:
                directions[f"nuisance_{nuisance_column}"] = Direction(
                    f"nuisance_{nuisance_column}", _unit(fit(nuisance_y)),
                    f"probe fit to predict {nuisance_column} != {majority!r}", layer, config.seed,
                    "tests whether a strategy direction moves the endpoint as much as the probe")
    return directions


# --------------------------------------------------------------------------- #
# Behavioral endpoint
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ForcedChoiceEndpoint:
    """The model's own probability over its allowed answer tokens.

    This is independent of the probe: it reads the model's output head, not a
    fitted linear readout. ``valid_choice_mass`` doubles as the format-adherence
    quality control the brief asks for.
    """

    options: tuple[str, ...]
    positive_option: str
    name: str = "forced_choice_positive_probability"

    def __post_init__(self) -> None:
        if len(set(self.options)) < 2:
            raise ResearchIntegrityError("A forced-choice endpoint needs at least two distinct options")
        if self.positive_option not in self.options:
            raise ResearchIntegrityError("positive_option must be one of options")

    def score(self, logprobs: dict[str, float]) -> dict[str, Any]:
        missing = [o for o in self.options if o not in logprobs]
        if missing:
            raise ResearchIntegrityError(f"Runner did not return logprobs for options {missing}")
        values = np.array([logprobs[o] for o in self.options], dtype=np.float64)
        # Renormalise over the allowed options; the absolute mass they hold is
        # reported separately as a validity check.
        shifted = values - values.max()
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        chosen = int(np.argmax(values))
        return {
            "endpoint_name": self.name,
            "endpoint_value": float(probabilities[self.options.index(self.positive_option)]),
            "chosen_option": self.options[chosen],
            "chose_positive": bool(self.options[chosen] == self.positive_option),
            "valid_choice_mass": float(np.exp(values).sum()),
            "option_probabilities": {o: float(p) for o, p in zip(self.options, probabilities)},
        }


def assert_endpoint_independence(endpoint_name: str) -> None:
    """Refuse to treat a probe-derived quantity as the primary causal endpoint."""
    if endpoint_name in PROBE_DERIVED_ENDPOINTS:
        raise ResearchIntegrityError(
            f"{endpoint_name!r} is derived from the probe whose direction defines the "
            "intervention; using it as the primary endpoint would be circular. Report it as a "
            "diagnostic and choose a behavioral endpoint.")


@runtime_checkable
class InterventionRunner(Protocol):
    """Minimal model interface the causal suite needs."""

    def choice_logprobs(
        self, prompt: str, options: Sequence[str], *, intervention: Intervention
    ) -> dict[str, float]:
        """Log-probabilities of each option as the next answer, under ``intervention``."""

    def hidden_at(self, prompt: str, layer: int, *, site: str = "block") -> np.ndarray:
        """Hidden state at the final prompt token, used for patching and diagnostics."""


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Condition:
    """One experimental condition, resolvable into an :class:`Intervention`."""

    name: str
    kind: str
    direction_name: str | None = None
    layer_offset: int = 0
    token_index: int = -1
    timing_policy: str = "prefill_only"
    signed: int = 1
    patch_from_mismatched_prompt: bool = False
    notes: str = ""


def default_conditions(*, wrong_layer_offset: int = -4, wrong_token_index: int = 0) -> tuple[Condition, ...]:
    """The full control set the reviewers asked for.

    ``wrong_layer`` and ``wrong_token_position`` apply the *same* probe direction
    somewhere it should not work, which is what makes them controls rather than
    separate experiments.
    """
    return (
        Condition("baseline", "none", notes="no intervention"),
        Condition("positive_steering", "add", "probe", signed=+1,
                  notes="adds the probe direction"),
        Condition("negative_steering", "add", "probe", signed=-1,
                  notes="subtracts the probe direction"),
        Condition("random_matched_norm", "add", "random_matched_norm", signed=+1),
        Condition("orthogonal", "add", "orthogonal", signed=+1),
        Condition("shuffled_label", "add", "shuffled_label", signed=+1),
        Condition("wrong_layer", "add", "probe", layer_offset=wrong_layer_offset, signed=+1,
                  notes="probe direction applied at a different layer"),
        Condition("wrong_token_position", "add", "probe", token_index=wrong_token_index, signed=+1,
                  notes="probe direction applied away from the decision position"),
        Condition("activation_patch_mismatched_prompt", "replace", patch_from_mismatched_prompt=True,
                  notes="true patching: activation replaced with one from a different prompt"),
    )


def nuisance_condition(direction_name: str) -> Condition:
    return Condition(f"nuisance_steering_{direction_name}", "add", direction_name, signed=+1,
                     notes="a strategy direction, for comparison with the probe direction")


# --------------------------------------------------------------------------- #
# Paired statistics
# --------------------------------------------------------------------------- #

def paired_bootstrap_effect(
    baseline: Sequence[float], intervened: Sequence[float], groups: Sequence[Any] | None = None,
    *, seed: int, n_resamples: int = 1000,
) -> dict[str, Any]:
    """Paired mean difference with a grouped bootstrap CI and an effect size.

    Pairs are held together and *groups* are resampled, so clustered poker hands
    are not presented as independent observations.
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    intervened = np.asarray(intervened, dtype=np.float64)
    if baseline.shape != intervened.shape:
        raise ResearchIntegrityError("Paired effect requires equal-length baseline and intervened arrays")
    if len(baseline) == 0:
        raise ResearchIntegrityError("Paired effect requires at least one pair")
    differences = intervened - baseline
    keys = np.asarray(list(groups)) if groups is not None else np.arange(len(baseline))
    unique = np.unique(keys)
    rng = np.random.default_rng(seed)
    means: list[float] = []
    for _ in range(n_resamples):
        picked = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.flatnonzero(keys == key) for key in picked])
        means.append(float(differences[idx].mean()))
    spread = float(differences.std(ddof=1)) if len(differences) > 1 else 0.0
    return {
        "n_pairs": int(len(differences)), "n_groups": int(len(unique)),
        "baseline_mean": float(baseline.mean()), "intervened_mean": float(intervened.mean()),
        "mean_difference": float(differences.mean()),
        "ci_lower": float(np.quantile(means, .025)), "ci_upper": float(np.quantile(means, .975)),
        # Cohen's d_z for paired samples; None when the differences are constant.
        "effect_size_dz": float(differences.mean() / spread) if spread > 0 else None,
        "signed_consistency_rate": float(np.mean(np.sign(differences) == np.sign(differences.mean()))
                                         if differences.mean() != 0 else 0.0),
        "valid_resamples": len(means), "requested_resamples": n_resamples,
    }


# --------------------------------------------------------------------------- #
# Suite
# --------------------------------------------------------------------------- #

def _resolve(
    condition: Condition, directions: dict[str, Direction], *, layer: int, strength: float,
    patch_vector: np.ndarray | None,
) -> Intervention:
    if condition.kind == "none":
        return Intervention()
    if condition.kind == "replace":
        if patch_vector is None:
            raise ResearchIntegrityError(
                f"Condition {condition.name!r} is a patch but no source activation was supplied")
        return Intervention(kind="replace", vector=patch_vector, layer=layer, token_index=condition.token_index,
                            timing_policy=condition.timing_policy, strength=0.0,
                            direction_name="mismatched_prompt_activation",
                            direction_source="hidden state from a different held-out prompt")
    direction = directions.get(condition.direction_name or "")
    if direction is None:
        raise ResearchIntegrityError(
            f"Condition {condition.name!r} needs direction {condition.direction_name!r}, which was not built")
    target_layer = layer + condition.layer_offset
    if target_layer < 0:
        raise ResearchIntegrityError(
            f"Condition {condition.name!r} resolves to negative layer {target_layer}")
    return Intervention(kind="add", vector=condition.signed * direction.vector, layer=target_layer,
                        token_index=condition.token_index, timing_policy=condition.timing_policy,
                        strength=float(strength), direction_name=direction.name,
                        direction_source=direction.source)


def run_causal_suite(
    runner: InterventionRunner,
    prompts: pd.DataFrame,
    directions: dict[str, Direction],
    endpoint: ForcedChoiceEndpoint,
    *,
    layer: int,
    strengths: Sequence[float],
    conditions: Sequence[Condition] | None = None,
    seed: int = 2026,
    n_resamples: int = 1000,
    group_column: str = "split_group_id",
    probe_direction_name: str = "probe",
) -> dict[str, Any]:
    """Paired held-out causal evaluation across every condition and strength.

    ``prompts`` must be held-out rows carrying ``sample_id`` and ``statement``.
    Every condition is evaluated on the *same* prompts so comparisons are paired,
    and decoding is not involved at all for the forced-choice endpoint -- the
    endpoint is read from a single deterministic forward pass, which removes
    sampling noise from the primary comparison.
    """
    assert_endpoint_independence(endpoint.name)
    if "statement" not in prompts.columns or "sample_id" not in prompts.columns:
        raise ResearchIntegrityError("prompts must carry sample_id and statement")
    if prompts.empty:
        raise ResearchIntegrityError("Causal evaluation needs at least one held-out prompt")
    if not strengths:
        raise ResearchIntegrityError("Provide at least one intervention strength")
    conditions = tuple(conditions if conditions is not None else default_conditions())
    if not any(c.kind == "none" for c in conditions):
        raise ResearchIntegrityError("A baseline condition is required for paired comparison")

    statements = prompts["statement"].astype(str).tolist()
    sample_ids = prompts["sample_id"].astype(str).tolist()
    groups = (prompts[group_column].astype(str).tolist() if group_column in prompts.columns
              else list(sample_ids))
    # Patch sources come from a deterministic rotation of the same held-out set,
    # so a mismatched-prompt patch is a real activation from a real other prompt.
    patch_partner = [(i + 1) % len(statements) for i in range(len(statements))]

    probe_vector = directions[probe_direction_name].vector if probe_direction_name in directions else None
    rows: list[dict[str, Any]] = []
    for position, (sid, prompt) in enumerate(zip(sample_ids, statements)):
        hidden = runner.hidden_at(prompt, layer)
        hidden_norm = float(np.linalg.norm(hidden))
        patch_vector = runner.hidden_at(statements[patch_partner[position]], layer)
        for condition in conditions:
            # Neither a baseline nor a true patch has a dose: the baseline does
            # nothing and the patch replaces outright. Evaluating them once keeps
            # the pairing exact and avoids implying a strength grid they lack.
            grid = [0.0] if condition.kind in ("none", "replace") else list(strengths)
            for strength in grid:
                intervention = _resolve(condition, directions, layer=layer, strength=strength,
                                        patch_vector=patch_vector)
                scored = endpoint.score(
                    runner.choice_logprobs(prompt, endpoint.options, intervention=intervention))
                # Identity and dose fields go last so no spread can overwrite them.
                record = {
                    **scored, **intervention.to_record(),
                    "sample_id": sid, "group": groups[position], "condition": condition.name,
                    "condition_notes": condition.notes, "strength": float(strength),
                    "hidden_norm": hidden_norm,
                }
                if intervention.vector is not None and intervention.kind == "add":
                    # Magnitude relative to the hidden state it perturbs, which is
                    # the only interpretable unit for a steering strength.
                    record["relative_magnitude"] = float(
                        strength * np.linalg.norm(intervention.vector) / max(hidden_norm, 1e-12))
                if probe_vector is not None:
                    # Diagnostic only, and named so it cannot be mistaken for the endpoint.
                    record["diagnostic_probe_projection"] = float(hidden @ probe_vector)
                rows.append(record)

    table = pd.DataFrame(rows)
    baseline = table[table.condition == "baseline"].set_index("sample_id")
    effects: dict[str, Any] = {}
    for condition in conditions:
        if condition.kind == "none":
            continue
        per_strength: dict[str, Any] = {}
        subset = table[table.condition == condition.name]
        for strength, block in subset.groupby("strength"):
            aligned = block.set_index("sample_id")
            shared = [s for s in aligned.index if s in baseline.index]
            if not shared:
                continue
            per_strength[f"{float(strength):g}"] = {
                **paired_bootstrap_effect(
                    baseline.loc[shared, "endpoint_value"].to_numpy(),
                    aligned.loc[shared, "endpoint_value"].to_numpy(),
                    aligned.loc[shared, "group"].to_numpy(),
                    seed=seed, n_resamples=n_resamples),
                "baseline_chose_positive_rate": float(baseline.loc[shared, "chose_positive"].mean()),
                "intervened_chose_positive_rate": float(aligned.loc[shared, "chose_positive"].mean()),
                "mean_relative_magnitude": (float(aligned.loc[shared, "relative_magnitude"].mean())
                                            if "relative_magnitude" in aligned else None),
            }
        effects[condition.name] = {"notes": condition.notes, "by_strength": per_strength}

    return {
        "primary_endpoint": endpoint.name,
        "primary_endpoint_is_probe_derived": False,
        "probe_score_role": "diagnostic only; never the primary endpoint",
        "decoding": "deterministic single forward pass (no sampling) for the forced-choice endpoint",
        "layer": int(layer),
        "n_prompts": int(len(statements)),
        "n_groups": int(len(set(groups))),
        "strengths": [float(s) for s in strengths],
        "conditions": [c.name for c in conditions],
        "directions": {name: d.to_record() for name, d in directions.items()},
        "effects": effects,
        "quality_controls": summarize_quality(table),
        "records": rows,
    }


def summarize_quality(table: pd.DataFrame) -> dict[str, Any]:
    """Format-adherence and choice-validity metrics, reported beside the effects.

    An intervention that "works" by destroying the model's ability to answer in
    the required format is a degradation, not a behavioral effect, so these sit
    next to the causal numbers rather than in an appendix.
    """
    summary: dict[str, Any] = {}
    for condition, block in table.groupby("condition"):
        by_strength: dict[str, Any] = {}
        for strength, rows in block.groupby("strength"):
            by_strength[f"{float(strength):g}"] = {
                "n": int(len(rows)),
                "mean_valid_choice_mass": float(rows["valid_choice_mass"].mean()),
                "min_valid_choice_mass": float(rows["valid_choice_mass"].min()),
                "chose_positive_rate": float(rows["chose_positive"].mean()),
            }
        summary[str(condition)] = by_strength
    return summary


def score_generation_quality(text: str) -> dict[str, Any]:
    """Cheap deterministic text-quality checks for free-form conditions.

    Reuses the existing behavioral metrics rather than reimplementing them; no
    paid API and no LLM judge is involved.
    """
    stripped = str(text).strip()
    return {
        "length_chars": len(stripped), "n_tokens": len(stripped.split()),
        "repetition_diversity": repetition_score(stripped),
        "refusal_like": bool(refusal_heuristic(stripped)),
        "empty": not stripped,
    }


class TransformersInterventionRunner:
    """Real :class:`InterventionRunner` backed by a Hugging Face causal LM.

    Constructing this loads weights, so it is never instantiated by imports,
    tests, or the audit path.

    The forced-choice endpoint is read from the **first-token** logits of each
    option after a single deterministic forward pass. No sampling is involved, so
    the paired baseline/intervened comparison carries no decoding noise. Options
    whose first token collides are rejected rather than silently conflated.
    """

    def __init__(self, model_name: str, *, device: str = "cpu", site: str = "block",
                 revision: str | None = None, torch_dtype: str = "float32") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer  # local: heavy

        self._torch = torch
        self.model_name = model_name
        self.device = device
        self.site = site
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, dtype=getattr(torch, torch_dtype, torch.float32))
        self.model.to(device)
        self.model.eval()

    def _first_token_ids(self, options: Sequence[str]) -> dict[str, int]:
        ids: dict[str, int] = {}
        for option in options:
            encoded = self.tokenizer.encode(option, add_special_tokens=False)
            if not encoded:
                raise ResearchIntegrityError(f"Option {option!r} tokenizes to nothing")
            ids[option] = int(encoded[0])
        if len(set(ids.values())) != len(ids):
            raise ResearchIntegrityError(
                f"Options {list(options)} do not have distinct first tokens under this tokenizer; "
                "choose options the model can actually distinguish in one step")
        return ids

    def _hook(self, intervention: Intervention):
        torch = self._torch

        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            updated = hidden.clone()
            length = updated.shape[1]
            position = intervention.token_index
            position = position if position >= 0 else length + position
            position = int(max(0, min(position, length - 1)))
            vector = torch.as_tensor(intervention.vector, dtype=updated.dtype, device=updated.device)
            if intervention.kind == "add":
                updated[:, position, :] = updated[:, position, :] + intervention.strength * vector
            elif intervention.kind == "replace":
                # Replacement, not a scaled steer: the activation is overwritten.
                updated[:, position, :] = vector.expand_as(updated[:, position, :])
            return (updated,) + tuple(output[1:]) if isinstance(output, tuple) else updated

        return hook

    def _run(self, prompt: str, intervention: Intervention):
        from .activation_sites import resolve_hook_module

        torch = self._torch
        encoded = self.tokenizer(prompt, return_tensors="pt")
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        handles = []
        if intervention.kind != "none":
            module = resolve_hook_module(self.model, int(intervention.layer), intervention.site or self.site)
            handles.append(module.register_forward_hook(self._hook(intervention)))
        try:
            with torch.no_grad():
                return self.model(**encoded, output_hidden_states=True)
        finally:
            for handle in handles:
                handle.remove()

    def choice_logprobs(
        self, prompt: str, options: Sequence[str], *, intervention: Intervention
    ) -> dict[str, float]:
        torch = self._torch
        outputs = self._run(prompt, intervention)
        logprobs = torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)
        return {option: float(logprobs[token_id])
                for option, token_id in self._first_token_ids(options).items()}

    def hidden_at(self, prompt: str, layer: int, *, site: str = "block") -> np.ndarray:
        outputs = self._run(prompt, Intervention())
        # hidden_states[0] is the embedding output, so index i is block i.
        return outputs.hidden_states[layer + 1][0, -1, :].float().cpu().numpy()


__all__ = [
    "Condition", "Direction", "ForcedChoiceEndpoint", "Intervention", "InterventionRunner",
    "PROBE_DERIVED_ENDPOINTS", "assert_endpoint_independence", "build_direction_set",
    "default_conditions", "nuisance_condition", "orthogonal_direction",
    "TransformersInterventionRunner", "paired_bootstrap_effect", "run_causal_suite",
    "score_generation_quality",
    "summarize_quality",
]
