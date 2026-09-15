"""Leakage-safe, provenance-tracked, resumable activation extraction.

This module is part of the strict paper path and shares its rules: it never
fabricates an activation, it records exactly how every tensor was produced, and
it refuses to mix artifacts that came from different extraction settings.

The central design decision is that leakage safety is *structural*.  The
pre-decision modes never receive the ``response`` column at all (see
:func:`render_extraction_prompt`), so a pre-decision probe cannot read answer
tokens even if a caller asks it to.  Response-token extraction still exists, but
only as an explicitly opted-in diagnostic that is stamped into the manifest.

Model access is injected through :class:`HiddenStateProvider` so that tests and
audits can run without downloading weights or contacting a network.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np
import pandas as pd
import torch

from .paper import ResearchIntegrityError

# Modes whose prompt is built strictly before the model's answer exists.  Nothing
# in this set may ever see the dataset's ``response`` column.
PRE_DECISION_MODES = ("prompt_end", "decision_token_prelogit", "mean_prompt")
# Explicitly diagnostic: reads the recorded response text.  Never the paper primary.
DIAGNOSTIC_MODES = ("response_token",)
ACTIVATION_MODES = PRE_DECISION_MODES + DIAGNOSTIC_MODES
# Weight dtypes for loading the subject model. Activations are always stored as
# float32 regardless, so this trades model memory and speed, not artifact precision.
TORCH_DTYPES = ("float32", "float16", "bfloat16")

DEFAULT_PREDECISION_TEMPLATE = "{statement}\nAction:"
DEFAULT_RESPONSE_TEMPLATE = "{statement}\nAction:{response}"

SPEC_FILENAME = "extraction_spec.json"
JOURNAL_FILENAME = "extraction_manifest.jsonl"
MANIFEST_FILENAME = "extraction_manifest.json"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExtractionSpec:
    """Every choice that changes the numbers in an activation artifact.

    Two runs whose specs share a :meth:`fingerprint` are interchangeable; two
    runs whose specs differ must not write into the same directory.
    """

    subject_model: str
    activation_mode: str = "prompt_end"
    activation_site: str = "block"
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    layer_indices: tuple[int, ...] | None = None
    max_sequence_length: int | None = None
    prompt_template_id: str = "unspecified"
    prompt_template: str = DEFAULT_PREDECISION_TEMPLATE
    # The token the prompt must end at, so the read position is unambiguous.
    # "Action:" suits a decision prompt; "Answer:" suits a forced-choice question.
    decision_boundary_marker: str = "Action:"
    torch_dtype: str = "float32"
    # Must be True to run a ``response_token`` extraction.  Recorded so that an
    # auditor can tell a diagnostic artifact from a pre-decision one.
    allow_response_leakage: bool = False

    def __post_init__(self) -> None:
        if self.activation_mode not in ACTIVATION_MODES:
            raise ResearchIntegrityError(
                f"activation_mode must be one of {list(ACTIVATION_MODES)}, got {self.activation_mode!r}"
            )
        if not str(self.subject_model).strip() or "REPLACE" in str(self.subject_model):
            raise ResearchIntegrityError(
                "subject_model must be a real model identifier before extraction"
            )
        if "{statement}" not in self.prompt_template:
            raise ResearchIntegrityError("prompt_template must contain {statement}")
        if self.activation_mode in PRE_DECISION_MODES:
            if "{response}" in self.prompt_template:
                raise ResearchIntegrityError(
                    f"activation_mode={self.activation_mode!r} is a pre-decision mode; "
                    "its prompt template must not reference {response}"
                )
            if self.allow_response_leakage:
                raise ResearchIntegrityError(
                    "allow_response_leakage must be False for pre-decision modes"
                )
        if not str(self.decision_boundary_marker).strip():
            raise ResearchIntegrityError("decision_boundary_marker must be a non-empty string")
        if self.activation_mode in ("prompt_end", "decision_token_prelogit"):
            # The real requirement is that the *rendered* prompt ends at the
            # boundary, which is checked per row in render_extraction_prompt.
            # Here we can only check a literal template suffix: if the template
            # appends its own text after the statement, that text must end at the
            # boundary. A template ending in {statement} is allowed, because the
            # boundary then comes from the data (e.g. a prompt whose final line is
            # already the stated action) and only the render check can verify it.
            suffix = self.prompt_template.rsplit("}", 1)[-1]
            if suffix.strip() and not suffix.rstrip().endswith(self.decision_boundary_marker):
                raise ResearchIntegrityError(
                    "prompt_end/decision_token_prelogit templates that append text after the "
                    f"statement must end at the explicit decision boundary "
                    f"{self.decision_boundary_marker!r}; got suffix {suffix!r}"
                )
        if self.activation_mode in DIAGNOSTIC_MODES:
            if not self.allow_response_leakage:
                raise ResearchIntegrityError(
                    f"activation_mode={self.activation_mode!r} reads response text; pass "
                    "allow_response_leakage=True to acknowledge it is a diagnostic only"
                )
            if "{response}" not in self.prompt_template:
                raise ResearchIntegrityError(
                    "response_token template must reference {response}"
                )
            before, _, _ = self.prompt_template.partition("{response}")
            if not before.rstrip().endswith(self.decision_boundary_marker):
                raise ResearchIntegrityError(
                    "response_token template must place {response} immediately after the "
                    f"decision boundary {self.decision_boundary_marker!r}"
                )
        if self.torch_dtype not in TORCH_DTYPES:
            raise ResearchIntegrityError(
                f"torch_dtype must be one of {list(TORCH_DTYPES)}, got {self.torch_dtype!r}")
        if self.max_sequence_length is not None and self.max_sequence_length < 1:
            raise ResearchIntegrityError("max_sequence_length must be >= 1 when set")
        if self.layer_indices is not None:
            if len(self.layer_indices) == 0:
                raise ResearchIntegrityError("layer_indices must be non-empty when set")
            if list(self.layer_indices) != sorted(set(self.layer_indices)):
                raise ResearchIntegrityError("layer_indices must be sorted and unique")
            if any(int(i) < 0 for i in self.layer_indices):
                raise ResearchIntegrityError("layer_indices must be non-negative")

    @property
    def is_pre_decision(self) -> bool:
        return self.activation_mode in PRE_DECISION_MODES

    def to_dict(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["layer_indices"] = None if self.layer_indices is None else list(self.layer_indices)
        return raw

    def fingerprint(self) -> str:
        """SHA-256 over the canonical spec JSON."""
        return _sha256_text(json.dumps(self.to_dict(), sort_keys=True))

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExtractionSpec":
        data = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        layers = data.get("layer_indices")
        if layers is not None:
            data["layer_indices"] = tuple(int(i) for i in layers)
        return cls(**data)

    @classmethod
    def from_paper_config(cls, config: Any, **overrides: Any) -> "ExtractionSpec":
        """Build a spec from a :class:`~deception_circuits.paper.PaperConfig`."""
        fields = paper_config_spec_fields(config)
        fields.update(overrides)
        return cls(**fields)


def paper_config_spec_fields(config: Any) -> dict[str, Any]:
    """Map a :class:`~deception_circuits.paper.PaperConfig` onto spec fields.

    Kept separate from :class:`ExtractionSpec` construction so that the
    config-versus-manifest comparison in :func:`verify_extraction_manifest` can
    report *which* field disagrees, instead of tripping over spec validation of
    the disagreement itself.  The mapping is explicit so that config fields which
    used to be inert documentation now constrain a real extraction run.
    """
    site = str(getattr(config, "activation_site", "block"))
    # ``residual_stream`` is the paper's wording for the decoder block output.
    site = "block" if site in ("residual_stream", "resid", "residual") else site
    layers = getattr(config, "activation_layers", "all")
    layer_indices: tuple[int, ...] | None
    if layers is None or (isinstance(layers, str) and layers.strip().lower() == "all"):
        layer_indices = None
    elif isinstance(layers, str):
        layer_indices = tuple(sorted({int(p) for p in layers.replace(",", " ").split()}))
    else:
        layer_indices = tuple(sorted({int(p) for p in layers}))
    return {
        "subject_model": str(getattr(config, "subject_model", "")),
        "activation_mode": str(getattr(config, "activation_mode", "prompt_end")),
        "activation_site": site,
        "model_revision": getattr(config, "model_revision", None),
        "tokenizer_revision": getattr(config, "tokenizer_revision", None),
        "layer_indices": layer_indices,
        "max_sequence_length": getattr(config, "max_sequence_length", None),
        "prompt_template_id": str(getattr(config, "prompt_template_id", "unspecified")),
        "decision_boundary_marker": str(getattr(config, "decision_boundary_marker", "Action:")),
        "torch_dtype": str(getattr(config, "torch_dtype", "float32")),
    }


@dataclass(frozen=True)
class PromptRendering:
    """A rendered extraction prompt and the exact token it will be read at."""

    text: str
    mode: str
    # Index into the *unpadded* token sequence.  ``None`` for pooled modes.
    token_index: int | None
    n_tokens: int
    truncated: bool
    # Inclusive-exclusive token span of the prompt (pre-decision) region.
    prompt_span: tuple[int, int]
    # Kept in memory only so extraction tokenizes once; never written to disk,
    # because artifacts must not carry private dataset text.
    token_ids: tuple[int, ...] = ()

    def to_record(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "token_index": self.token_index,
            "n_prompt_tokens": self.n_tokens,
            "truncated": self.truncated,
            "prompt_span": list(self.prompt_span),
            "prompt_sha256": _sha256_text(self.text),
        }


@runtime_checkable
class HiddenStateProvider(Protocol):
    """Minimal model interface needed for extraction.

    Implementations must not pad: extraction is per-sample so that the recorded
    ``token_index`` refers unambiguously to a real token.
    """

    def encode(self, text: str) -> list[int]:
        """Return token ids for ``text`` with no padding and no truncation."""

    def hidden_states(self, token_ids: Sequence[int], site: str) -> torch.Tensor:
        """Return a ``[layers, seq, hidden]`` tensor for one unpadded sequence."""


def _row_value(row: pd.Series, column: str) -> str:
    if column not in row.index:
        raise ResearchIntegrityError(f"Dataset row is missing required column {column!r}")
    value = row[column]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        raise ResearchIntegrityError(f"Column {column!r} is empty for this row")
    return str(value)


def render_extraction_prompt(
    row: pd.Series, spec: ExtractionSpec, provider: HiddenStateProvider
) -> PromptRendering:
    """Render the prompt for ``row`` and resolve the exact read position.

    For every mode in :data:`PRE_DECISION_MODES` the row is projected down to the
    columns the template may legitimately use *before* formatting, so response
    text is not merely unused — it is not present.  A template that tries to
    reference it fails at :class:`ExtractionSpec` construction, and a row whose
    statement is missing fails here rather than producing a silent artifact.

    Caveat for the ``response_token`` diagnostic: the prompt/response boundary is
    located by tokenizing the prompt prefix separately, so a tokenizer that merges
    characters across the boundary can shift ``prompt_span`` by a token.  The read
    position itself (the final token) is unaffected, and the pre-decision modes do
    not depend on this at all because they never concatenate a response.
    """
    statement = _row_value(row, "statement")

    if spec.is_pre_decision:
        prompt = spec.prompt_template.format(statement=statement)
        prompt_tokens = provider.encode(prompt)
        if not prompt_tokens:
            raise ResearchIntegrityError("Rendered pre-decision prompt tokenized to zero tokens")
        full_tokens = prompt_tokens
        prompt_len = len(prompt_tokens)
    else:
        response = _row_value(row, "response")
        prompt_only = spec.prompt_template.split("{response}")[0].format(statement=statement)
        prompt = spec.prompt_template.format(statement=statement, response=response)
        prompt_len = len(provider.encode(prompt_only))
        full_tokens = provider.encode(prompt)
        if prompt_len >= len(full_tokens):
            raise ResearchIntegrityError(
                "response_token mode requires at least one token after the prompt boundary"
            )

    limit = spec.max_sequence_length
    truncated = limit is not None and len(full_tokens) > limit
    n_tokens = min(len(full_tokens), limit) if limit is not None else len(full_tokens)

    if spec.activation_mode in ("prompt_end", "decision_token_prelogit"):
        # Both read the final prompt token: the hidden state that the model would
        # use to emit its first action token.  They differ only in intent, so the
        # mode is kept in the manifest for provenance.
        #
        # Per-row boundary check: this is the guarantee that matters, and it is
        # stronger than inspecting the template, because it also catches a row
        # whose own text does not end where the spec says the decision is.
        if not prompt.rstrip().endswith(spec.decision_boundary_marker):
            raise ResearchIntegrityError(
                f"Rendered pre-decision prompt does not end at the declared decision boundary "
                f"{spec.decision_boundary_marker!r}; it ends with "
                f"{prompt.rstrip()[-40:]!r}. The read position would not be the decision point."
            )
        if truncated:
            raise ResearchIntegrityError(
                f"Pre-decision prompt has {len(full_tokens)} tokens but max_sequence_length is "
                f"{limit}; truncation would move the decision boundary. Raise the limit or drop the row."
            )
        token_index: int | None = n_tokens - 1
        span = (0, n_tokens)
    elif spec.activation_mode == "mean_prompt":
        token_index = None
        span = (0, min(prompt_len, n_tokens))
    else:  # response_token
        if prompt_len >= n_tokens:
            raise ResearchIntegrityError(
                "Truncation removed every response token; cannot run response_token diagnostic"
            )
        token_index = n_tokens - 1
        span = (0, prompt_len)

    return PromptRendering(
        text=prompt, mode=spec.activation_mode, token_index=token_index,
        n_tokens=n_tokens, truncated=truncated, prompt_span=span,
        token_ids=tuple(int(t) for t in full_tokens[:n_tokens]),
    )


def _select_layers(hidden: torch.Tensor, spec: ExtractionSpec) -> tuple[torch.Tensor, list[int]]:
    total = hidden.shape[0]
    if spec.layer_indices is None:
        return hidden, list(range(total))
    invalid = [i for i in spec.layer_indices if i >= total]
    if invalid:
        raise ResearchIntegrityError(
            f"Configured layer_indices {invalid} exceed the model's {total} available layers"
        )
    idx = list(spec.layer_indices)
    return hidden[idx], idx


def extract_sample(
    row: pd.Series, spec: ExtractionSpec, provider: HiddenStateProvider
) -> tuple[torch.Tensor, PromptRendering, list[int]]:
    """Extract one ``[layers, hidden]`` activation and its provenance."""
    rendering = render_extraction_prompt(row, spec, provider)
    token_ids = rendering.token_ids
    hidden = provider.hidden_states(token_ids, spec.activation_site)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise ResearchIntegrityError(
            "HiddenStateProvider.hidden_states must return a [layers, seq, hidden] tensor"
        )
    if hidden.shape[1] != len(token_ids):
        raise ResearchIntegrityError(
            f"Provider returned {hidden.shape[1]} positions for {len(token_ids)} tokens"
        )
    selected, layer_indices = _select_layers(hidden, spec)
    if rendering.token_index is None:
        start, stop = rendering.prompt_span
        if stop <= start:
            raise ResearchIntegrityError("mean_prompt pooling received an empty prompt span")
        pooled = selected[:, start:stop, :].mean(dim=1)
    else:
        if not 0 <= rendering.token_index < selected.shape[1]:
            raise ResearchIntegrityError(
                f"Resolved token_index {rendering.token_index} is outside the {selected.shape[1]}-token sequence"
            )
        pooled = selected[:, rendering.token_index, :]
    pooled = pooled.detach().to(torch.float32).cpu()
    if not torch.isfinite(pooled).all():
        raise ResearchIntegrityError("Extracted activation contains NaN or Inf")
    return pooled, rendering, layer_indices


def _read_spec(directory: Path) -> tuple[ExtractionSpec, str] | None:
    path = directory / SPEC_FILENAME
    if not path.is_file():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ExtractionSpec.from_dict(raw["spec"]), str(raw.get("fingerprint", ""))


def _read_journal(directory: Path) -> dict[str, dict[str, Any]]:
    path = directory / JOURNAL_FILENAME
    if not path.is_file():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        records[str(record["sample_id"])] = record
    return records


def assert_spec_compatible(directory: str | Path, spec: ExtractionSpec) -> None:
    """Refuse to reuse a directory whose recorded spec differs from ``spec``."""
    existing = _read_spec(Path(directory))
    if existing is None:
        return
    recorded, fingerprint = existing
    if fingerprint != spec.fingerprint():
        raise ResearchIntegrityError(
            f"Activation directory {directory} was produced with a different extraction spec "
            f"(recorded fingerprint {fingerprint[:12]}, requested {spec.fingerprint()[:12]}). "
            "Choose a new activation_dir rather than mixing artifacts. Recorded spec: "
            f"{json.dumps(recorded.to_dict(), sort_keys=True)}"
        )


def run_extraction(
    df: pd.DataFrame,
    spec: ExtractionSpec,
    output_dir: str | Path,
    provider: HiddenStateProvider,
    *,
    resume: bool = True,
    progress_every: int = 0,
) -> dict[str, Any]:
    """Extract one activation per dataset row, resumably.

    Artifacts are written as ``sample_<sample_id>.pt`` so that
    :func:`deception_circuits.paper.load_activations` can consume them, and each
    tensor's provenance is appended to a JSONL journal immediately after the
    tensor lands on disk.  An interrupted run therefore resumes sample-wise.

    The provider is injected; this function never loads a model itself.
    """
    if "sample_id" not in df.columns:
        raise ResearchIntegrityError("Dataset must contain sample_id before extraction")
    if not df["sample_id"].is_unique:
        raise ResearchIntegrityError("sample_id values must be unique before extraction")

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    assert_spec_compatible(root, spec)
    (root / SPEC_FILENAME).write_text(
        json.dumps({"fingerprint": spec.fingerprint(), "spec": spec.to_dict()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    journal = _read_journal(root) if resume else {}
    journal_path = root / JOURNAL_FILENAME
    if not resume and journal_path.exists():
        journal_path.unlink()

    records: dict[str, dict[str, Any]] = {}
    n_extracted = 0
    n_resumed = 0
    for position, (_, row) in enumerate(df.iterrows()):
        sid = str(row["sample_id"])
        artifact = root / f"sample_{sid}.pt"
        prior = journal.get(sid)
        if resume and prior is not None and artifact.is_file():
            records[sid] = prior
            n_resumed += 1
            continue
        tensor, rendering, layer_indices = extract_sample(row, spec, provider)
        # Save the tensor first; the journal entry is the commit record.
        torch.save(tensor, artifact)
        record = {
            "sample_id": sid, "artifact": artifact.name,
            "shape": list(tensor.shape), "layer_indices": layer_indices,
            "subject_model": spec.subject_model, "model_revision": spec.model_revision,
            "tokenizer_revision": spec.tokenizer_revision,
            "activation_site": spec.activation_site,
            "prompt_template_id": spec.prompt_template_id,
            "max_sequence_length": spec.max_sequence_length,
            "spec_fingerprint": spec.fingerprint(),
            **rendering.to_record(),
        }
        with journal_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        records[sid] = record
        n_extracted += 1
        if progress_every and n_extracted % progress_every == 0:
            print(f"extracted {n_extracted} new / {len(df)} total", flush=True)

    shapes = {tuple(r["shape"]) for r in records.values()}
    if len(shapes) > 1:
        raise ResearchIntegrityError(f"Inconsistent activation shapes across samples: {sorted(shapes)}")
    manifest = {
        "schema_version": 1,
        "spec_fingerprint": spec.fingerprint(),
        "spec": spec.to_dict(),
        "is_pre_decision": spec.is_pre_decision,
        "n_samples": len(records),
        "n_newly_extracted": n_extracted,
        "n_resumed": n_resumed,
        "activation_shape": list(next(iter(shapes))) if shapes else None,
        "samples": [records[str(sid)] for sid in df["sample_id"]],
    }
    (root / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def load_extraction_manifest(activation_dir: str | Path) -> dict[str, Any]:
    path = Path(activation_dir) / MANIFEST_FILENAME
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing activation provenance manifest: {path}. Run `deception-paper "
            "collect-activations` so the artifacts record how they were produced."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def verify_extraction_manifest(
    df: pd.DataFrame, activation_dir: str | Path, config: Any | None = None
) -> dict[str, Any]:
    """Check a manifest covers the dataset and matches the config that will use it.

    This is what stops a ``prompt_end`` config from silently consuming
    ``response_token`` tensors.
    """
    manifest = load_extraction_manifest(activation_dir)
    recorded = ExtractionSpec.from_dict(manifest["spec"])
    manifest_ids = [str(entry["sample_id"]) for entry in manifest["samples"]]
    dataset_ids = [str(sid) for sid in df["sample_id"]]
    missing = [sid for sid in dataset_ids if sid not in set(manifest_ids)]
    if missing:
        raise ResearchIntegrityError(
            f"Activation manifest is missing {len(missing)} dataset samples "
            f"(first: {missing[:5]}); extraction is incomplete"
        )
    if recorded.fingerprint() != manifest.get("spec_fingerprint"):
        raise ResearchIntegrityError("Activation manifest fingerprint does not match its recorded spec")
    stale = [e["sample_id"] for e in manifest["samples"] if e.get("spec_fingerprint") != manifest["spec_fingerprint"]]
    if stale:
        raise ResearchIntegrityError(
            f"{len(stale)} activation records were produced under a different spec (first: {stale[:5]})"
        )
    if config is not None:
        expected = paper_config_spec_fields(config)
        conflicts = {
            key: (value, getattr(recorded, key))
            for key, value in expected.items()
            if value != getattr(recorded, key)
        }
        if conflicts:
            detail = ", ".join(f"{k}: config={c!r} manifest={m!r}" for k, (c, m) in sorted(conflicts.items()))
            raise ResearchIntegrityError(
                f"Activation artifacts do not match this config ({detail}). "
                "Re-extract or point activation_dir at the matching artifacts."
            )
    return manifest


def audit_activation_provenance(
    df: pd.DataFrame, activation_dir: str | Path, config: Any | None = None
) -> list[str]:
    """Return human-readable provenance failures for the audit command."""
    failures: list[str] = []
    try:
        manifest = verify_extraction_manifest(df, activation_dir, config)
    except (FileNotFoundError, ResearchIntegrityError, KeyError, json.JSONDecodeError) as exc:
        return [f"activation provenance failure: {exc}"]
    required = {"token_index", "n_prompt_tokens", "layer_indices", "shape", "subject_model",
                "activation_site", "prompt_template_id", "mode"}
    for entry in manifest["samples"]:
        gaps = required - set(entry)
        if gaps:
            failures.append(f"activation record {entry.get('sample_id')} is missing fields {sorted(gaps)}")
            break
    if not manifest.get("is_pre_decision"):
        failures.append(
            f"activation mode {manifest['spec']['activation_mode']!r} is a response-token diagnostic, "
            "not a valid primary pre-decision extraction"
        )
    root = Path(activation_dir)
    absent = [e["sample_id"] for e in manifest["samples"] if not (root / e["artifact"]).is_file()]
    if absent:
        failures.append(f"{len(absent)} manifest-recorded activation files are absent (first: {absent[:5]})")
    return failures


class TransformersHiddenStateProvider:
    """Real :class:`HiddenStateProvider` backed by a Hugging Face causal LM.

    Constructing this downloads/loads weights, so it is never instantiated by
    imports, tests, or the audit path.  Callers must opt in explicitly.
    """

    def __init__(self, spec: ExtractionSpec, device: str = "cpu") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer  # local import: heavy

        self.spec = spec
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            spec.subject_model, revision=spec.tokenizer_revision or spec.model_revision
        )
        # Validated in ExtractionSpec, so a typo has already failed loudly rather
        # than silently downgrading to float32 and quietly doubling memory.
        dtype = getattr(torch, spec.torch_dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            spec.subject_model, revision=spec.model_revision, dtype=dtype
        )
        self.model.to(device)
        self.model.eval()

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=True)["input_ids"])

    @torch.no_grad()
    def hidden_states(self, token_ids: Sequence[int], site: str) -> torch.Tensor:
        ids = torch.tensor([list(token_ids)], dtype=torch.long, device=self.device)
        if site in ("block", "residual_stream", "resid", "residual"):
            out = self.model(input_ids=ids, output_hidden_states=True)
            # hidden_states[0] is the embedding output; drop it so index i is block i.
            return torch.stack(list(out.hidden_states[1:]), dim=0).squeeze(1)
        return self._hooked_hidden_states(ids, site)

    @torch.no_grad()
    def _hooked_hidden_states(self, ids: torch.Tensor, site: str) -> torch.Tensor:
        from .activation_sites import get_transformer_layers, resolve_hook_module

        layers = get_transformer_layers(self.model)
        if layers is None:
            raise ResearchIntegrityError("Cannot identify transformer layers for this model")
        captured: dict[int, torch.Tensor] = {}
        handles = []

        def make_hook(idx: int):
            def hook(module, inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captured[idx] = hidden.detach()
            return hook

        for idx in range(len(layers)):
            handles.append(resolve_hook_module(self.model, idx, site).register_forward_hook(make_hook(idx)))
        try:
            self.model(input_ids=ids)
        finally:
            for handle in handles:
                handle.remove()
        if len(captured) != len(layers):
            raise ResearchIntegrityError(
                f"Captured {len(captured)} of {len(layers)} layers at site {site!r}"
            )
        return torch.stack([captured[i] for i in range(len(layers))], dim=0).squeeze(1)


__all__ = [
    "ACTIVATION_MODES", "DIAGNOSTIC_MODES", "PRE_DECISION_MODES",
    "ExtractionSpec", "HiddenStateProvider", "PromptRendering",
    "TransformersHiddenStateProvider", "assert_spec_compatible",
    "audit_activation_provenance", "extract_sample", "load_extraction_manifest", "paper_config_spec_fields",
    "render_extraction_prompt", "run_extraction", "verify_extraction_manifest",
]
