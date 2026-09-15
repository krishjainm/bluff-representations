# P1: real data / extraction provenance plan

Dated progress log and plan for the `conference-rebuild` P1 chunk. This file is
the working record; `docs/OPUS_HANDOFF.md` remains the authoritative summary.

## 1. Audit of the existing activation-extraction code

Audited against the assignment brief Phase 5 (leakage-aware extraction), Phase 18
(resumable jobs), and Phase 20 (audit must verify activation metadata).

| # | Finding | Location | Severity |
|---|---|---|---|
| A1 | The strict paper path has **no extraction stage at all**. `paper_cli` exposes only `validate-data`, `make-splits`, `train-probes`, `audit`. `load_activations` consumes `sample_<id>.pt` files that nothing in the strict path produces. | `deception_circuits/paper_cli.py`, `paper.py:load_activations` | blocking |
| A2 | The only existing producer of those artifacts builds `f"Q: {s}\nA: {r}"` and pools the last non-pad token, so the probe reads the **completed response**. This is exactly Reviewer A's "possible trivial probing if the probe reads the label/answer token". | `model_integration.py:690` (`ModelIntegrationPipeline.create_complete_dataset`) | blocking |
| A3 | No provenance is recorded per artifact. `load_activations` validates shape and finiteness only. Model id, revision, tokenizer revision, hook site, layer list, prompt template, sequence length, activation mode, and **token index** are all absent. A `activation_mode: prompt_end` config can silently consume response-token tensors. | `paper.py:load_activations` | blocking |
| A4 | Extraction is not resumable and not batched. `ActivationExtractor.extract_activations` tokenizes an entire list in one call and holds every layer's full `[batch, seq, hidden]` tensor in memory; an interrupted run loses all work. | `model_integration.py:498` | high |
| A5 | Pooling is attention-mask based, not boundary based. "last" means last non-pad token of whatever string it was handed; it has no concept of a prompt/response boundary and is wrong under left padding. | `model_integration.py:352` (`_pool_hidden_states`) | high |
| A6 | `render_predecision_prompt` is a formatting guardrail that **no extraction code calls**. The `Action:` boundary is asserted but never used to locate a token. | `paper.py:33` | high |
| A7 | `PaperConfig.activation_layers`, `max_sequence_length`, `activation_site`, `activation_mode`, `model_revision`, `tokenizer_revision`, and `prompt_template_id` are recorded in the resolved config but **never read by any code**. They are documentation, not constraints. | `paper.py:PaperConfig` | high |
| A8 | `audit_run` performs no activation checks whatsoever, contradicting the brief's required "activation metadata exists" and "no missing samples" checks. | `paper.py:audit_run` | high |
| A9 | `activation_sites.py` annotates `Optional[nn.ModuleList]` without importing `Optional`; it only survives because of `from __future__ import annotations`. | `activation_sites.py:13` | low |

Data availability: no canonical dataset exists. The repository contains a 4-row
`quick_demo_data/quick_demo.csv` that does not even carry the canonical schema
(`sample_id`, `base_item_id`, `scenario` are absent), and `data/openai_runs/` is
empty apart from a README. **No real activations have been extracted.**

## 2. Design

New module `deception_circuits/paper_extraction.py`, strict by construction.

### 2.1 Leakage safety is structural, not conventional

`render_extraction_prompt` dispatches on mode and the pre-decision modes
(`prompt_end`, `decision_token_prelogit`, `mean_prompt`) are handed a row
projection that **does not contain the `response` column**. Reading response text
in those modes raises rather than silently succeeding. `response_token` remains
available as an explicitly named diagnostic and must be opted into with
`allow_response_leakage=True`, which is also stamped into the manifest so an
auditor can see it.

### 2.2 Provenance

`ExtractionSpec` carries subject model, model revision, tokenizer revision, hook
site, activation mode, layer selection, max sequence length, prompt template id
and template string, dtype, and the diagnostic opt-in. `spec.fingerprint()` is a
SHA-256 over its canonical JSON. Every run writes:

- `extraction_spec.json` — the resolved spec plus fingerprint,
- `extraction_manifest.jsonl` — one append-only record per sample, written
  immediately after the tensor is saved,
- `extraction_manifest.json` — the consolidated manifest with a coverage summary.

Each per-sample record stores `sample_id`, artifact filename, `token_index`,
`n_prompt_tokens`, `truncated`, `layer_indices`, tensor `shape`, and the prompt
SHA-256 (not the prompt text, to keep private data out of committed artifacts).

### 2.3 Resumability and mismatch refusal

`run_extraction` loads any existing `extraction_spec.json`. If the fingerprint
differs from the requested spec it **raises** instead of overwriting or mixing
artifacts. If it matches, already-recorded samples with a present artifact file
are skipped, so an interrupted job resumes sample-wise.

### 2.4 Model access is injected

Hidden states are obtained through a `HiddenStateProvider` protocol
(`tokenize`, `hidden_states`). The real implementation
(`TransformersHiddenStateProvider`) is constructed lazily and only when a caller
explicitly asks for it, so importing the module never touches the network. Tests
inject a deterministic stub provider — **no model downloads, no API calls**.

### 2.5 Wiring

- `paper.py:load_activations` gains an optional `config`/manifest check that
  verifies the manifest exists, its spec matches the config's model/site/mode/
  template, and it covers every dataset `sample_id`.
- `paper_cli.py` gains `collect-activations`.
- `audit_run` gains activation-provenance checks.

## 3. Out of scope for this chunk

Deliberately not attempted here, and not to be described as done:

- specifying the canonical dataset release (needs the team's real data),
- P2 confound/matched analyses,
- P3 behavioral causal runner,
- P4 SAE audit,
- any real model execution.
