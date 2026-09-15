# Opus handoff: conference rebuild

## Scope and branch state

Work is on local branch `conference-rebuild`, based on V5 commit `ef7ef6b`.
It has **not been pushed**. The working tree was clean at the end of the Codex
pass. Do not rewrite history; review and extend the three local commits:

| Commit | Purpose |
|---|---|
| `b77615f` | strict reproducible paper-probe workflow |
| `a970aaa` | grouped CIs, baselines, provenance fields, and audit checks |
| `4b2c926` | explicit pre-decision boundary and causal-control corrections |

`Claude-API-KEYS.md` is intentionally Git-ignored. Do not read, print, stage,
or commit it. The user rotated its contents after a prior accidental log exposure.
Load credentials only inside a process environment, never source code or result
artifacts.

## What has been completed

### Strict paper workflow

`deception_circuits/paper.py` is a separate, intentionally strict V2 path; it
does not depend on the legacy/demo loader behavior.

- Canonical CSV validation requires `sample_id`, `base_item_id`, `statement`,
  `response`, `label`, and `scenario`; IDs must be unique and labels must have
  both binary classes.
- It uses `split_group_id` when present (otherwise `base_item_id`) to create a
  deterministic group-safe train/validation/test manifest. The manifest stores
  a dataset SHA-256 checksum and is validated for coverage, overlap, and group
  leakage.
- It requires `sample_<sample_id>.pt` finite tensors of matching `[layers,
  hidden]` shape. Missing directories/files, malformed tensors, NaN/Inf, and
  shape mismatch are hard failures.
- It fits per-layer logistic probes on train only, selects one layer by
  validation AUROC only, and reports the frozen selection on test. It never
  selects by test AUROC.
- It has configurable repeated seeds, full classification metrics, grouped
  bootstrap AUROC/PR-AUC CIs, majority baseline, prompt-text baseline,
  response-text diagnostic baseline, and optional configured nuisance-only
  baseline.
- It writes resolved config, run metadata (including git commit/environment),
  split manifest, and probe results. Its audit checks expected artifacts, split
  integrity, validation-only selection, seed count, CIs, and baselines.

CLI:

```bash
deception-paper validate-data --config my_run.yaml
deception-paper make-splits --config my_run.yaml
# Loads subject-model weights and runs real forward passes; requires the flag.
deception-paper collect-activations --config my_run.yaml --confirm-model-load
deception-paper run-baselines --config my_run.yaml     # no model needed; run this first
deception-paper train-probes --config my_run.yaml
deception-paper analyze-confounds --config my_run.yaml
deception-paper train-sae --config my_run.yaml
deception-paper run-interventions --config my_run.yaml --confirm-model-load
deception-paper make-figures --config my_run.yaml
deception-paper audit --config my_run.yaml
```

Use `configs/paper_v2.yaml` as the starting template. It must be copied and
populated with real dataset/model/extraction details; placeholder paths are not
runnable data.

### Legacy correctness corrections

- `DeceptionDataLoader.load_csv(..., research_mode=True)` now fails if real
  activations are absent and fails on missing/non-finite/shape-mismatched files.
  Synthetic activations remain allowed only with `research_mode=False` for old
  demos/tests.
- `DeceptionTrainingPipeline.run_full_experiment` defaults to
  `research_mode=True` and passes this through to the loader.
- Probe early stopping now deep-copies the best `state_dict`; reported accuracy
  is recomputed after restoring the best checkpoint.
- Legacy pipeline layer selection now requires explicit validation activations
  and selects by validation metric rather than test AUROC.

### Activation and causal guardrails

- `render_predecision_prompt(statement)` creates a prompt ending at a required
  `Action:` boundary and deliberately cannot receive response text. This is a
  formatting guardrail, **not** a completed model extraction pipeline.
- Causal generation now records a `timing_policy`: `prefill_only`,
  `first_decision_token`, or `every_decode_step`.
- Replacement mode now truly replaces the selected activation; it does not
  scale a vector by steering strength.
- Causal control helpers include positive/negative steering, random and
  orthogonal controls, optional shuffled-label and wrong-layer controls,
  wrong-token-position control, and mismatched activation replacement.
  These helpers are not yet a paper-valid behavioral experiment runner.

### Documentation and tests

Created/updated:

- `README.md`
- `docs/EXPERIMENT_PROTOCOL.md`
- `docs/REPRODUCIBILITY.md`
- `docs/REVIEW_RESPONSE_MATRIX.md`
- `docs/PAPER_V2_EXPERIMENT_PLAN.md`
- `docs/COMPUTE_PLAN.md`
- `docs/PAPER_CLAIM_GUARDRAILS.md`
- `docs/ACCEPTANCE_CHECKLIST.md`
- `tests/test_paper_integrity.py`

Most recent validation:

```bash
MPLCONFIGDIR=/private/tmp/deception-llms-matplotlib \
UV_CACHE_DIR=/private/tmp/deception-llms-uv-cache \
uv run --with pytest pytest -q tests/test_paper_integrity.py
```

Result: **5 passed**. Tests cover group-safe manifests, validation-only selection,
strict missing activation failure, duplicate ID failure, prompt boundary, and
replacement/prefill timing semantics. No large model/API calls were made.

## Important limitations

The current branch is an honest foundation, not a completed conference suite.
Do not describe any of the following as completed or use old generated results
as evidence:

- no real dataset has been validated or versioned;
- no real subject-model activations have been extracted;
- no Claude API call has been made;
- no behavioral intervention endpoint has been measured;
- no SAE result, figure, or paper metric has been generated;
- no public/review-ready experiment result exists.

The legacy repository still contains broad claims and experimental code. The
new strict path is the only current intended paper path; either migrate legacy
functions deliberately or label them legacy in docs/code rather than treating
them as validated evidence.

## Prioritized Opus plan

### P0 — inspect and preserve

1. Inspect `git status`, the three commits above, `paper.py`, the legacy
   pipeline, and current dataset availability before edits.
2. Do not commit secrets, private source data, activation tensors, model weights,
   raw paid-API outputs, or caches. Keep the existing branch/history.
3. Read the assignment brief at
   `/Users/home/Documents/deception-llms-global/ALGOVERSE - Deception LLMs.md`.
   Treat its non-negotiable scientific rules as binding.

### P1 — complete the real data/extraction path

1. Specify one canonical dataset release with dataset card, label source,
   construction prompts, examples, composition report, and metadata dictionary.
2. Implement batched/resumable activation extraction from the selected open
   subject model. Record exact HF model and tokenizer revisions, hook site,
   layers, prompt template ID, sequence length, sample ID, **token index**, and
   activation mode in a per-artifact manifest.
3. Use prompt-end/pre-decision extraction as primary. Add tests proving response
   tokens cannot enter this path. Treat response-token activation analysis as a
   clearly labelled diagnostic only.
4. Ensure the extraction job can resume sample-wise and refuses metadata/model
   mismatches rather than overwriting artifacts.

### P2 — data controls and probe analysis

1. Add group-stratified manifest construction appropriate to the real dataset;
   test label balance and class availability per partition.
2. Implement metadata validation and poker analyses: action-matched sets,
   bluff/value aggressive-action comparisons, and configured action/street/
   position/hand-strength/board/bet-size nuisance models. Do not fabricate
   unavailable poker metadata.
3. Add learning curves with independent subsamples, calibrated metrics/ECE, and
   seed-level plus grouped CI summaries. Preserve source validation vs target
   test selection in cross-context evaluation.
4. Run only after the data and extraction provenance are auditable. Store real
   artifacts, not reported numbers copied into code/docs.

### P3 — behavioral causal suite

1. Define a task-native independent endpoint before running: structured poker
   action/logit or pre-registered bluff/value criterion, plus legal-action and
   response-quality checks. Probe score is diagnostic only.
2. Implement paired held-out evaluation with deterministic primary decoding or
   recorded matched seeds. Record prompt IDs, baseline/intervened output,
   condition, direction source, direction/hidden norms, layer/site/token timing,
   strength, and independent endpoint.
3. Run baseline, positive, negative, random matched-norm, orthogonal,
   shuffled-label, nuisance direction (if available), wrong layer, wrong token/
   timing, and true patching/mismatched-prompt conditions. Use paired bootstrap
   effects/CIs and quality metrics.
4. Verify `replace` is called only for actual activation replacement; call
   vector addition/subtraction steering.

### P4 — SAE and reporting

1. Audit existing SAE width, normalization, tied weights, loss semantics, and
   whether any legacy result was supervised. Build one defensible unsupervised
   path only if it can be tested.
2. Freeze SAE before held-out feature ranking/evaluation. Report reconstruction,
   explained variance if applicable, L0, dead features, activation frequencies,
   seed stability, and top/contrast examples with metadata.
3. Generate figures only from real structured artifacts, saving plot source data
   alongside each plot. Do not create mock publication figures.
4. Complete every row in `REVIEW_RESPONSE_MATRIX.md`, then make audit fail for
   any required artifact missing before paper writing.

## Claude API usage

The original brief asks for a Claude coding pass and gave an approximate budget,
but it also prohibits paid calls unless authorized. The user has now authorized
use of a rotated local credential. Before an API request:

1. Confirm the actual model identifier and per-run cost/budget with the user if
   it is not already explicit in the API-key file or task context.
2. Keep prompts targeted: first have Opus audit the branch and propose a
   file-by-file P0/P1 implementation plan; then execute bounded implementation
   chunks with tests. Do not send credentials, private datasets, or entire large
   activation artifacts in prompts.
3. Do not start paid model inference, large HF downloads, or paid LLM judging
   merely because the API credential exists.

## Definition of a good handoff outcome

At the end, report changed files, reviewer-matrix status, actual tests and
their results, exact future commands, compute-heavy actions performed/not
performed, remaining risks, and a candid go/no-go. “Code compiles” is not a
submission-ready conclusion.

---

## Running log

### 2026-09-15 — P1 chunk 1: leakage-safe extraction with provenance

**Status of the prioritized P1 plan above:** items 2, 3, and 4 are now
implemented and tested; item 1 (canonical dataset release) remains blocked on
the team's real data. No real model was loaded and no API call was made.

Full audit findings are in `docs/P1_EXTRACTION_PLAN.md`. The headline gap was
that the strict path had **no extraction stage at all**, and the only code that
produced the `sample_<id>.pt` artifacts it consumes was
`ModelIntegrationPipeline.create_complete_dataset`, which builds
`f"Q: {s}\nA: {r}"` and pools the last non-pad token — i.e. the probe read the
completed response. That is exactly Reviewer A's trivial-probing concern.

Added `deception_circuits/paper_extraction.py`:

- Leakage safety is **structural**. Pre-decision modes (`prompt_end`,
  `decision_token_prelogit`, `mean_prompt`) never receive the `response` column,
  and an `ExtractionSpec` whose template references `{response}` in those modes
  fails at construction. `response_token` still exists but requires
  `allow_response_leakage=True`, which is stamped into the manifest, and the
  audit reports it as not a valid primary extraction.
- `prompt_end`/`decision_token_prelogit` templates must end at an explicit
  `Action:` boundary, and truncation that would move that boundary is a hard
  error rather than a silent shift.
- `ExtractionSpec.fingerprint()` is a SHA-256 over every choice that changes the
  numbers: model, both revisions, hook site, mode, layer subset, sequence limit,
  template id and text, dtype, diagnostic opt-in.
- Per-sample provenance is written to `extraction_manifest.jsonl` immediately
  after each tensor lands, then consolidated into `extraction_manifest.json`:
  `sample_id`, artifact, **exact `token_index`**, `n_prompt_tokens`, `truncated`,
  `prompt_span`, `layer_indices`, shape, and a prompt SHA-256 (the hash, not the
  text, so artifacts carry no private dataset content).
- Extraction resumes sample-wise, re-extracts any sample whose artifact was
  deleted, and **refuses** to write into a directory whose recorded spec
  fingerprint differs instead of mixing artifacts.
- Model access is injected via a `HiddenStateProvider` protocol.
  `TransformersHiddenStateProvider` is imported lazily and constructed only on
  explicit request, so importing the package never touches the network.

Wiring:

- `paper.py:load_activations` now requires a provenance manifest by default and,
  when given a `PaperConfig`, verifies the manifest agrees with it field by
  field. A `prompt_end` config can no longer consume `response_token` tensors.
  `require_manifest=False` remains only for unit-testing the tensor checks.
- `paper.py:audit_run` now runs `audit_activation_provenance`.
- `paper_cli.py` gained `collect-activations`. It refuses to run without
  `--confirm-model-load` and prints exactly what it would load first, so no
  large download can happen by accident.
- Config fields that were previously inert documentation (`activation_site`,
  `activation_mode`, `activation_layers`, `max_sequence_length`,
  `model_revision`, `tokenizer_revision`, `prompt_template_id`) now constrain a
  real run. `subject_model` containing `REPLACE` is rejected.
- Fixed `activation_sites.py` missing `Optional` import.

**Note on a pre-existing test.** `test_missing_activation_is_a_hard_error`
started passing for the wrong reason once provenance checks landed: its
`match="Missing activation"` also matched the new "Missing activation provenance
manifest" error. It is now
`test_missing_activation_file_is_a_hard_error`, anchored on
`Missing activation for sample_id=0-0`, so it cannot pass because provenance is
merely absent. The fixture in `tests/test_paper_integrity.py` now builds its
activations through `run_extraction` with a named stub, so it exercises the real
provenance path.

Tests: `31 passed` (`tests/test_extraction_integrity.py` 26 new,
`tests/test_paper_integrity.py` 5). Command:

```bash
MPLCONFIGDIR=/private/tmp/deception-llms-matplotlib \
UV_CACHE_DIR=/private/tmp/deception-llms-uv-cache \
uv run --with pytest pytest -q tests/
```

Also smoke-tested the whole CLI on stub artifacts:
`validate-data` → `make-splits` → `train-probes` → `audit` all pass, and audit
correctly FAILs on a mismatched `subject_model` and on a provenance-less
activation directory. The console script is not installed in `.venv`; use
`uv run python -m deception_circuits.paper_cli <command>`.

**Still true and unchanged:** no real dataset, no real activations, no API call,
no behavioral endpoint measured, no SAE result, no paper metric. The repository
contains only a 4-row `quick_demo_data/quick_demo.csv` that does not carry the
canonical schema.

### 2026-09-15 — P2 chunk: confound controls, stratified splits, curves

**Status:** P2 items 1–3 are implemented and tested. P2 item 4 ("run only after
data and extraction provenance are auditable") is still blocked: no real data
exists. No model was loaded and no API call was made.

`docs/DATASET_REQUIREMENTS.md` is new and answers "what is actually needed to
close P1". The substantive point in it: **decide whether the model is the actor
or an observer before collecting anything.** If the subject model chooses the
action and the label describes its own bluff, then P3's causal endpoint is native
and Reviewer A's "did steering change deception-specific behavior" is answerable.
If the model merely reads a human hand history, that question has no native
endpoint and the causal claim should be dropped. Verified sourcing: the
`uoftcprg/phh-dataset` poker hand histories are MIT-licensed and include the
10,000-hand Pluribus subset, which logs hole cards throughout and so avoids the
showdown-selection bias of the scraped online subsets. None of these ship bluff
labels — the label must be derived by a pre-registered mechanical rule over an
aggressive action and a computed equity figure, with the ambiguous middle band
excluded rather than forced into a class.

New `deception_circuits/paper_confounds.py`:

- `describe_metadata_availability` / `validate_nuisance_metadata` report per
  column presence, coverage, and inferred kind, and map each of the seven named
  analyses to the columns it needs. A control that cannot run is emitted as
  `not_run` **with a reason**, so a missing control is visible in the artifact
  rather than absent from it.
- `bluff_vs_value_subset` restricts to aggressive actions, which is the
  comparison that separates bluff-like deception from mere aggression.
- `build_matched_subset` equalises label counts inside every nuisance stratum
  (exact match on categoricals, quantile bins on continuous), drops and counts
  single-class strata, and is deterministic given a seed.
- `nuisance_decodability` dispatches on column type: multinomial logistic with a
  majority baseline for categoricals, ridge with R² against a train-mean
  predictor for continuous. An earlier version stringified `hand_strength` into
  hundreds of "classes" and reported a meaningless accuracy; that is fixed and
  a degenerate categorical target is now refused with an explanation.
- `residualize_activations` fits the nuisance removal on **train only** and
  applies the training coefficients to both partitions, so the control itself
  cannot leak. `controlled_probe_analysis` reports unadjusted and controlled
  AUROC plus the drop; neither direction is treated as the expected result.
- `run_confound_suite` never selects a layer — it consumes the
  validation-selected layer from `probe_results.json`.

In `paper.py`:

- `binary_ece` migrated in from `linear_probe.py`, which now re-exports it so
  the two cannot drift. The strict version raises on empty input instead of
  returning a calibrated-looking `0.0`.
- `compute_binary_metrics` (public; `_metrics` is now an alias) adds `ece`, `n`,
  and `positive_rate`.
- Splits are group-safe **and** label-stratified by default
  (`split_strategy: grouped_stratified`, via `StratifiedGroupKFold`). Requesting
  a richer stratification than the group count supports drops optional columns
  and records what was used rather than failing. Manifests are `schema_version: 2`
  and carry a `partition_summary`; a single-class partition is a hard error.
- `summarize_across_seeds` reports mean/std/median/IQR and selected-layer
  stability — a different uncertainty source from the grouped bootstrap, which
  is what Reviewer B asked for.
- `run_learning_curve` subsamples at **group** level over
  `learning_curve_subsamples` independent draws per size, so a smaller training
  set is fewer poker hands rather than fewer rows from the same hands. Each
  replicate records a 16-char hash of its chosen group set, which makes the draws
  auditable and reproducible without inlining id lists.
- `audit_notes` is separate from `audit_run`. A control that legitimately could
  not run is a note, not a failure, so PASS keeps meaning "nothing is wrong".

CLI gained `analyze-confounds`, which refuses to run before `train-probes`
because it reuses that stage's validation-selected layer.

Tests: **79 passed** (`tests/` — 26 extraction, 32 confound, 16 split/curve,
5 paper integrity). Full CLI smoke test on stub artifacts:
`validate-data → make-splits → train-probes → analyze-confounds → audit` all
pass, and audit reports `equity_matched` as not-run because the stub fixture's
`hand_strength` nearly separates the label, so equity-matching leaves no
two-class stratum. That is the correct behaviour and a good illustration of what
the control is for.

**Still true:** no real dataset, no real activations, no API call, no behavioral
endpoint, no SAE result, no paper metric. Every confound number produced so far
is chance-level stub output, as expected.

### 2026-09-15 — the real dataset: found, adapted, and characterised

`sample_data/normalized_poker_gpt4o.csv` **does exist** — on
`origin/dataset-integration` (`9695f3c`), not in the `conference-rebuild` working
tree, which is why the earlier P1 note said no dataset was available. 44,631
rows, 44,311 distinct dealt hands, 16.8% positive.

`deception_circuits/poker_adapter.py` maps it to the canonical schema and
recovers metadata at 100% parse coverage: position, street, hole cards, board,
amount faced, board texture, and a made-hand category from a tested 5–7 card
evaluator. `prepare_poker_dataset.py` materialises the CSV and a dataset card
into git-ignored `data/derived/`.

Four measured properties change what this data can support. Full detail in
`docs/POKER_DATASET_NOTES.md`; the short version:

1. **`response` is the label verbatim** for all 44,631 rows. Response-token
   analysis scores 1.0 and means nothing. The pre-decision guard built in the P1
   chunk is not hypothetical here — it is load-bearing.
2. **Every prompt asks "Is this a bluff?"**, so the probe is substantially
   reading the model's answer to an explicitly asked question. The adapter's
   `prompt_variant=neutral_state` strips the question so the size of this
   instruction confound can be measured instead of argued about.
3. **The model is a classifier, not the actor** — the judged raise has already
   happened. So the strongest available causal claim is "intervening changes the
   model's bluff *judgement*", not "changes its deceptive *behaviour*". The
   second needs the model-as-actor design in `docs/DATASET_REQUIREMENTS.md`.
4. **The label is nearly a function of hand strength.** A 9-category one-hot of
   made-hand alone gives **AUROC 0.885 / PR-AUC 0.551** against a 0.168 base
   rate; adding street, position, and board texture gives **0.921 / 0.600**.
   In-sample, so an upper bound, but a few categorical levels over 44k rows
   barely overfit.

Point 4 is the one that matters most for the rebuild. A layerwise probe scoring
in the low 0.90s here is not clearly beating mechanical hand evaluation, so the
paper's claim has to be "beyond what made-hand strength and game state explain",
reported next to the nuisance-only baseline and the new `made_hand_matched`
control. That control was added to `paper_confounds.ANALYSIS_REQUIREMENTS` in
this chunk.

Also generalised `ExtractionSpec.decision_boundary_marker` (default `Action:`)
so a forced-choice judgement prompt can end at `Answer:` and still have its
boundary structurally enforced.

Tests: **114 passed** (35 new adapter tests, including every made-hand category,
the wheel, ace-high straights, and four-to-a-straight/flush negatives — a bug in
that evaluator would silently corrupt a control variable).

No model was loaded and no API call was made. No probe has been trained on this
data yet.

### 2026-09-15 — P3: behavioral causal intervention suite

**Status:** P3 items 1–4 implemented and tested against stubs. Not run on a real
model. No API call, no weights loaded.

`deception_circuits/paper_causal.py` exists to enforce the brief's ninth rule:
a direction from a probe plus an intervention that adds that direction, measured
by the same probe's score, is circular. So:

- The primary endpoint is `ForcedChoiceEndpoint` — the model's own probability
  over its allowed answer tokens, read from the output head after a single
  deterministic forward pass. No sampling, so the paired comparison carries no
  decoding noise.
- `assert_endpoint_independence` raises on `probe_score`/`probe_probability`/
  `probe_logit`, `run_causal_suite` calls it before doing any work, and the audit
  fails any causal run that does not declare
  `primary_endpoint_is_probe_derived: false`.
- The probe projection is still recorded, as `diagnostic_probe_projection`, named
  so it cannot be quietly promoted to the result.

**Steering vs patching is enforced by the type, not by convention.**
`Intervention` refuses `kind="replace"` with a nonzero strength — a scaled
replacement is a steer, and calling it patching is what Reviewer B objected to.
Patching has no dose grid and sources a real activation from a different
held-out prompt via a deterministic rotation.

**Directions** are all fitted on the train partition only (`build_direction_set`,
tested by flipping test labels and asserting the direction is unchanged): probe,
random matched-norm, orthogonalised-against-probe, probe refit on permuted
labels, and optionally a nuisance direction fit to predict a metadata column.

**Conditions**: baseline, positive/negative steering, random matched-norm,
orthogonal, shuffled-label, wrong-layer, wrong-token-position, mismatched-prompt
patching, plus nuisance steering when metadata allows. `causal_eval_size`
defaults to 300, not the 100 both reviewers criticised.

**Statistics**: `paired_bootstrap_effect` holds pairs together and resamples
*groups*, returning mean difference, percentile CI, Cohen's d_z (None when
differences are constant rather than infinite), and signed-consistency rate.

**Quality controls** sit beside the effects, not in an appendix:
`valid_choice_mass` per condition and strength catches an "effect" that is really
the model losing the ability to answer in the required format.

Every record logs kind, layer, site, token index, timing policy, strength,
direction name/source/norm, hidden norm, and `relative_magnitude` — the strength
in units of the hidden-state norm, which is the only interpretable dose unit.

CLI gained `run-interventions`, guarded by `--confirm-model-load`. Per-row
records go to `intervention_records.csv`; `causal_results.json` keeps summaries
only, so no large arrays end up in JSON.

**Two real bugs were found and fixed while testing.** `intervention.to_record()`
was spread into the row dict *after* the loop's `strength` key, silently
overwriting it — invisible for steering, where the values agree, but it collapsed
every patch row onto strength 0.0 and produced duplicate sample_ids in the
pairing, which is what surfaced it. Patch conditions were also being evaluated
once per strength despite patching having no dose. Identity fields are now
written last and `replace` conditions are evaluated once.

The stub suite reproduces the pattern a valid experiment should show: positive
steering +0.34, negative −0.49, random +0.03, orthogonal −0.01, shuffled −0.04,
wrong-layer and wrong-token exactly 0.0, mismatched-prompt patching centred on
zero with a wide CI. Note that orthogonal is not exactly zero and should not be
expected to be: it is orthogonal to the *probe*, which is only ~0.99 aligned with
the stub's readout.

Also: `build_direction_set` now raises an actionable error when the probe learns
a zero-norm direction, which is what happens if it is pointed at activations
carrying no label signal.

Tests: **154 passed** (40 new). Full CLI smoke test passes end to end; audit
PASSes on a complete causal run and FAILs when a control condition is removed.

**What this suite can and cannot license on the current dataset** is in
`docs/PAPER_CLAIM_GUARDRAILS.md`: because the model judges an already-taken
raise, a clean result licenses "intervening changes the model's bluff
*judgement*", not "changes its deceptive *behaviour*".

### 2026-09-15 — P4: SAE audit and rebuild, figures, matrix completion

**Status:** P4 items 1–4 done. Stub-tested only; no model loaded, no API call.

**The finding.** `training_pipeline.py` selected `best_autoencoder_*.pt` from the
**supervised** results, keyed by **classification accuracy**:

```python
best_layer_name = sorted(supervised_results.items(),
                         key=lambda x: x[1]['final_metrics']['accuracy'])[0]
```

`SupervisedDeceptionAutoencoder` has a `classify` head and a cross-entropy term,
so its features are label-predictive by construction. Any legacy interpretability
claim resting on that artifact is supervised feature *construction*, not
unsupervised discovery — and the checkpoint selection used labels too. The
`.meta.json` does record `"supervised": true`, so existing artifacts can at least
be identified. Full defect list in `docs/SAE_AUDIT.md`, including two that would
invalidate any frozen-SAE evaluation:

- `decode()` did `with torch.no_grad(): decoder.weight.copy_(encoder.weight.t())`
  on **every forward**. The copy is outside the autograd graph, so no gradient
  flows encoder-ward through the tie, while the optimizer updates the decoder
  independently and the next forward discards that update. `tied_weights=True`
  was the default.
- That copy also runs at inference, so a "frozen" autoencoder mutates its own
  parameters when you call it.

Also: `bottleneck_dim=0` (= input width) in every caller so nothing was
overcomplete and no `8192` exists anywhere in the repo; no decoder unit-norm, so
the L1 penalty was gameable by growing decoder norms; top-k selected on `abs(x)`
with no ReLU so the "sparse code" carried negatives; L1 meaned over features so
effective sparsity scaled as 1/width; no variance-explained, L0, or dead-feature
reporting at all.

The legacy module is left in place and marked experimental rather than migrated —
the tying and mutation defects are structural.

**`deception_circuits/paper_sae.py`** is the replacement: mandatory explicit
`n_features` with the expansion factor recorded, TopK by default (exact L0),
genuine tying (decoder transposes the encoder parameter *inside* the graph, so
there is one matrix and gradients flow), decoder columns renormalised to unit norm
after every optimizer step, input normalization fitted on train only, epoch
selected on validation MSE with a deep-copied checkpoint, and `freeze()` before
any held-out evaluation. `parameter_fingerprint()` exists so a test can assert
that evaluating a frozen SAE changes not one weight — which is the bug class the
legacy code had. Labels enter only to **rank** already-frozen features on
train+validation; selected features are scored once on test. Feature exports carry
metadata and `interpretation: "not assigned; requires human inspection"`.

**`deception_circuits/paper_figures.py`** generates 10 figure kinds and has no
mock path: each builder returns `not_run` with a reason when its artifact is
absent, so a publication directory cannot fill with plausible-looking plots that
correspond to nothing. Every figure ships its source CSV; `figures_manifest.json`
records the source artifact and its SHA-256, and `audit_figures` detects deleted
images, orphaned figures, and **stale** figures whose source changed after
generation. The categorical palette was checked with the data-viz validator (all
checks pass, worst adjacent CVD ΔE 9.1 protan / 5.8 tritan); the tritan value sits
in the floor band, which is why per-series markers are mandatory rather than
decorative, and the source CSV is the table view that discharges the sub-3:1
contrast warning.

Rendering the figures and looking at them caught four things the validator cannot
see: a dead band between title and plot, a legend colliding with the provenance
strip, snake_case endpoint names in axis labels, and — the real one — a hard
`ylim=(0, 1.05)` on variance-explained that **clipped negative values out of
view**. A negative EV means the SAE reconstructs worse than the partition mean;
clamping it turned a failed reconstruction into a blank panel. The axis is now
derived from the data with a zero line and sign-aware labels.

Also added `choose_wrong_layer_offset`: the wrong-layer control's fixed −4 offset
resolves to layer −1 on a shallow model and hard-failed. The offset is now derived
from the model's depth.

`docs/REVIEW_RESPONSE_MATRIX.md` is complete — every reviewer concern has a row.
Two are marked **paper-only** (reference formatting, manuscript rewrite) and
**one remains unimplemented**: the cross-context generalization matrix, which
needs a second deception context. `figure_cross_scenario` reports that rather than
drawing an empty heatmap.

Tests: **206 passed** (52 new). Full nine-stage CLI runs end to end on stub
artifacts; `make-figures` produced 9 figures and correctly refused the 10th; audit
PASSes with notes.

**Still true:** no real dataset run, no real activations, no API call, no probe or
SAE trained on the real poker data, no paper metric. Every number in the smoke
artifacts is stub output.

### 2026-09-15 — first real numbers: the model-free baselines

**These are real results on the real dataset.** No model, no activations, no API
call. Fitted on the training partition and scored once on the frozen test
partition, with group-safe label-stratified splits.

Dataset: `data/derived/poker_bluff_judge_question_v1.csv`, 44,631 rows, 44,311
groups, adapted from `normalized_poker_gpt4o.fixed.csv`. Splits:

| partition | rows | groups | positive rate |
|---|---|---|---|
| train | 31,241 | 31,017 | 0.1678 |
| validation | 4,463 | 4,431 | 0.1678 |
| test | 8,927 | 8,863 | 0.1678 |

Stratification is exact: the positive rate is 0.1678 in all three partitions and
matches the corpus rate, with no group crossing a partition boundary.

| baseline | AUROC | PR-AUC |
|---|---|---|
| majority class | — | accuracy 0.8322 |
| prompt text only (TF-IDF 1-2gram) | 0.7817 | 0.3762 |
| **nuisance metadata only** | **0.9204** | **0.6166** |
| response text (leakage diagnostic) | **1.0000** | **1.0000** |

Two findings.

**The response-text diagnostic returns exactly 1.0000/1.0000.** That is the
`response`-is-the-label property confirmed on the real data by measurement rather
than inspection. Any pipeline that lets response tokens into the probe scores
perfectly and means nothing. The pre-decision extraction guard is what stands
between this dataset and a meaningless result.

**The nuisance-only baseline is AUROC 0.9204 / PR-AUC 0.6166 on held-out test
data**, using only mechanically parsed game state (action, street, position,
made-hand category, board texture, board paired, bet faced). The earlier
in-sample estimate was 0.9208, so it held up out of sample almost exactly. Full
metrics: accuracy 0.8512, balanced accuracy 0.8768, F1 0.6737, MCC 0.6218,
ECE 0.1305, confusion {tn 6228, fp 1201, fn 127, tp 1371}.

**This is now the bar.** A hidden-state probe on this dataset has to clear
AUROC 0.9204 / PR-AUC 0.6166 to be saying anything that mechanical hand
evaluation does not already say. That number must appear next to any probe
number in the paper.

Note also that the prompt-only text baseline (0.7817) is *worse* than the
structured parse (0.9204), which makes sense: TF-IDF over card names cannot
compute made-hand strength, whereas the parser does.

One thing the text baselines cannot measure: the "Is this a bluff?" question is
byte-identical in every row, so it contributes no discriminative signal to any
text model. Its effect is entirely on the subject model's internal computation,
which means the `judge_question` vs `neutral_state` comparison is only
informative once activations exist.

Metadata coverage is 1.0000 for every parsed column. All confound controls are
available except `bet_size_matched` (needs pot size, not reconstructed) and
`equity_matched` (needs true equity, not derivable without a solver).

Added `deception-paper run-baselines`: model-free, cheap, and the bar the probe
must clear, so it belongs before any GPU time rather than after.

**Extraction is not yet run.** It needs a subject model, which needs
authorization. Measured requirements for this dataset: 44,631 forward passes,
~9.0M prompt tokens of prefill (mean 201 estimated tokens per prompt, max 230).
Activation storage, float32, one vector per sample per layer:

| model | layers | hidden | all layers | 8-layer subset |
|---|---|---|---|---|
| gpt2 (124M) | 12 | 768 | 1.6 GB | 1.1 GB |
| Qwen2.5-1.5B | 28 | 1536 | 7.7 GB | 2.2 GB |
| Llama-3.2-3B | 28 | 3072 | 15.4 GB | 4.4 GB |
| Llama-3.1-8B / Mistral-7B | 32 | 4096 | 23.4 GB | 5.8 GB |

This machine: 16 GB unified memory, MPS available, no CUDA, 541 GB free disk, no
Hugging Face token on disk. **The original paper's Llama-3.1-8B is not viable
here** — fp16 weights alone are ~16 GB — and it is gated, so it would also need a
login. A replication at 8B needs a rented GPU.

### 2026-09-15 — GPU run prepared; no download performed

Decision taken: **do not download a subject model on this machine.** The 8B
replication goes to rented compute. Both prompt variants will be extracted.

`docs/GPU_RUNBOOK.md` is the turnkey sequence. Two configs are committed and both
parse with distinct extraction fingerprints (so they cannot be confused for one
another):

- `configs/real_llama31_8b_judge_question.yaml` — the primary run, all nine
  stages including the causal suite.
- `configs/real_llama31_8b_neutral_state.yaml` — the instruction-confound
  comparison, **causal stage deliberately disabled**.

Why the causal stage is off for the neutral variant: the neutral prompt asks the
model nothing, so there is no forced choice to measure and hence no behavioural
endpoint independent of the probe. Running it would have nothing to score. The
variant exists to answer one question — how much of the probe's performance
survives when the prompt does not name "bluff" — which is a probe comparison.

**A real design limitation surfaced while writing those configs and was fixed.**
`decision_boundary_marker` validated the *prompt template*, but the neutral
variant's boundary lives in the *data*: its statements already end with "The
player decided to: Raise." and the template appends nothing. Template validation
therefore rejected a legitimate config. The boundary is now enforced on the
**rendered prompt, per row**, which is strictly stronger — it also catches a row
whose own text does not end where the spec claims the decision is. Template
validation is retained for the case where the template appends a literal suffix.

Read positions verified on the real data:

| variant | marker | prompt ends with | read position |
|---|---|---|---|
| `judge_question` | `Answer:` | `…Reply with only 'Yes' or 'No'.\nAnswer:` | final token, ~165 in |
| `neutral_state` | `Raise.` | `…The player decided to: Raise.` | final token, ~154 in |

Both were checked to be **byte-identical under a label flip**, so neither prompt
can encode the answer. Note that a substring test for the response value is
useless on this dataset: every prompt's instruction contains both `'Yes'` and
`'No'`. The meaningful check is label-independence, and it passes.

Baselines for the neutral variant (model-free, frozen test partition):
prompt-text AUROC 0.7825 / PR-AUC 0.3779, nuisance-only 0.9204 / 0.6166,
response-text 1.0000 / 1.0000 — prompt-text is within 0.001 of the judge variant
because the question is byte-identical in every row and so carries no
discriminative text signal. That confirms the instruction confound is invisible
to any text baseline and can only be measured in the model's hidden states, which
is exactly what the two-variant extraction is for.

Also added: `extraction` now refuses a rendered prompt that does not end at the
declared boundary; SAE configs sized for 4096-dim residuals (32,768 features,
TopK k=64); causal grid `[0.5, 1, 2, 4, 8]` with `causal_eval_size: 1000`.

Tests: **208 passed**.

**Nothing compute-heavy was run.** No model downloaded, no weights loaded, no API
call, no GPU job. The only real computation performed to date is the model-free
baselines and the split construction, both on CPU in under two minutes.
