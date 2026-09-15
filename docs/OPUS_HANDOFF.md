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
deception-paper train-probes --config my_run.yaml
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
