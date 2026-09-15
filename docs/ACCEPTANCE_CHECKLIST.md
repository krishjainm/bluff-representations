# Review-driven acceptance checklist

- [x] Canonical required schema, duplicate-ID validation, and group-safe fixed splits
- [x] Primary strict path fails on missing, malformed, non-finite, or shape-mismatched activations
- [x] Validation-only layer selection and restored early-stopping checkpoint metrics
- [x] Config, commit, package/environment provenance, repeated seeds, AUROC/PR-AUC, grouped CIs
- [x] Prompt-text, response-text diagnostic, majority, and optional nuisance-only baselines
- [~] Dataset construction, labeling prompts, public examples, and final composition report
      (real dataset adapted + composition report + integrity flags via `poker_adapter.py`;
      **upstream label provenance still undocumented** — see `docs/POKER_DATASET_NOTES.md`)
- [x] Nuisance-only baseline strong enough to be a real test (made-hand alone: AUROC 0.885)
- [x] Instruction confound measurable via a question-stripped prompt variant
- [x] Audited prompt-end/decision-token activation extractor with logged token index and boundary tests
      (`deception_circuits/paper_extraction.py`, `tests/test_extraction_integrity.py`; implemented and tested against stubs — **not yet run on a real subject model**)
- [x] Poker matched subsets and residualized/conditional nuisance analyses
      (`deception_circuits/paper_confounds.py`, `tests/test_confound_controls.py`; every control
      reports `not_run` with a reason when the metadata it needs is absent — **no real metadata yet**)
- [x] Group-safe *and* label-stratified splits with per-partition balance reporting
- [x] Calibration (ECE) in the strict metric set, canonical implementation shared with legacy
- [x] Learning curves over multiple independent group-level subsamples per size
- [x] Seed-level variability reported alongside grouped bootstrap CIs
- [ ] Cross-context/model protocol with source-validation-only selection
- [ ] Behavioral endpoint independent of probe score, paired intervention analysis, and quality controls
- [ ] Positive/negative steering, true patching, shuffled-label, wrong-layer, and timing controls
- [ ] SAE architecture, diagnostics, held-out feature ranking, stability, and example reports
- [ ] Real-artifact-only figures/tables and paper reference cleanup
- [ ] Full pre-paper audit of activation metadata, causal controls, behavioral endpoints, and figures

Unchecked work requires real dataset/model artifacts or a separate implementation pass;
it must not be represented as complete in a manuscript.
