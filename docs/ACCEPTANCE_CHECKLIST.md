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
- [x] Behavioral endpoint independent of probe score, paired intervention analysis, and quality controls
      (`deception_circuits/paper_causal.py`, `tests/test_causal_suite.py`; **stub-tested only, not yet run on a real model**)
- [x] Positive/negative steering, true patching, shuffled-label, wrong-layer, and timing controls
      (audit fails a causal run missing any of them)
- [x] Configurable held-out causal evaluation size (default 300, not 100)
- [x] Dose-response grid with intervention magnitude in hidden-state-norm units
- [x] SAE architecture, diagnostics, held-out feature ranking, stability, and example reports
      (`deception_circuits/paper_sae.py`, `tests/test_sae.py`; **stub-tested only**)
- [x] SAE trained unsupervised, frozen before held-out evaluation, audit asserts both
- [x] Legacy supervised-autoencoder artifact identified and documented (`docs/SAE_AUDIT.md`)
- [x] Real-artifact-only figures with source data and traceable checksums
      (`deception_circuits/paper_figures.py`, `tests/test_figures.py`)
- [ ] Paper reference/citation cleanup (**paper-only, cannot be done in code**)
- [x] Full pre-paper audit of activation metadata, causal controls, behavioral endpoints, and figures
      (`deception-paper audit`; fails on any missing required artifact, separates notes from failures)
- [ ] Cross-context generalization matrix (**the one remaining unimplemented analysis**)

Unchecked work requires real dataset/model artifacts or a separate implementation pass;
it must not be represented as complete in a manuscript.
