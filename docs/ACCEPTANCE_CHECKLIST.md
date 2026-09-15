# Review-driven acceptance checklist

- [x] Canonical required schema, duplicate-ID validation, and group-safe fixed splits
- [x] Primary strict path fails on missing, malformed, non-finite, or shape-mismatched activations
- [x] Validation-only layer selection and restored early-stopping checkpoint metrics
- [x] Config, commit, package/environment provenance, repeated seeds, AUROC/PR-AUC, grouped CIs
- [x] Prompt-text, response-text diagnostic, majority, and optional nuisance-only baselines
- [ ] Dataset construction, labeling prompts, public examples, and final composition report
- [x] Audited prompt-end/decision-token activation extractor with logged token index and boundary tests
      (`deception_circuits/paper_extraction.py`, `tests/test_extraction_integrity.py`; implemented and tested against stubs — **not yet run on a real subject model**)
- [ ] Poker matched subsets and residualized/conditional nuisance analyses
- [ ] Cross-context/model protocol with source-validation-only selection
- [ ] Behavioral endpoint independent of probe score, paired intervention analysis, and quality controls
- [ ] Positive/negative steering, true patching, shuffled-label, wrong-layer, and timing controls
- [ ] SAE architecture, diagnostics, held-out feature ranking, stability, and example reports
- [ ] Real-artifact-only figures/tables and paper reference cleanup
- [ ] Full pre-paper audit of activation metadata, causal controls, behavioral endpoints, and figures

Unchecked work requires real dataset/model artifacts or a separate implementation pass;
it must not be represented as complete in a manuscript.
