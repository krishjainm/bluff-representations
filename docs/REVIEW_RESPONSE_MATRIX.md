# Review response matrix

| Concern | Status | Evidence / remaining work |
|---|---|---|
| Reproducible dataset schema and fixed splits | fixed in V2 probe path | `paper.py`, protocol, manifest checksum; splits are now group-safe **and** label-stratified with a per-partition balance summary, and a single-class partition is a hard error |
| Silent fake activations | fixed in V2 probe path | strict activation loader hard-fails |
| Answer-token leakage / unclear location | fixed in code, blocked on real run | `paper_extraction.py` pre-decision modes never receive the `response` column; per-sample `token_index` is recorded in `extraction_manifest.json`; `tests/test_extraction_integrity.py` proves response text cannot enter the path. Real artifacts still need to be extracted. |
| Test-set layer selection | fixed in V2 probe path | selection is validation-only and audit checks it |
| Repeated runs and metrics | fixed in code, blocked on real run | configurable seeds; full metric set incl. ECE; `seed_summary` reports mean/std/median/IQR and selected-layer stability alongside grouped bootstrap CIs (`paper.py:summarize_across_seeds`, `tests/test_splits_and_curves.py`) |
| Poker confounds and nuisance baselines | fixed in code, blocked on real metadata | `paper_confounds.py` implements bluff-vs-value, action/street/position/board/bet-size/equity matching, typed nuisance decodability (logistic for categorical, ridge for continuous), and train-fitted residualisation reporting unadjusted vs controlled AUROC. Each control reports `not_run` with a reason when metadata is absent; `analyze-confounds` CLI + 30 tests. |
| Cross-context transfer | not yet implemented | must select on source validation only |
| Behavioral causal endpoint and controls | not yet implemented | legacy controls are experimental and not paper-valid |
| Steering versus activation replacement | paper-only / legacy audit required | terminology guardrail needed before paper drafting |
| SAE diagnostics and feature examples | not yet implemented | do not make SAE claims from legacy code |
| Figures from real artifacts | not yet implemented | no publication figures should be generated yet |
| Concrete examples, labeling prompts, composition | blocked on real dataset | requirements, sourcing options, and label-provenance rules specified in `docs/DATASET_REQUIREMENTS.md`; store with the data release |
| Early stopping and stale metrics | fixed in legacy probe path | deep copied checkpoint; metrics recomputed after restore |
| Calibration, PR-AUC, and uncertainty | fixed in code, blocked on real run | PR-AUC, grouped bootstrap CIs, ECE in `compute_binary_metrics`, and group-level learning curves with multiple independent subsamples per size (`paper.py:run_learning_curve`) |
| Exact intervention timing and dose | not yet implemented | must be config-recorded and evaluated on held-out paired prompts |
| Quality degradation versus behavioral change | not yet implemented | task validity/refusal/repetition/quality must accompany causal results |
| Overstated terminology / reasoning-trace framing | partially fixed | README guardrail; manuscript rewrite remains |
