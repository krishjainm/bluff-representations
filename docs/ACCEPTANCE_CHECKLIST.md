# Review-driven acceptance checklist

- [x] Canonical required schema, duplicate-ID validation, and group-safe fixed splits
- [x] Primary strict path fails on missing, malformed, non-finite, or shape-mismatched activations
- [x] Validation-only layer selection and restored early-stopping checkpoint metrics
- [x] Config, commit, package/environment provenance, repeated seeds, AUROC/PR-AUC, grouped CIs
- [x] Prompt-text, response-text diagnostic, majority, and nuisance-only baselines
- [~] Dataset construction, labeling prompts, public examples, and final composition report
      - Real dataset adapted and characterized with 44,631 rows and 44,311 statement-derived groups.
      - Historical Git archaeology recovered the GPT-4o poker-analysis labeling procedure and prompt.
      - Exact upstream PokerBench split/revision, final 44,631-row selection script, original hand/session IDs,
        upstream action-generation procedure, and exact served GPT-4o snapshot remain unrecoverable.
      - These lineage limits are stated explicitly in `docs/POKER_DATASET_NOTES.md` and the manuscript.
- [x] Nuisance-only baseline strong enough to be a real test
      - Strict judge run nuisance-only AUROC 0.9204 / PR-AUC 0.6166.
      - Historical made-hand-only diagnostic AUROC approximately 0.885.
- [x] Instruction confound measured with a question-stripped prompt variant
      - Explicit prompt test AUROC 0.9360.
      - Neutral-state test AUROC 0.9351.
- [x] Audited prompt-end activation extractor with logged token index, hashes, physical-layer mapping, and boundary tests
      - Real extraction complete for both prompt variants.
      - 44,631 activation tensors per variant, each with shape [32, 4096].
- [x] Poker matched subsets and residualized/conditional nuisance analyses
      - Real judge confound run complete.
      - Unadjusted AUROC 0.9360; residualized AUROC 0.7220.
      - Made-hand-matched AUROC 0.8426.
      - `bet_size_matched` and `equity_matched` correctly report `not_run` because required metadata is unavailable.
- [x] Group-safe and label-stratified frozen splits with per-partition balance reporting
      - 31,241 train / 4,463 validation / 8,927 test.
- [x] Calibration (ECE) in the strict metric set
- [x] Learning curves over five independent group-level subsamples per size
      - Group provenance is explicitly described as statement-derived groups, not verified independent poker hands.
- [x] Seed-level variability reported alongside grouped bootstrap CIs
      - Five real probe runs completed for each prompt variant.
- [~] Cross-context/model transfer protocol implemented with source-validation-only selection
      - `paper_transfer.py` and `tests/test_transfer.py` enforce target-test exclusion from selection.
      - Real cross-context transfer is not claimed because the strict real protocol contains only one real context.
- [x] Behavioral endpoint independent of probe score, paired intervention analysis, and quality controls
      - Real causal run completed on 500 held-out explicit-judgement examples.
      - Primary endpoint is the subject model's own Yes/No output distribution.
- [x] Positive/negative steering, random matched-norm, orthogonal, shuffled-label, wrong-layer, wrong-token,
      mismatched-prompt patching, and nuisance-direction controls
- [x] Configurable held-out causal evaluation size
      - Strict real run used 500 examples rather than the previously criticized 100.
- [x] Dose-response grid with intervention magnitude recorded in hidden-state-norm units
      - Real strengths: 0.5, 1, 2, 4.
- [x] Causal quality diagnostics distinguish endpoint movement from output degradation
      - Valid-choice mass remains near baseline for intended steering and most controls.
      - Wrong-layer strength 4 is explicitly treated as degenerate because valid-choice mass collapses.
- [x] SAE architecture, diagnostics, held-out feature ranking, stability, and example reports
      - Real SAE run completed for the primary explicit-judgement condition at physical layer 30.
      - Test variance explained approximately 0.9517.
      - Mean L0 approximately 64.
      - Dead-feature fraction approximately 0.7558.
      - Mean cross-seed maximum cosine similarity approximately 0.1009.
      - Held-out single-feature AUROCs are modest, roughly 0.42 to 0.59.
- [x] SAE trained unsupervised and frozen before held-out evaluation
- [x] Neutral-state SAE deliberately disabled because the neutral condition is an instruction-removal probe ablation,
      not a second SAE experiment
- [x] Legacy supervised-autoencoder artifact identified and documented in `docs/SAE_AUDIT.md`
- [x] Real-artifact-only figures with source CSVs and traceable checksums
      - Judge figure suite generated from strict real artifacts.
      - Neutral figure suite generated from strict real artifacts.
      - Cross-context figure correctly remains `not_run`.
- [x] Paper reference/citation cleanup
      - Canonical manuscript compiles successfully with BibTeX and no unresolved citations.
- [x] Manuscript terminology rewritten to match supported claims
      - No reasoning-trace framing.
      - No claim of a universal deception circuit.
      - Probe result described as bluff-label decodability.
      - Causal result described as causal modulation without demonstrated direction specificity.
- [x] Full pre-paper audit of activation metadata, confound controls, causal controls, behavioral endpoints, SAE,
      and figures
      - Explicit-judgement strict audit: PASS.
      - Neutral-state strict audit: PASS after disabling the intentionally unused neutral SAE requirement.
      - Missing equity and reliable pot-size metadata remain explicit audit notes, not hidden omissions.
- [x] Full local regression suite
      - 254 tests passing on the final strict code path.
- [~] Cross-context generalization matrix
      - Implementation and audit support are complete.
      - A real matrix is intentionally not reported because there is only one strict real context.

Remaining work before archival completion is operational rather than scientific: preserve the large ignored activation
artifacts and SAE checkpoint outside ephemeral compute, update the repository default branch, and create a final
release/checkpoint. Missing metadata controls and unavailable cross-context evidence must continue to be represented
as `not_run`, not as completed experiments.
