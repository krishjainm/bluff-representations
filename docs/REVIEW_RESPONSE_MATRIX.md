# Review response matrix

| Concern | Final status | Evidence / interpretation |
|---|---|---|
| Reproducible dataset schema and fixed splits | complete | Strict schema validation, checksum enforcement, grouped and label-stratified frozen split. Final split: 31,241 train / 4,463 validation / 8,927 test. |
| Silent fake activations | complete | Strict activation loader hard-fails on missing, malformed, non-finite, shape-mismatched, or provenance-inconsistent artifacts. |
| Answer-token leakage / unclear activation location | complete real run | Both real extractions use pre-response prompt-end activations. Per-sample manifests record token position, physical-layer mapping, prompt hash, tensor hash, model/tokenizer revision, and activation shape. Response label text is never supplied to the extractor. |
| Test-set layer selection | complete | Layer selection is validation-only. Explicit prompt selects physical layer 30 in all five runs; neutral prompt selects layer 31 in all five runs. |
| Repeated runs and metrics | complete real run | Five probe seeds per prompt variant, grouped bootstrap uncertainty, AUROC, PR-AUC, ECE, seed summaries, and selected-layer stability are recorded. Explicit test AUROC 0.9360; neutral test AUROC 0.9351. |
| Poker confounds and nuisance baselines | complete real run | Nuisance-only judge AUROC 0.9204. Residualization reduces AUROC from 0.9360 to 0.7220. Made-hand matching reduces AUROC to 0.8426. Street, position, board texture, board pairing, made hand, and amount faced are themselves strongly decodable. Equity and reliable pot-size controls remain `not_run` because the required metadata is unavailable. |
| Cross-context transfer | implementation complete, real evidence unavailable | Source-train fitting and source-validation-only layer selection are enforced and tested. Target-test labels cannot affect selection. No real transfer matrix is claimed because the final strict protocol contains only one real context. |
| Behavioral causal endpoint and controls | complete real run | Real live-model intervention suite completed on 500 held-out prompts. Primary endpoint is the subject model's own forced-choice Yes/No distribution, never a probe score. Probe-direction steering changes the judgement endpoint, but matched random, shuffled-label, and nuisance directions can produce comparable or larger changes. Supported claim: causal modulation without demonstrated target-direction specificity. |
| Steering versus activation replacement | complete | Steering and activation replacement are distinct intervention types. Patching uses a real held-out activation source and has no steering dose grid. |
| Intervention control coverage | complete real run | Positive/negative probe steering, random matched-norm, orthogonal, shuffled-label, wrong-layer, wrong-token-position, mismatched-prompt patching, and made-hand nuisance controls were evaluated. |
| Exact intervention timing and dose | complete real run | Physical layer, site, token/timing policy, strength, direction source/norm, hidden norm, and relative magnitude are recorded. Real dose grid: 0.5, 1, 2, 4. |
| Only 100 intervention examples | resolved | Strict real causal evaluation uses 500 held-out examples. |
| Quality degradation versus behavioral change | complete real run | Valid Yes/No probability mass stays near baseline for intended steering and most controls. Wrong-layer strength 4 collapses valid-choice mass and is treated as degeneration rather than mechanistic evidence. |
| SAE diagnostics and feature examples | complete real run for primary condition | Unsupervised TopK SAE trained at the validation-selected explicit-judgement layer. Test variance explained approximately 0.9517, mean L0 approximately 64, dead-feature fraction approximately 0.7558, and cross-seed dictionary stability approximately 0.1009. Selected single features reach only about 0.42 to 0.59 held-out AUROC. No stable interpretable deception-feature claim is made. |
| Neutral-state SAE | intentionally not run | Neutral-state is an instruction-removal probe ablation, not a second SAE study. Its configuration now explicitly disables SAE requirements and its strict audit passes. |
| Figures from real artifacts | complete | Figure builders consume only real result artifacts, emit source CSVs, and record source checksums in manifests. Real judge and neutral figure suites are included in the canonical manuscript. Cross-context remains explicitly `not_run`. |
| Concrete examples, labeling prompts, composition | substantially resolved with explicit lineage limits | The real 44,631-row dataset is adapted and characterized. Historical repository analysis recovered the GPT-4o poker-analysis labeling procedure and prompt. Exact PokerBench revision/split, original hand/session IDs, final-row selection script, upstream action-generation procedure, and exact served GPT-4o snapshot remain unavailable and are disclosed. |
| Early stopping and stale metrics | complete | Legacy checkpoint restoration issue fixed; strict result path recomputes metrics from frozen selected artifacts. |
| Calibration, PR-AUC, and uncertainty | complete real run | PR-AUC, ECE, grouped bootstrap CIs, repeated seeds, and five-draw group-level learning curves are produced on real data. |
| Prompt names the target concept | resolved experimentally | Question-stripped neutral-state variant reaches test AUROC 0.9351 versus 0.9360 for the explicit judgement prompt, showing that the headline decodability does not depend on the explicit bluff question alone. |
| Probe may be reading poker strategy rather than deception | confirmed as a major confound and reported | Nuisance-only AUROC 0.9204, residualized AUROC 0.7220, and made-hand-matched AUROC 0.8426 show that much of the signal overlaps ordinary poker-state information. Manuscript explicitly avoids interpreting the raw probe as a clean deception representation. |
| Overstated terminology / reasoning-trace framing | resolved in manuscript | Final manuscript studies bluff judgement, not model deceptive behavior. It does not use reasoning-trace framing and does not claim a universal deception circuit. |
| SAE disconnected from layerwise probing | complete | SAE consumes the validation-selected probe layer and cannot run before probe results exist. |
| Best-probe-layer vs final-layer SAE inconsistency | complete | Real SAE uses the selected explicit-judgement physical layer 30. |
| L1 SAE choice not justified against modern practice | resolved | Strict path uses an explicit TopK dictionary with k=64 and 32,768 features. |
| Supervised autoencoder mistaken for unsupervised discovery | resolved | Legacy supervised artifact documented in `docs/SAE_AUDIT.md`. Strict SAE training is unsupervised and frozen before held-out evaluation. |
| SAE width mismatch | resolved | Strict SAE width is explicitly configured and reported as 32,768 features, an 8x expansion over the 4,096-dimensional residual stream. |
| Variable naming overloaded `w` | resolved | Strict path uses named direction objects and explicit direction source/provenance fields. |
| Figures unclear / hard to interpret | resolved for canonical manuscript | Final figures use real artifacts, source CSVs, explicit model/split/layer provenance, named baselines/reference lines, and scientifically corrected statement-derived-group wording. Venue-specific formatting is intentionally deferred until a submission venue is selected. |
| References / citation formatting | resolved for canonical manuscript | Canonical LaTeX manuscript compiles with BibTeX, no unresolved citations, and no substantive LaTeX warnings. Venue-specific bibliography formatting remains template-dependent. |
| Reproducibility and artifact traceability | complete in strict path | Dataset checksum, split manifest, model/tokenizer revisions, activation-layer mapping, prompt hashes, artifact hashes, environment metadata, figure source data, and figure manifests are recorded. |
| Full strict audit | complete | Explicit-judgement audit passes. Neutral-state audit passes after explicitly disabling its intentionally unused SAE stage. Missing equity and pot-size metadata appear as audit notes rather than silent omissions. |
| Full regression suite | complete | Final local suite: 254 tests passing. |

## Final evidence boundary

The strict real-run evidence supports the following narrow conclusions:

1. The operational poker bluff label is strongly linearly decodable from late Llama-3.1-8B hidden states.
2. Removing the explicit bluff question leaves raw probe performance almost unchanged.
3. Poker-state nuisance variables explain a large fraction of the apparent label signal.
4. Live-model interventions can causally modulate the model's bluff judgement, but the present control suite does not establish that the learned probe direction is uniquely bluff-specific.
5. The SAE reconstructs the selected activation space well, but individual features are weak and unstable as standalone bluff-label predictors.
6. The experiments do not establish deceptive intent, deceptive behavior, a universal deception circuit, monosemantic deception features, or cross-context generalization.

The only substantive evidence gaps intentionally retained are unavailable equity/pot-size controls, incomplete upstream dataset lineage, the absence of a second strict real context, and the lack of a model-as-actor behavioral experiment.
