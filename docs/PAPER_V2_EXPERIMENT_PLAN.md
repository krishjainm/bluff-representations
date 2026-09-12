# Paper V2 experiment plan

1. **Leakage-safe probing:** one 7B–8B open-weight subject model, pre-decision
   residual activations, fixed group split, five seeds, AUROC and PR-AUC.
2. **Poker controls:** full data plus action-matched bluff/value subsets;
   prompt-only, response-only diagnostic, action-only, and nuisance-only baselines.
3. **Small replication:** repeat the frozen protocol on one or two additional
   models, without selecting layers on target-test performance.
4. **Behavioral intervention:** paired held-out prompts; baseline, positive and
   negative steering, random/orthogonal/shuffled-label/wrong-layer/wrong-token
   controls, and genuine activation replacement. Primary outcome is selected
   action or another independent behavioral criterion, with quality metrics.
5. **SAE only if validated:** unsupervised features, held-out ranking/evaluation,
   reconstruction and sparsity diagnostics, and human-inspected top examples.

Every outcome is informative: a failed controlled probe or null behavioral
effect constrains the claim and must be reported rather than tuned away.
