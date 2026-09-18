# Real-run results — Llama-3.1-8B-Instruct and Mistral-7B-v0.3

Regenerate every table here with `python scripts/summarize_run.py`. Numbers are
read from the committed artifacts under `results/`; nothing is transcribed.

Run performed on a rented A100-40GB. Four extractions (2 models × 2 prompt
variants), pre-decision `prompt_end`, bfloat16, pinned revisions, per-sample
provenance manifests. 44,631 rows; group-safe label-stratified split
31,241 / 4,463 / 8,927; test positive rate 0.1678.

## 1. Probe vs the nuisance baseline

| variant | model | layer | AUROC | PR-AUC | nuisance-only | delta | residualized | made-hand-matched |
|---|---|---|---|---|---|---|---|---|
| judge_question | llama31_8b | 30 | 0.9363 | 0.7322 | 0.9204 | **+0.0159** | 0.7220 | 0.8456 |
| judge_question | mistral7b_v03 | 31 | 0.9494 | 0.7781 | 0.9204 | **+0.0290** | 0.7563 | 0.8643 |
| neutral_state | llama31_8b | 30 | 0.9355 | 0.7222 | 0.9204 | **+0.0151** | 0.7334 | 0.8244 |
| neutral_state | mistral7b_v03 | 31 | 0.9338 | 0.7368 | 0.9204 | **+0.0134** | 0.7138 | 0.8165 |

The probe beats a logistic model on game-state metadata alone by **1.3–2.9 AUROC
points**. On PR-AUC the margin is larger (0.616 → 0.72–0.78), which matters more
at a 0.168 base rate. Any headline printed without the 0.9204 next to it is
misleading — see `docs/PAPER_CLAIM_GUARDRAILS.md`.

## 2. The instruction confound is small

| model | judge_question | neutral_state | difference |
|---|---|---|---|
| llama31_8b | 0.9363 | 0.9355 | **+0.0008** |
| mistral7b_v03 | 0.9494 | 0.9338 | **+0.0156** |

Stripping "Is this a bluff?" costs Llama essentially nothing and Mistral 1.6
points. So the probe is **not** mostly reading the model's answer to a question it
was asked. This was the reviewer objection most likely to sink the paper, and it
is now answered with a measurement rather than an argument.

## 3. What the hidden states actually encode

Same layer, same activations, decoding game state instead of the label:

| target | llama31_8b | mistral7b_v03 | majority baseline |
|---|---|---|---|
| street | 1.0000 | 1.0000 | 0.4890 |
| position | 0.9998 | 1.0000 | 0.2887 |
| board_paired | 0.9673 | 0.9868 | 0.7061 |
| board_texture | 0.9162 | 0.9327 | 0.5957 |
| made_hand | 0.7530 | 0.8589 | 0.2672 |
| bet_size (R²) | 0.9661 | 0.9609 | — |

The residual stream at the selected layer encodes the **entire game state almost
perfectly**. After train-fitted residualisation against those variables the probe
falls from 0.9363 to 0.7220 (Llama; drop 0.2144) and 0.9494 to 0.7563 (Mistral).

**This is the paper's central honest finding.** Most of the probe's performance is
game-state information, not a bluff-specific representation. What survives
residualisation (0.72–0.76, well above 0.5) and the made-hand-matched control
(0.82–0.86) is real but modest, and `made_hand` is category, not equity — a strong
draw scores as high card — so the matched control is weaker than it looks.

## 4. Causal suite — a negative result

Primary endpoint is the model's own forced-choice answer distribution, never the
probe score. Baseline valid-choice mass 0.980 / 0.978 (floor 0.50), so the
endpoint is live. 500 paired held-out prompts, grouped bootstrap CIs.

Effects on P(positive) at the largest dose:

| condition | llama31_8b | mistral7b_v03 |
|---|---|---|
| **positive_steering** | **−0.0192** | **+0.0159** |
| negative_steering | +0.0186 | −0.0125 |
| random_matched_norm | +0.0468 | −0.0096 |
| orthogonal | +0.0087 | −0.0320 |
| shuffled_label | +0.0273 | −0.0064 |
| wrong_layer | +0.2431 | +0.2848 |
| wrong_token_position | −0.0006 | +0.0000 |
| activation_patch_mismatched_prompt | +0.0000 | +0.0048 |
| nuisance_steering (made_hand) | −0.0301 | +0.0100 |

Read this honestly:

1. **The steering effect is not larger than its controls.** For Llama, random
   matched-norm (+0.047), shuffled-label (+0.027) and the nuisance direction
   (−0.030) all move the endpoint *more* than the probe direction (−0.019). For
   Mistral the orthogonal control (−0.032) beats the probe direction (+0.016).
   There is no evidence of a targeted causal mechanism here.
2. **The sign is inconsistent across models.** Adding the direction raises
   P("Yes") for Mistral and lowers it for Llama.
3. **Not one discrete choice flipped.** Both models answer "No" on essentially
   every prompt at every dose (`pos_rate` 0.000–0.002).
4. **The dose grid is far too small.** `mean_relative_magnitude` at the largest
   strength is 0.087 (Llama) and **0.011** (Mistral) — the perturbation is 1–9% of
   the hidden-state norm. We never tested a dose capable of changing behaviour.
   Strength must be specified in units of relative magnitude, not raw alpha; the
   grid was trimmed to raw `[0.5, 1, 2, 4]` to make the run fit.
5. **`wrong_layer` dominates everything** (+0.24 / +0.28, flipping 95% / 82% of
   choices). A control condition producing an order-of-magnitude larger effect
   than the experimental condition is the signature of generic perturbation.

Defensible claim: *under this protocol, at doses up to 9% of the hidden-state
norm, steering along the learned direction did not change the model's bluff
judgement more than random, orthogonal, or shuffled-label controls.* The causal
question is **open**, not answered, and a re-run on a relative-magnitude dose grid
is the single highest-value next experiment.

## 5. SAE — an honest negative

| variant | model | var. explained | L0 | dead | rankable features |
|---|---|---|---|---|---|
| judge_question | llama31_8b | 0.9517 | 64 | 0.756 | 743 / 32768 |
| judge_question | mistral7b_v03 | 0.9669 | 64 | 0.613 | 574 / 32768 |
| neutral_state | llama31_8b | 0.9803 | 64 | 0.752 | 1196 / 32768 |
| neutral_state | mistral7b_v03 | 0.9805 | 64 | 0.804 | 1209 / 32768 |

Reconstruction is good and L0 is exactly 64 (TopK, as configured), but 61–80% of
features are dead and **only ~2–4% are frequent enough to rank at all**. No
interpretable bluff feature was found. Report this as a negative result; do not
mine 32,768 features for one that correlates.

## 6. Known methodological weaknesses in this run

- **`n_seeds: 5` but seed std is exactly 0.0** on every metric. The splits are
  fixed by manifest and the probe is deterministic, so the seeds vary nothing.
  Either vary something real (split resampling, probe init) or report n=1 and drop
  the seed language. A reviewer will catch this.
- **Causal dose grid under-powered** — see §4.4.
- **`equity_matched` and `bet_size_matched` still `not_run`** — blocked on true
  equity and pot size, which need a solver or full action replay.
- **Cross-context generalization: not implemented.** Needs a second deception
  context.
- **Upstream label provenance** is partially reconstructed — see
  `docs/DATASET_PROVENANCE.md`. The labeling run behind the shipped CSV is not in
  the repository.
