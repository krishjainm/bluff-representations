# Paper claim guardrails

Probe evidence can support: “the tested bluff labels are linearly decodable from
the specified hidden states under this protocol.” It cannot by itself support a
claim that a deception circuit was found.

With controlled paired behavioral interventions, it can support: “intervening
along the learned direction changed the specified bluff-related behavior under
the tested conditions.” It still cannot establish a universal deception mechanism.

Use **positive/negative steering** for vector addition/subtraction, and
**activation replacement (patching)** only when an activation is actually
replaced. Avoid “reasoning traces” unless the study genuinely analyzes them.

## Judgement is not behaviour

On the `normalized_poker_gpt4o` dataset the subject model does not act — it is
asked whether an already-taken raise was a bluff. So even a clean, fully
controlled intervention result licenses only:

> Intervening along the learned direction changes the model's **bluff
> judgement** under the tested conditions.

It does **not** license:

> Intervening along the learned direction changes the model's **deceptive
> behaviour**.

The second requires the model to be the one choosing to bluff. See
`docs/DATASET_REQUIREMENTS.md` (Design A) and `docs/POKER_DATASET_NOTES.md`.

## Claims that require the nuisance baseline stated alongside them

Made-hand category alone reaches AUROC 0.885 on this dataset's label. Therefore
"hidden states decode bluffing" is not a publishable claim here; the only
defensible form names the comparison:

> After controlling for made-hand strength, street, position, and board texture,
> pre-decision hidden states carry information about the bluff label **beyond**
> what those variables explain (probe AUROC X vs nuisance-only baseline Y, both
> on the frozen test partition).

Any headline probe number printed without Y next to it is misleading on this
dataset.

## Instruction confound must be quantified, not waved away

Every source prompt asks "Is this a bluff?". Report both `judge_question` and
`neutral_state` prompt variants, or state plainly that the result is
instruction-confounded and that the neutral comparison was not run.

## The endpoint is the model's output, never the probe score

`paper_causal.assert_endpoint_independence` refuses probe-derived endpoints, and
the audit fails a causal run that does not declare a probe-independent primary
endpoint. `diagnostic_probe_projection` appears in the per-row records and is
named that way deliberately: it is a diagnostic, and reporting it as the causal
result would be circular.

## Control conditions are not optional decoration

A signed steering effect means little without the controls beside it. Report
every condition the suite emits — random matched-norm, orthogonal,
shuffled-label, wrong-layer, wrong-token-position, mismatched-prompt patching,
and a nuisance direction where metadata allows — with paired bootstrap CIs and
the quality-control metrics. If a steering effect is not clearly larger than the
random and orthogonal controls, say so.

## Dose and units

Report intervention strength in units of the hidden-state norm
(`relative_magnitude` in the records), not raw alpha. An effect that only appears
at a magnitude comparable to the hidden state itself is a perturbation result,
not evidence of a targeted mechanism, and should be described as such.
