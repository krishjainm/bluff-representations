# Notes on `normalized_poker_gpt4o` — read before interpreting any result

The repository does contain a real poker dataset. It is not in this working
tree; it lives on `origin/dataset-integration` (commit `9695f3c`):

```
sample_data/normalized_poker_gpt4o.csv        # statement, label, scenario
sample_data/normalized_poker_gpt4o.fixed.csv  # adds a response column
```

Restore and convert it with:

```bash
git checkout origin/dataset-integration -- sample_data/normalized_poker_gpt4o.fixed.csv
python prepare_poker_dataset.py --prompt-variant judge_question
python prepare_poker_dataset.py --prompt-variant neutral_state
```

`deception_circuits/poker_adapter.py` maps it onto the canonical schema and
recovers the nuisance metadata; `tests/test_poker_adapter.py` covers it.

## What it is

44,631 rows, 44,311 distinct statement-derived groups, scenario `poker`, labels
37,141 / 7,490 (**16.8% positive**, so PR-AUC matters more than AUROC). Metadata
parses at 100% coverage: position, street, hole cards, board, amount faced,
made-hand category, board texture.

The 44,311 groups are derived from exact statement text, not original PokerBench
hand/session identifiers. The source hand identifiers and the construction script
for this 44,631-row dataset are unavailable, so these groups must not be described
as 44,311 independently dealt hands. Exact duplicate rows also remain in the
dataset.

Each row is a 6-max No-Limit Hold'em game summary followed by:

```
The player decided to: Raise.
Is this a bluff? Reply with only 'Yes' or 'No'.
```

## Four properties that constrain what it can support

These are measured, not assumed. They do not make the dataset useless — they
make it a dataset for a narrower claim than the previous paper made.

### 1. `response` is the label verbatim

`response` is `"Yes"`/`"No"` and equals the label for **all 44,631 rows**. Any
response-token analysis scores perfectly and means nothing. This is precisely
Reviewer A's "possible trivial probing if the probe reads the label/answer
token", present in the data itself. The strict extractor's pre-decision modes
structurally cannot read it, and `run_baselines` reports the response-text
baseline as a diagnostic — expect it to come back at ~1.0 and treat that as
confirmation of this note, not as a result.

### 2. The prompt names the target concept

Every prompt asks "Is this a bluff?". A probe that finds a linear bluff
direction here is substantially reading the model's computed answer to a question
it was explicitly asked. This is the same class of problem the brief flags for
lie/sandbag-instructed benchmark adapters: **instruction-confounded**.

The adapter's `prompt_variant` exists so this is measurable rather than debated:

- `judge_question` — the prompt as shipped.
- `neutral_state` — the same game summary and stated action with the question
  stripped.

Running both and comparing gives a direct estimate of the confound's size. Report
both numbers.

### 3. The model is a classifier, not the actor

The judged action has already been taken ("The player decided to: Raise"). The
subject model evaluates *someone else's* raise. It never chooses to bluff.

Consequence for P3: there is no task-native endpoint in which the model behaves
deceptively, so the strongest available causal claim is "intervening along the
learned direction changes the model's bluff *judgement*", not "…changes the
model's deceptive *behaviour*". Those are different claims and the second is the
one the previous paper implied. If the team wants the second, they need the
model-as-actor design in `docs/DATASET_REQUIREMENTS.md`.

### 4. The label is close to a deterministic function of hand strength

This is the most important finding, and it is the one most likely to explain the
previous paper's headline number.

Bluff rate by made-hand category:

| Made hand | Bluff rate | n |
|---|---|---|
| quads | 0.000 | 400 |
| trips | 0.006 | 7,541 |
| full house | 0.008 | 3,706 |
| two pair | 0.010 | 8,029 |
| pair | 0.094 | 12,216 |
| straight flush | 0.200 | 105 |
| flush | 0.354 | 4,576 |
| straight | 0.455 | 4,896 |
| high card | 0.734 | 3,162 |

A logistic model on a **9-category one-hot of made-hand alone** reaches
**AUROC 0.885 / PR-AUC 0.551**. Adding street, position, and board texture
reaches **AUROC 0.921 / PR-AUC 0.600**. Base rate is 0.168.

Both figures are in-sample and therefore upper bounds, but a handful of
categorical levels over 44k rows has almost no capacity to overfit, so expect
them to hold out. Reproduce with `analyze-confounds`, which fits the
nuisance-only baseline on train and scores it on test.

**Implication.** A layerwise probe reporting AUROC in the low 0.90s on this data
is not obviously doing better than mechanically evaluating the hand. The paper's
claim cannot be "hidden states decode bluffing" — it has to be "hidden states
decode bluffing *beyond* what made-hand strength and game state already
explain", which requires the nuisance-only baseline and the
`made_hand_matched` control side by side with the probe. `run_confound_suite`
produces exactly that comparison, and `made_hand_matched` is the control that
carries it.

Also note `action` is `raise` for **every** row. The dataset is therefore already
action-matched, which is convenient: `bluff_vs_value` and `action_matched` are
trivially satisfied, and an action-only baseline is at chance by construction.
The confound that remains is game state, not action type.

## 5. Recovered source and label provenance

Git history and the committed intermediate artifacts establish substantial, but
not complete, provenance for this dataset.

### Source examples

The underlying poker examples come from `RZ412/PokerBench` on Hugging Face.
Evidence includes the repository's dataset-integration documentation, the
PokerBench 6-max No-Limit Hold'em prompt format, the solver action vocabulary,
and source-style indices spanning the PokerBench range.

The exact PokerBench split and dataset revision are not recoverable. No committed
script downloads the source dataset.

The historical GPT-4o judge consumed an intermediate file named
`llm_raise.jsonl` containing `index`, `instruction`, `output`, and `llm_move`.
That file is not present on any recovered branch, and the model or procedure that
generated `llm_move` is not recoverable.

### Bluff-judgement prompt

All 44,631 shipped statements reproduce the GPT-4o judge prompt structure:

```text
{instruction before "Now it is your turn"}

The player decided to: Raise.
Is this a bluff? Reply with only 'Yes' or 'No'.

## What is still missing

- **Equity.** `made_hand_rank` is made-hand *category*, not equity: a strong draw
  scores as high card, which is why `straight`/`flush` sit at 35–46% bluff rather
  than near zero. Proper equity needs a solver or an enumeration; until then
  `equity_matched` correctly reports `not_run`.
- **Pot size.** Only the amount *faced* is parsed reliably. Reconstructing the
  pot needs a full action-sequence replay including calls, which is not
  implemented, so `bet_size_matched` reports `not_run`.
- **Exact source lineage.** PokerBench is recoverable as the source dataset, but
  the exact split, dataset revision, source hand/session identifiers, and final
  44,631-row selection script are not.
- **Upstream action-generation lineage.** The historical judge input
  `llm_raise.jsonl` and the model/procedure that generated its `llm_move` field
  are not recoverable.
- **Exact GPT-4o snapshot.** The recovered judge used the bare model name
  `gpt-4o`, not a dated snapshot, so the exact served model version cannot be
  reproduced.
- **Independent-hand identity.** `split_group_id` groups exact statement-derived
  items. Without original hand/session identifiers, it does not prove that
  different groups correspond to independent dealt hands.

## Honest framing this data can support

> After controlling for made-hand strength, street, position, and board texture,
> do a model's pre-decision hidden states carry information about whether a
> described aggressive action was a bluff, beyond what those variables explain?

That is a real, publishable question with a genuine confound story and a
pre-registered control set. It is narrower than "we found the deception circuit",
and it is defensible.
