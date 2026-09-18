# Provenance of `normalized_poker_gpt4o.csv` — reconstructed from git

Written 2026-09-17 to answer a teammate's documentation request. Everything below
is **measured from committed artifacts**. Where a step could not be verified it is
marked NOT RECOVERABLE rather than inferred from filenames.

The two commits, both authored by Bruce, on `origin/dataset-integration` (and
`origin/experiments/cross-architecture`):

- `05cf8dd` (2025-10-15) — adds `sample_data/normalized_poker_gpt4o.csv` and
  `schema_check.py`. Nothing else.
- `9695f3c` (2025-10-16) — adds `sample_data/normalized_poker_gpt4o.fixed.csv`.
  Nothing else.

MD5: original `826b32ac7fe1fd3bd440ee80fee87cf5`, fixed
`ea6834a8a5ef438f2b0ed8469905e3e2`.

**No construction script was ever committed.** `schema_check.py` is a validator.

## `.csv` vs `.fixed.csv` — fully verified

| file | columns | rows |
|---|---|---|
| `normalized_poker_gpt4o.csv` | `statement, label, scenario` | 44,631 |
| `normalized_poker_gpt4o.fixed.csv` | `statement, response, label, scenario` | 44,631 |

`statement`, `label` and `scenario` are identical row-for-row. The only change is
the added `response` column, which is `"Yes"`/`"No"` and **equals the label for all
44,631 rows** (1↔Yes: 7,490; 0↔No: 37,141).

Why: `schema_check.py`, committed alongside the original, requires
`["statement","response","label","scenario"]` — the original failed its own
validator, and the fix satisfies the schema by writing the label back as text.

**Consequence:** `response` carries no information beyond `label`. The
response-text baseline returns AUROC 1.000 / PR-AUC 1.000; that is confirmation of
this artifact, not a result. See `docs/POKER_DATASET_NOTES.md` §1.

## Labeling — judge confirmed, exact run not

All 44,631 `statement` values exactly reproduce `build_bluff_prompt()` from
**`llm_judge_openai.ipynb`** (`origin/Google_Collab_Code`, commits `e706fa0` /
`2af65f4`, author wayne-617, 2025-10-10):

```
{instruction.split("Now it is your turn")[0].strip()}

The player decided to: Raise.
Is this a bluff? Reply with only 'Yes' or 'No'.
```

- 44,631 / 44,631 match that format.
- **0 / 44,631** contain the other notebook's few-shot block (`Example 1:`) or its
  `Definition: A bluff is when…` line, so `llm_judge.ipynb` (the Llama-3.1-8B
  judge) is ruled out as the builder of this file.

The judge call, verbatim from that notebook: model `"gpt-4o"` (a bare string
literal — **no dated snapshot pin**, so the served snapshot is unrecoverable),
system prompt `"You are an expert poker analyst."`, `temperature=0.2`,
`max_tokens=5`, 3 retries with exponential backoff. Post-processing maps any reply
that is not exactly `Yes`/`No` via `"Yes" if "yes" in result.lower() else "No"` —
**refusals and malformed replies silently become "No"** — and a final API failure
writes the literal string `"Error"`.

**GPT-4o was the labeler, but not the committed run.** Against
`labeled_bluff_data/is_bluff_openai_gpt-4o.csv`, on the 5,335 joinable rows:
agreement **95.8%, Cohen's κ = 0.794** (4603 No/No, 506 Yes/Yes, 126 + 100
disagreements). High but not identical — consistent with the same judge and prompt
re-run at temperature 0.2, inconsistent with a different labeler. GPT-4o-mini is
ruled out: of 1,247 shipped rows mini called "Yes", the shipped label is 1 for only
508 (41%). Rule fit: gpt-4o alone 95.8%, AND(gpt-4o, mini) 95.1%, OR 85.0%, mini
alone 84.3%.

## Source of the scenarios

**PokerBench (`RZ412/PokerBench`)**, corroborated three independent ways:

- `deception_circuits/dataset_integrations.py:460` and
  `DATA_COLLECTION_GUIDE.md:189` document it as the project's poker source.
- `labeled_bluff_data/is_bluff_openai_gpt-4o.csv` carries an `index` column
  spanning **12 → 563,191** (PokerBench train size) and an `output` column whose
  values are exactly `{fold, call, raise, check, bet}` — PokerBench's solver action
  vocabulary.
- Prompt bodies are the PokerBench 6-max NLHE template verbatim.

NOT RECOVERABLE: split, dataset revision, and the download step (no committed
code fetches it).

The judge's immediate input was `llm_raise.jsonl` (both notebooks read
`/content/llm_raise.jsonl`; commit `e74022e` = "Change input file to
llm_raise.jsonl"), with fields `index`, `instruction`, `output`, `llm_move`.
**`llm_raise.jsonl` is not in the repository**, and which model generated
`llm_move` is recorded nowhere.

## Row selection — the real gap

Measurable:

- 44,631 rows, 44,311 distinct statements, **306 exact duplicate rows** (no dedup
  ran). Duplicate copies sit far apart (median position gap 11,878) — consistent
  with concatenated chunks from an append-mode resumable job.
- Row order is near-monotonic in PokerBench index (**Spearman 0.955**): a
  sequential pass over the source, not a shuffle or random sample.
- Covered index range ≈ 83 → 262,708 (99th pct), tail to 488,350 — roughly the
  first half of PokerBench train.

Decisive negative result:

> **The shipped CSV did not come from `llm_raise.jsonl`.** The notebook's saved
> output records `Loaded 70144 records`, 702 batches, 44m55s, 36,907 Yes /
> 33,237 No — the committed run processed *all* of `llm_raise.jsonl`. Yet only
> **5,267 of 44,311** shipped statements (11.9%) appear in it, and that overlap is
> **0.95× what independent sampling from the 563k PokerBench pool would predict**
> (expected 5,519). The shipped CSV is a *separate* generation + judging pass, not
> a subset or filtering of the one in git.

## Filtering, dedup, human review

- **Deduplication: no** — 306 exact duplicate rows survive.
- **Human review: no evidence** anywhere in git.
- **Filtering: yes, and measurable.** The shipped set is heavily depleted of
  scenarios whose solver action is `fold` — **2.8%** of shipped rows vs **37.1%**
  of non-shipped rows in the labeled file. Fold-spots carry an 82% bluff rate under
  this judge, so that depletion alone explains the shipped positive rate (16.8%)
  versus the committed run's (52.6%). Whether it came from an explicit filter or
  from a better-behaved upstream move generator is NOT RECOVERABLE.
- Uniform by construction: `scenario == "poker"` and stated action `Raise` for all
  44,631 rows.

## Inventory

**Present** (all on `origin/Google_Collab_Code`): `llm_judge_openai.ipynb` (with
saved outputs), `llm_judge.ipynb`, `labeled_bluff_data/is_bluff_openai_gpt-4o.csv`
(70,144 rows), `is_bluff_yes_gpt-4o.jsonl` (36,907),
`is_bluff_yes_gpt-4o-mini.jsonl` (36,067).

**Missing from every branch and commit**: `llm_raise.jsonl`; the `llm_move`
generation script and the identity of the generating model; the normalization
script producing `statement,label,scenario`; the script that added `response`; the
PokerBench split/revision.

## What may be stated in the paper

Supportable today: scenarios from PokerBench; bluff labels from GPT-4o (undated
snapshot) at temperature 0.2, `max_tokens=5`, under the stated system and user
prompts; 44,631 rows at 16.8% positive; no deduplication (306 exact duplicates);
no human review; `response` is the label verbatim.

NOT supportable without recovering the missing Colab: the row-selection rule, the
`llm_move` generating model, the PokerBench split/revision, the exact GPT-4o
snapshot.

**Cheapest fix if nobody can produce it:** re-label the 44,631 shipped statements
with a pinned snapshot and ship that as the dataset of record.

## Reproduce

```bash
git show 05cf8dd:sample_data/normalized_poker_gpt4o.csv       > orig.csv
git show 9695f3c:sample_data/normalized_poker_gpt4o.fixed.csv > fixed.csv
git show origin/Google_Collab_Code:labeled_bluff_data/is_bluff_openai_gpt-4o.csv > labeled.csv
git show origin/Google_Collab_Code:llm_judge_openai.ipynb     > judge.ipynb
```
