# What the pipeline needs before P1 can be closed

P1 items 2–4 (extraction, provenance, resumability) are implemented and tested.
P1 item 1 — "specify one canonical dataset release" — is the only thing blocking
a real run, and it cannot be fabricated. This document states exactly what the
code will accept, the one design decision you have to make first, and verified
places to get the raw material.

## 0. The decision that determines everything else

The brief's central question is whether manipulating an internal representation
**changes actual model behavior**. That only makes sense if the model is the one
acting. So decide which of these you are building:

### Design A — model as actor (recommended)

You give the subject model a poker situation and it chooses an action. The label
describes *its own* action given the hand it was actually dealt.

- `statement` = the rendered game state (hole cards, board, position, stacks,
  action history), ending at the `Action:` boundary.
- `label` = 1 if the model's chosen aggressive action was a bluff (weak holding),
  0 if it was value (strong holding).
- `response` = the model's actual generated action text.

Why this one: the pre-decision activation genuinely precedes the model's own
decision, and P3's causal endpoint (the action it selects) is native and
independent of the probe. Reviewer A's "not enough evidence that steering changed
deception-specific behavior" is answerable under this design and largely
unanswerable under B.

Cost: you must run the subject model once over all game states to obtain actions
before labels exist. That is a real but modest generation job.

### Design B — model as observer

The model reads a human hand history and the label describes whether *that human*
was bluffing. This is a claim about decoding someone else's strategy from a
transcript, not about the model's own deception, and it leaves P3 with no native
behavioral endpoint. If you pick B, say so plainly in the paper and drop the
causal-behavior claim.

**Recommendation: Design A.** The rest of this document assumes it, but the
schema below works for either.

## 1. The machine-checkable contract

One CSV. `validate_dataset` rejects it otherwise.

### Required — hard failure if absent

| Column | Meaning |
|---|---|
| `sample_id` | Unique per row. Becomes the artifact filename `sample_<id>.pt`, so keep it filesystem-safe. |
| `base_item_id` | Ties paired/related rows together. |
| `statement` | The pre-decision prompt body. **Must not contain the action/response.** |
| `response` | The action text. Never read by the primary extraction path. |
| `label` | 0/1, both classes must be present. |
| `scenario` | Context tag, e.g. `poker`. |

Also enforced: no missing values in required fields, `sample_id` unique, and each
`base_item_id` maps to exactly one `split_group_id`.

### Grouping — get this wrong and the splits leak

`split_group_id` must be **one poker hand / one trajectory**. If you generate
several decision points from the same dealt hand, or a bluff/value pair from the
same board, they share a group. Rows sharing a group can never cross splits.

### Nuisance metadata — each column unlocks a specific P2 analysis

Nothing here is required, but each missing column disables a named analysis, and
the confound report will say so rather than silently skipping it.

| Column | Unlocks |
|---|---|
| `action` | action-matched subset; action-only baseline; bluff-vs-value comparison |
| `street` | street-matched comparison |
| `position` | position-matched comparison |
| `hand_strength` | equity-matched comparison (the single most important control) |
| `board_texture` | board-matched comparison |
| `bet_size`, `pot_size` | bet-sizing control (use the ratio) |
| `solver_action` | strategy-deviation analysis |
| `label_source`, `label_confidence` | label provenance for the dataset card |
| `difficulty_bucket` | stratified split balance |
| `prompt_template_id` | per-row template provenance |
| `metadata_json` | anything else, as a JSON string |

The highest-value control is **bluff raise vs value raise**: same action, same
street, similar bet size, opposite label. If you only have budget to get one
metadata column right, make it `hand_strength` alongside `action`.

## 2. Where to get the raw game states

Verified, permissively licensed:

- **`uoftcprg/phh-dataset`** — poker hand histories in the standardised PHH
  format, **MIT licensed**. Subsets include ~620M ACPC hands, ~21.6M HandHQ
  online hands (July 2009), the 83 televised WSOP 2023 Event #43 final-table
  hands, and **all 10,000 Pluribus hands**. Full release on Zenodo.
- **PHH format spec** — `uoftcprg/phh-std` / phh.readthedocs.io, so you can parse
  actions, streets, positions, bet sizes, and pots programmatically.

Start with the **Pluribus 10,000-hand subset**: it is small enough to iterate on,
it is AI self-play so hole cards are logged throughout rather than only at
showdown, and it avoids the showdown-selection bias that HandHQ has.

I could not confirm that any of these carry bluff labels. **They do not ship the
label you need** — you derive it. Do not plan around finding a labelled
bluff-detection corpus.

## 3. Deriving the label defensibly

Whatever you choose, it must be a documented rule, not a vibe, and it must be
stored in `label_source`.

Preferred: a **mechanical rule** over an aggressive action, using the actual
holding and a computed equity/hand-strength figure — e.g. an aggressive action
with equity below a pre-registered threshold is a bluff, above a second threshold
is value, and the ambiguous middle band is **excluded** rather than forced into a
class. Mechanical rules are fully reproducible, need no judge prompts, and dodge
Reviewer B's "dataset lacked judge prompts" complaint entirely.

Pre-register the thresholds and the exclusion band before looking at any probe
result. Record them in the dataset card.

If you use an LLM judge instead, you must store the exact prompt, model id,
temperature, raw judgement, and parse result per row — and the paid-API rule
still applies, so ask before running it.

## 4. Where the files go

```
data/
  poker_v2.csv                        # the canonical CSV (git-ignored if private)
  poker_v2_dataset_card.md            # construction, label rule, composition
  activations/
    poker_v2_prompt_end/              # written by collect-activations
      sample_<id>.pt
      extraction_spec.json
      extraction_manifest.json
      extraction_manifest.jsonl
```

`data/activations/` and `data/private/` are already git-ignored, as is `*.pt`.
Decide deliberately whether the CSV itself is publishable; Reviewer A's first
complaint was that the dataset was not publicly provided, so prefer a release
that can be shared, or at minimum a public construction script.

## 5. Then run

```bash
cp configs/paper_v2.yaml configs/my_run.yaml   # fill in subject_model, paths
uv run python -m deception_circuits.paper_cli validate-data     --config configs/my_run.yaml
uv run python -m deception_circuits.paper_cli make-splits       --config configs/my_run.yaml
uv run python -m deception_circuits.paper_cli collect-activations --config configs/my_run.yaml --confirm-model-load
uv run python -m deception_circuits.paper_cli analyze-confounds  --config configs/my_run.yaml
uv run python -m deception_circuits.paper_cli train-probes      --config configs/my_run.yaml
uv run python -m deception_circuits.paper_cli audit             --config configs/my_run.yaml
```

`validate-data` and `analyze-confounds` are cheap and need no model, so run them
the moment the CSV exists — before spending any GPU time. `analyze-confounds`
prints which controls your metadata supports and which are unavailable.

## 6. Minimum viable size

Not a hard gate in code, but for the statistics to mean anything:

- enough independent **groups** (hands) that a 70/10/20 group split leaves both
  classes in every partition — the split code now asserts this,
- at least a few hundred held-out rows for P3's causal evaluation, since
  Reviewer A and B both objected to n=100,
- roughly balanced labels within the action-matched cells, because that matched
  subset is the analysis that carries the paper.
