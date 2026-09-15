# GPU runbook: the real Llama-3.1-8B run

Everything that does not need a subject model has already been run and committed
(see the dated entries in `docs/OPUS_HANDOFF.md`). This file is the turnkey
sequence for the part that needs rented compute.

## Why this is not run locally

The development machine has 16 GB unified memory, MPS, no CUDA. Llama-3.1-8B in
fp16 is ~16 GB of weights alone, so it does not fit alongside the OS and the
activation buffers. It is also a gated repo.

## What you need

| Resource | Requirement | Why |
|---|---|---|
| VRAM | ≥ 20 GB | bf16 weights (~16 GB) plus activation buffers. The configs set `torch_dtype: bfloat16`; fp32 would be ~32 GB. An A100 (40 or 80 GB) has ample headroom. |
| Disk | ~50 GB free | 23.4 GB of activations per prompt variant, two variants |
| Hugging Face | a token with Llama-3.1 access | the repo is gated |
| Time | measured: 44,631 forward passes, ~9.0M tokens of prefill, per variant | no generation is involved in extraction |

A single 24 GB card (A10G, L4, 3090, 4090) is sufficient. No multi-GPU needed.

### Throughput note

Extraction is **one forward pass per sample, unbatched** — deliberately, because
the recorded `token_index` must refer to an unpadded position and batching would
introduce padding into exactly the place the leakage guarantees live. On an A100
expect roughly 30–60 ms per prompt, so about **35–50 minutes per variant**
(44,631 prompts), or ~1.5 h for both. That is dominated by per-call overhead
rather than compute, so a batched implementation would be several times faster if
the run ever needs to scale up — but it would need careful left-padding index
handling and a new round of boundary tests. At this dataset size it is not worth
the correctness risk.

## Step 0 — environment

```bash
git clone <this repo> && cd deception-llms
git checkout conference-rebuild
uv sync
huggingface-cli login          # required: Llama-3.1 is gated
```

## Step 1 — rebuild the dataset

The raw CSV lives on `origin/dataset-integration`, not in this branch, and the
adapted CSVs are git-ignored because they are ~44 MB each. Regenerate them:

```bash
mkdir -p data/derived
git show 9695f3c:sample_data/normalized_poker_gpt4o.fixed.csv \
  > data/derived/normalized_poker_gpt4o.fixed.csv

uv run python prepare_poker_dataset.py \
  --source data/derived/normalized_poker_gpt4o.fixed.csv --prompt-variant judge_question
uv run python prepare_poker_dataset.py \
  --source data/derived/normalized_poker_gpt4o.fixed.csv --prompt-variant neutral_state
```

Expect `44631 rows, 44311 groups` for both, and these integrity flags:
`response_is_label_verbatim: True`, `single_action_category: True`,
`equity_available: False`.

## Step 2 — pin the model revision

Both configs ship `model_revision: REPLACE_WITH_PINNED_COMMIT_SHA`. **Replace it
with the actual commit SHA** before running. The extraction manifest records
whatever is in the config, so an unpinned id produces artifacts that cannot be
reproduced later. Get it from the model page or:

```bash
python -c "from huggingface_hub import HfApi; \
print(HfApi().model_info('meta-llama/Llama-3.1-8B-Instruct').sha)"
```

Set the same value for `tokenizer_revision`.

## Step 3 — the primary variant

```bash
CFG=configs/real_llama31_8b_judge_question.yaml
uv run python -m deception_circuits.paper_cli validate-data        --config $CFG
uv run python -m deception_circuits.paper_cli make-splits          --config $CFG
uv run python -m deception_circuits.paper_cli run-baselines        --config $CFG
uv run python -m deception_circuits.paper_cli collect-activations  --config $CFG \
      --device cuda --confirm-model-load
uv run python -m deception_circuits.paper_cli train-probes         --config $CFG
uv run python -m deception_circuits.paper_cli analyze-confounds    --config $CFG
uv run python -m deception_circuits.paper_cli train-sae            --config $CFG
uv run python -m deception_circuits.paper_cli run-interventions    --config $CFG \
      --device cuda --confirm-model-load
uv run python -m deception_circuits.paper_cli make-figures         --config $CFG
uv run python -m deception_circuits.paper_cli audit                --config $CFG
```

`collect-activations` and `run-interventions` refuse to run without
`--confirm-model-load` and print what they would load first. Extraction is
resumable: if it dies, re-run the same command and it picks up from the last
committed sample. It will refuse to write into a directory whose recorded
extraction spec differs, so a changed config gets a new `activation_dir` rather
than silently mixed artifacts.

Expected from `run-baselines` (already verified locally — if these differ, the
dataset regeneration went wrong):

```
majority_class_accuracy      0.8322
prompt_text                  AUROC 0.7817  PR-AUC 0.3762
response_text_diagnostic     AUROC 1.0000  PR-AUC 1.0000
nuisance_only                AUROC 0.9204  PR-AUC 0.6166
```

## Step 4 — the second variant (instruction confound)

```bash
CFG=configs/real_llama31_8b_neutral_state.yaml
uv run python -m deception_circuits.paper_cli validate-data        --config $CFG
uv run python -m deception_circuits.paper_cli make-splits          --config $CFG
uv run python -m deception_circuits.paper_cli collect-activations  --config $CFG \
      --device cuda --confirm-model-load
uv run python -m deception_circuits.paper_cli train-probes         --config $CFG
uv run python -m deception_circuits.paper_cli analyze-confounds    --config $CFG
uv run python -m deception_circuits.paper_cli make-figures         --config $CFG
uv run python -m deception_circuits.paper_cli audit                --config $CFG
```

**The causal stage is deliberately disabled for this variant** and its config has
`intervention_strengths: []`. The neutral prompt asks the model nothing, so there
is no forced choice to measure and therefore no behavioural endpoint independent
of the probe. This variant answers one question — how much of the probe's
performance survives when the prompt does not name "bluff" — and that is a probe
comparison, not a causal one.

Read positions differ between variants and both were verified on the real data:

| variant | boundary marker | prompt ends with | read position |
|---|---|---|---|
| `judge_question` | `Answer:` | `…Reply with only 'Yes' or 'No'.\nAnswer:` | final token, ~165 tokens in |
| `neutral_state` | `Raise.` | `…The player decided to: Raise.` | final token, ~154 tokens in |

Both were checked to be byte-identical under a label flip, so neither prompt can
encode the answer.

## Step 5 — what to read first

1. **`audit`** must print `PASS`. It fails on missing provenance, a
   non-validation-selected layer, a missing causal control condition, an
   unfrozen SAE, and stale or orphaned figures. Notes are printed separately and
   are not failures.
2. **The probe number against `nuisance_only`.** This is the whole result. The
   bar is AUROC 0.9204 / PR-AUC 0.6166 from mechanically parsed game state with
   no model. A probe in the low 0.90s has not demonstrated anything beyond hand
   evaluation. Read `figures/nuisance_baselines.png` first.
3. **`confound_results.json` → `controlled_probe.auroc_drop_after_control`** and
   the `made_hand_matched` subset. These say how much survives the controls.
4. **`causal_results.json` → `effects`.** Positive and negative steering must
   clearly exceed `random_matched_norm`, `orthogonal`, and `shuffled_label`, with
   `wrong_layer` and `wrong_token_position` near zero, and format adherence in
   `quality_controls` must not collapse.

## Step 6 — claims

Before writing anything, read `docs/PAPER_CLAIM_GUARDRAILS.md` and
`docs/POKER_DATASET_NOTES.md`. On this dataset the model classifies an
already-taken raise, so the strongest available causal claim is that intervening
changes the model's bluff **judgement**, not its deceptive **behaviour**.

## If storage is tight

Set `activation_layers` to an explicit list, e.g.
`[0, 4, 8, 12, 16, 20, 24, 28, 31]`, which cuts 23.4 GB to ~6.6 GB per variant.
Layerwise probing is the primary experiment, so prefer all layers if you can
afford them; the subset still supports every other stage.
