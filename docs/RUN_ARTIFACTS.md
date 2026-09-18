# Where the real-run artifacts live

Written 2026-09-17, after recovering the run from the rented A100.

## Committed to this branch

`results/poker_bluff_{judge_question,neutral_state}_{llama31_8b,mistral7b_v03}/`
— 104 files, ~24 MB, force-added past the `results/` gitignore entry:

- `probe_results.json`, `confound_results.json`, `sae_results.json`,
  `causal_results.json`, `baseline_results.json`
- `split_manifest.json`, `resolved_config.yaml`, `run_metadata.json`
- `intervention_records.csv` (per-row causal records, ~6 MB each)
- `sae_feature_ranking.csv`, `sae_feature_examples.json`
- `figures/*.png` + the source `*.csv` for each figure + `figures_manifest.json`

`data/activations/*/extraction_spec.json` — 4 files, the pinned extraction
fingerprints. Example (Llama, judge_question):

```
fingerprint  64b57f74cf212afb0c3fc06fda60a4927e01ff0707cc844142e56a59627c63f7
model        meta-llama/Llama-3.1-8B-Instruct @ 0e9e39f249a16976918f6564b8830bc894c89659
dtype        bfloat16     mode  prompt_end     site  block
template     "{statement}\nAnswer:"    boundary marker  "Answer:"
```

Regenerate every reported number with `python scripts/summarize_run.py`.

## NOT committed — held on the A100 only

| artifact | size | why not committed |
|---|---|---|
| `data/activations/*/` (the arrays) | **89 GB** | far past any git limit |
| `results/*/sae_model.pt` | 1.1 GB × 4 | `*.pt` is gitignored; regenerable from activations |
| `data/activations/*/extraction_manifest.json` | 52 MB × 4 | per-sample provenance; too large for git |
| `data/activations/*/extraction_manifest.jsonl` | 34 MB × 4 | append-journal form of the same |

The per-sample manifests are pulled to local disk (uncommitted) and anchored by
SHA-256 so they can be verified if moved:

```
925a2dfe3727f6e802d25b775b05ab14b18be7777bc7075039211080d2209367  judge_question_llama31_8b/extraction_manifest.json
f10fdd6eb5d75985b4a50d2fba83cc931672dec1426dcfc97739ed4b9cce91ce  judge_question_llama31_8b/extraction_manifest.jsonl
00eb8bb559560074490705643a80bc12e85ef5306ea30c9853418f013d2c5027  judge_question_mistral7b_v03/extraction_manifest.json
a26cc6cc5b9e6df8c1d851815cd57e2e055d05939246d4c8f26ad9ea5dd4ec5c  judge_question_mistral7b_v03/extraction_manifest.jsonl
71d52870ccc040d7bb07734e2c9a71bc5b4d0ff3fbbd52e0dc744373ad756667  neutral_state_llama31_8b/extraction_manifest.json
3a30cce2a4b74cd64ac2681de89b2329b20702a4cb4e2daff8fff382b292f962  neutral_state_llama31_8b/extraction_manifest.jsonl
91acd5e74d120d72de1ea8d0fdefece01cd711f61eabd786ed178558f8bb67e0  neutral_state_mistral7b_v03/extraction_manifest.json
887a6cb4f9ad78988119017953f27d0c022c43bbffd1d56f8651ab312d41f4f8  neutral_state_mistral7b_v03/extraction_manifest.jsonl
```

## The instance

`ubuntu@129.146.164.18`, key `~/.ssh/lambda_key2.pem`, repo at
`~/deception-llms`, A100-SXM4-40GB, Ubuntu 22.04. Up 2d 5h as of 2026-09-17,
idle (0 MiB GPU, load 0.00), **still billing**.

```bash
ssh -i ~/.ssh/lambda_key2.pem ubuntu@129.146.164.18
rsync -az --exclude='*.pt' -e "ssh -i ~/.ssh/lambda_key2.pem" \
  ubuntu@129.146.164.18:deception-llms/results/ ./results/
```

**Before terminating it**, decide about the 89 GB of activations. They cost a full
extraction pass to rebuild (bounded by GPU time, not by any irreplaceable input —
the dataset and pinned revisions are committed, so re-extraction is deterministic
given the same `extraction_spec.json` fingerprints). Any re-run of the causal
sweep on a corrected dose grid needs them, so terminating now converts a cheap
re-run into an expensive one.
