# SAE audit

Audit of `deception_circuits/sparse_autoencoder.py` and its callers against
brief Phase 10. Every finding below is a defect in the legacy code, not a
judgement about the previous paper's intent.

## The finding that matters most

`training_pipeline.py` trains both an unsupervised and a **supervised**
autoencoder per layer, then:

```python
best_layer_name = sorted(
    supervised_results.items(),
    key=lambda x: x[1]['final_metrics']['accuracy'],
)[0]
best_autoencoder = supervised_results[best_layer_name]['autoencoder']
torch.save(best_autoencoder.state_dict(), models_dir / f"best_autoencoder_{best_layer_name}.pt")
```

So the artifact named `best_autoencoder_*.pt` is **always the supervised
autoencoder**, selected by its **classification accuracy**.
`SupervisedDeceptionAutoencoder` has a `classify` head and a cross-entropy term,
so its features are trained to be label-predictive by construction.

That is supervised feature *construction*, not unsupervised feature
*discovery*. Any interpretability claim resting on that artifact -- "the SAE
learned an interpretable deception feature" -- does not hold, because the
objective put the label there. This is precisely the brief's "whether supervised
autoencoder results are being mistaken for unsupervised feature discovery".

It also means the saved-model selection used labels, so it is not a clean
held-out artifact either.

`sae_steering.py` loads whichever checkpoint it is given and branches on
`isinstance(model, SupervisedDeceptionAutoencoder)`, so it will happily steer
along a supervised feature. The `.meta.json` does record `"supervised": true`,
which is the one saving grace: existing artifacts can be identified.

## Implementation defects

| # | Defect | Location | Effect |
|---|---|---|---|
| S1 | **Weight tying is fake and corrupts training.** `decode()` does `with torch.no_grad(): self.decoder.weight.copy_(self.encoder.weight.t())` on every forward. The copy is outside the autograd graph, so no gradient flows from the decoder back to the encoder weight through the tie; meanwhile the optimizer updates `decoder.weight` independently and the next forward overwrites it. The decoder's learned update is discarded every step. | `sparse_autoencoder.py:171-176` | Not weight tying. Silent training corruption, and `tied_weights=True` is the **default**. |
| S2 | **Mutates parameters inside `forward()`.** The same in-place copy runs at inference. A "frozen" autoencoder is not frozen: calling it changes its weights. | same | Any frozen-SAE held-out evaluation is invalid. |
| S3 | **Default is not overcomplete.** `bottleneck_dim=0` means "same as input", and every caller in the repo passes `0` or omits it. An SAE with as many features as dimensions is not a sparse dictionary in the usual sense. | `sparse_autoencoder.py:90`, callers in `training_pipeline.py:196`, `paper_experiments.py:123`, `model_integration.py:776`, `example_usage.py`, `production_example.py` | The claimed 8192-feature dictionary does not correspond to anything in the code; no `8192` appears anywhere in the repository. |
| S4 | **No decoder unit-norm.** Nothing constrains decoder column norms, so the model can shrink code activations and grow decoder norms to reduce the L1 penalty without becoming sparser. | `sparse_autoencoder.py` | The L1 sparsity penalty is gameable; feature magnitudes are not comparable across features. |
| S5 | **Top-k selects on `abs(x)` and returns `x * mask`,** with no ReLU. Negative pre-activations survive into the "sparse code". | `sparse_autoencoder.py:110-132` | The code is not non-negative, so "feature activation" has no sign convention and top-activating-example analysis is ambiguous. |
| S6 | **`batch_topk_activation` is per-example, not batch-wise** (`topk(..., dim=1)`), despite the name and the `BatchTopK` activation label. `batch_indices` is computed and never used. | same | Misleading name in a methods section; dead code. |
| S7 | **L1 penalty is `mean` over batch *and* features.** Standard practice is sum over features, mean over batch. | `sparse_autoencoder.py:197-199` | Effective sparsity strength scales as `1/n_features`, so changing width silently changes the penalty. Sparsity settings are not comparable across widths. |
| S8 | **No input normalization.** Raw residual-stream activations are fed in directly. | `sparse_autoencoder.py` | Reconstruction MSE is not comparable across layers or models, and the L1 scale is arbitrary. |
| S9 | **Missing required diagnostics.** No fraction of variance explained, no L0, no dead-feature fraction, no per-feature activation frequency. `get_feature_importance` reports only a zero-fraction. | `sparse_autoencoder.py:206-227` | None of the brief's required SAE reporting could be produced from this code. |

## Decision

The legacy module is left in place and marked experimental. It is **not**
migrated: too many of its defects are structural (S1/S2 in particular would
require rewriting the module's core), and the strict path needs an SAE whose
frozen-evaluation semantics are trustworthy.

`deception_circuits/paper_sae.py` is the one defensible unsupervised path:

- explicit `n_features` with no "same as input" fallback, and the expansion
  factor recorded;
- TopK activation by default (exact L0, no L1 shrinkage to game) with ReLU+L1
  available and the penalty summed over features;
- **genuine** weight tying when requested -- the decoder uses the encoder
  parameter transposed inside the graph, so one parameter is updated and
  gradients flow correctly -- and untied by default;
- decoder columns renormalised to unit norm after every optimizer step;
- input normalization fit on the **training partition only** and stored with the
  model;
- `freeze()` before any held-out evaluation, with a test asserting that
  evaluating a frozen SAE does not change a single weight;
- feature ranking on train/validation only, selected features evaluated on test;
- full diagnostics: reconstruction MSE, fraction of variance explained, L0, dead
  feature fraction, per-feature activation frequency, seed stability.

## What the new path still will not claim

A feature that correlates with the bluff label is **not** thereby a "deception
feature". The export tooling emits top and contrast examples with metadata for
human inspection and deliberately does not assign semantic names. See
`docs/PAPER_CLAIM_GUARDRAILS.md`.
