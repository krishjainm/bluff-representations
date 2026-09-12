# Experiment protocol

The primary V2 probe experiment uses a model state from before the answer is
generated (`activation_mode: prompt_end`). Each row has a unique `sample_id`, a
`base_item_id`, and a `split_group_id` when a game/trajectory creates related
items. Required CSV columns are `sample_id`, `base_item_id`, `statement`,
`response`, `label`, and `scenario`; labels are binary and both classes are
required.

Activations are one `sample_<sample_id>.pt` finite tensor per row, shaped
`[layers, hidden_size]`. They must be extracted from the exact model/revision,
site, and token policy recorded in the resolved configuration. Missing files,
NaNs, shape mismatch, duplicate IDs, and group leakage are fatal errors.

`make-splits` freezes a deterministic group-safe train/validation/test manifest.
Each seed fits all layer probes on train data, chooses one layer by validation
AUROC only, then reports that frozen model's untouched-test AUROC, PR-AUC,
accuracy, balanced accuracy, precision, recall, F1, MCC, and confusion counts.
Test performance must never choose a layer, direction, or regularization value.

Interventions and SAE work are not yet part of the validated V2 CLI. Do not call
probe-score changes behavioral causality. A future causal run must use paired
prompts, a task-native behavioral endpoint, quality checks, and random,
orthogonal, shuffled-label, wrong-layer/token, and true replacement controls.
