# Reproducibility

Install the pinned environment with `uv sync`. Populate a canonical CSV and its
real activation directory, then copy `configs/paper_v2.yaml` and set paths,
model identity/revision, extraction location, and seed.

```bash
deception-paper validate-data --config my_run.yaml
deception-paper make-splits --config my_run.yaml
# Loads subject-model weights and runs real forward passes; requires the flag.
deception-paper collect-activations --config my_run.yaml --confirm-model-load
deception-paper run-baselines --config my_run.yaml     # no model needed; run this first
deception-paper train-probes --config my_run.yaml
deception-paper analyze-confounds --config my_run.yaml
deception-paper train-sae --config my_run.yaml
deception-paper run-interventions --config my_run.yaml --confirm-model-load
deception-paper make-figures --config my_run.yaml
deception-paper audit --config my_run.yaml
```

The output directory contains `resolved_config.yaml`, `run_metadata.json`,
`split_manifest.json`, and `probe_results.json`. Preserve these four artifacts
with figures and any raw activation metadata. The commands intentionally do not
download a subject model or invoke an API; activation extraction is an explicit,
separate job until it has equivalent leakage tests and provenance logging.
