# Compute plan

Do not begin expensive jobs from this repository automatically. First run the
fixture tests and a small real-data activation extraction pilot. Cache activations
by model revision, prompt template, hook site, layer, and token policy; make each
sample atomic and resumable. Probe fitting is CPU-cheap once activations exist.

Budget larger GPU work in stages: one primary model and leakage-safe probes;
then matched controls; then the paired generation/intervention suite; then only
if prior diagnostics are credible, SAE and small model replication. Paid LLM
judging is optional and must save its exact prompt, model, temperature, raw output,
and parse result. No cost estimate should be presented as measured until an actual
hardware/batch configuration is selected.
