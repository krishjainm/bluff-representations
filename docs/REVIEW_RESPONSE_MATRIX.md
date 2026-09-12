# Review response matrix

| Concern | Status | Evidence / remaining work |
|---|---|---|
| Reproducible dataset schema and fixed splits | fixed in V2 probe path | `paper.py`, protocol, manifest checksum |
| Silent fake activations | fixed in V2 probe path | strict activation loader hard-fails |
| Answer-token leakage / unclear location | partially fixed | config records `prompt_end`; extractor implementation remains required |
| Test-set layer selection | fixed in V2 probe path | selection is validation-only and audit checks it |
| Repeated runs and metrics | partially fixed | five configurable seeds and full classification metrics; grouped CIs remain required |
| Poker confounds and nuisance baselines | blocked on metadata / implementation | schema can carry fields, matching analysis remains required |
| Cross-context transfer | not yet implemented | must select on source validation only |
| Behavioral causal endpoint and controls | not yet implemented | legacy controls are experimental and not paper-valid |
| Steering versus activation replacement | paper-only / legacy audit required | terminology guardrail needed before paper drafting |
| SAE diagnostics and feature examples | not yet implemented | do not make SAE claims from legacy code |
| Figures from real artifacts | not yet implemented | no publication figures should be generated yet |
| Concrete examples, labeling prompts, composition | blocked on real dataset | store with the data release |
