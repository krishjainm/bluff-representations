OpenAI + HF pipeline outputs (not paper figures)
================================================

gpt4o-dataset (default --out)
  data/openai_runs/gpt4o_YYYYMMDD_HHMMSS/
    deception_data.csv
    activations/          (sample_<id>.pt — gitignored by *.pt)
    activation_mapping.json

llm-judge (default --out)
  data/openai_runs/llm_judge_outputs/<input_stem>_judged.csv

Paper figures (ROC, layer curves, etc.) are written under repo paper_figures/
when you run run-probes or refresh-paper-figures on an experiment output dir —
not here.
