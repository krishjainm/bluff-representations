
# Llama 3.1 Replay (Hidden States)

## Install (local or Colab)
```bash
pip install "torch>=2.1" "transformers>=4.43" accelerate pandas huggingface_hub
# If gated, set your HF token:
# export HUGGING_FACE_HUB_TOKEN=hf_...
```

## Run
```bash
python llama_replay_hidden_states.py \
  --csv sample_data/normalized_poker_gpt4o.fixed.csv \
  --model meta-llama/Llama-3.1-8B \
  --device auto \
  --batch-size 2 \
  --max-tokens 512 \
  --pooling mean \
  --layers all \
  --out-dir activations/llama31-8b/poker
```

Outputs:
- One `.pt` per row under `activations/llama31-8b/poker/000000.pt`, etc.
- An `index.csv` with row → path, label, scenario.
- Each `.pt` contains a `features` tensor of shape `(num_selected_layers, hidden_size)`.

## Using with your pipeline
- If your probe expects a specific input dim (e.g., 768), either:
  1) **Adjust the probe input dim** to the model hidden size (e.g., 4096 for Llama 3.1 8B), or
  2) **Add a projection** (PCA or a learned linear layer) from 4096 → 768 before the probe.
```
