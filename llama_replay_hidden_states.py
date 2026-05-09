#!/usr/bin/env python3
"""
llama_replay_hidden_states.py
(See README for usage)
"""
import os, argparse, math, pandas as pd, torch
from pathlib import Path
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

def parse_layers(s: str, num_hidden_layers: int):
    if s == "all":
        return list(range(num_hidden_layers))
    out = []
    for part in s.split(","):
        i = int(part.strip())
        if i < 0:
            i = num_hidden_layers + i
        if not (0 <= i < num_hidden_layers):
            raise ValueError(f"Layer index {i} out of range 0..{num_hidden_layers-1}")
        out.append(i)
    return sorted(set(out))

def pool_tokens(hidden: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if hidden.dim() == 2:
        hidden = hidden.unsqueeze(0)
    B, T, H = hidden.shape
    if attention_mask is None:
        attention_mask = torch.ones(B, T, device=hidden.device, dtype=torch.long)
    if mode == "mean":
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1.0)
        return summed / counts
    elif mode == "last":
        lengths = attention_mask.sum(dim=1)
        idx = (lengths - 1).clamp(min=0)
        out = torch.stack([hidden[b, idx[b].item()] for b in range(B)], dim=0)
        return out
    elif mode == "bos":
        return hidden[:, 0, :]
    else:
        raise ValueError(f"Unknown pooling: {mode}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--text-col", default="statement")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B")
    ap.add_argument("--device", default="auto", choices=["cpu","cuda","mps","auto"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--pooling", default="mean", choices=["mean","last","bos"])
    ap.add_argument("--layers", default="all")
    ap.add_argument("--out-dir", default="activations/llama31-8b/poker")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.csv"

    df = pd.read_csv(args.csv)
    if args.text_col not in df.columns:
        raise SystemExit(f"[ERROR] text column '{args.text_col}' not in CSV. Available: {list(df.columns)}")
    texts = df[args.text_col].astype(str).tolist()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"[INFO] Loading model {args.model} on device={device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Ensure padding works for models without a native PAD token (e.g., Llama)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if device in ("cuda","mps") else torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto" if device == "cuda" else None
    )
    model.eval()

    num_hidden_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    chosen_layers = parse_layers(args.layers, num_hidden_layers)
    print(f"[INFO] Model hidden_size={hidden_size}, layers={num_hidden_layers}. Extracting layers={chosen_layers}")

    rows = []
    total = len(texts)
    bs = args.batch_size
    for start in range(0, total, bs):
        end = min(start + bs, total)
        batch_texts = texts[start:end]

        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_tokens
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states

        selected = [hs[i+1].detach().to("cpu") for i in chosen_layers]
        pooled_layers = [pool_tokens(layer, enc.get("attention_mask").to("cpu"), args.pooling) for layer in selected]

        L = len(pooled_layers)
        for i in range(end - start):
            per_sample = torch.stack([pl[i] for pl in pooled_layers], dim=0)
            sample_id = start + i
            save_path = out_dir / f"{sample_id:06d}.pt"
            torch.save({"features": per_sample, "layers": chosen_layers, "hidden_size": hidden_size,
                        "pooling": args.pooling, "model": args.model, "row": int(sample_id)}, save_path)
            rows.append({
                "row": sample_id,
                "path": str(save_path),
                "label": int(df.loc[sample_id, "label"]) if "label" in df.columns else None,
                "scenario": df.loc[sample_id, "scenario"] if "scenario" in df.columns else None
            })

        if ((start//bs) % 10) == 0:
            print(f"[INFO] Processed {end}/{total} rows")

    pd.DataFrame(rows).to_csv(index_path, index=False)
    print(f"[DONE] Wrote index: {index_path} ({len(rows)} items)")
    if rows:
        print(f"[HINT] Example load: torch.load('{rows[0]['path']}')['features'].shape -> (num_layers, hidden_size)")

if __name__ == "__main__":
    main()
