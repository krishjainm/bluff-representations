#!/usr/bin/env python3
"""
Paper experiment driver: probes, splits, optional activation collection, causal sweeps.

- Training / metrics: no OpenAI key (needs CSV + optional activation tensors).
- Activation collection: Hugging Face subject model only (local GPU/CPU).
- GPT-4o dataset *generation*: requires OPENAI_API_KEY (see run_gpt4o_dataset).

Set the key via environment variable, or create a repo-root ``.env`` file
(see ``.env.example``; ``.env`` is gitignored). Do not paste API keys into chat.

**Where API-related files go**

- ``data/openai_runs/`` — GPT-4o dataset runs (``deception_data.csv``, ``activations/``,
  ``activation_mapping.json``) and LLM-judge outputs (under ``llm_judge_outputs/``).
- ``paper_figures/`` — **only** regenerated figures from probe/experiment JSON + npz;
  OpenAI does not write here directly. Run ``run-probes`` or ``refresh-paper-figures``
  after you have new experiment outputs.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent
OPENAI_RUNS_ROOT = REPO_ROOT / "data" / "openai_runs"


def default_gpt4o_output_dir() -> str:
    OPENAI_RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    return str(OPENAI_RUNS_ROOT / f"gpt4o_{datetime.now():%Y%m%d_%H%M%S}")


def default_llm_judge_output_path(csv_path: str) -> str:
    out_dir = OPENAI_RUNS_ROOT / "llm_judge_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(csv_path).stem
    return str(out_dir / f"{stem}_judged.csv")

import pandas as pd
import torch

from deception_circuits import DeceptionTrainingPipeline
from deception_circuits.model_integration import ActivationExtractor, ModelIntegrationPipeline
from deception_circuits.paper_full_pipeline import run_paper_full_suite
from deception_circuits.paper_figure_suite import build_paper_figures_from_experiment_dir
from deception_circuits.paper_results_plots import plot_probe_curves_from_json
from deception_circuits.dataset_balance import balance_base_items_stratified
from deception_circuits.data_loader import DeceptionDataLoader
from deception_circuits.sae_steering import (
    load_checkpoint_steering_vector,
    suggest_deception_feature_indices,
)
from deception_circuits.llm_judge import judge_csv_column


def collect_activations_from_csv(
    csv_path: str,
    model_name: str,
    activation_out_dir: str,
    device: str = "cuda",
    pooling: str = "last",
    batch_size: int = 8,
    text_template: str = "Q: {statement}\nA: {response}",
    activation_site: str = "block",
) -> None:
    """
    Run the subject LM on each row and save ``sample_<sample_id>.pt`` tensors
    for use with ``DeceptionDataLoader(..., activation_dir=...)``.
    """
    df = pd.read_csv(csv_path)
    if "sample_id" not in df.columns:
        df["sample_id"] = range(len(df))
    texts: List[str] = [
        text_template.format(statement=str(r["statement"]), response=str(r["response"]))
        for _, r in df.iterrows()
    ]
    ext = ActivationExtractor(
        model_name, device=device, pooling=pooling, activation_site=activation_site
    )
    out = Path(activation_out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        acts = ext.extract_activations(batch)
        for j in range(acts.shape[0]):
            row_idx = start + j
            sid = int(df.iloc[row_idx]["sample_id"])
            torch.save(acts[j].cpu(), out / f"sample_{sid}.pt")
    print(f"Saved {len(texts)} activation files to {out}")


def run_probe_experiment(
    csv_path: str,
    activation_dir: Optional[str],
    output_dir: str,
    device: str = "cpu",
    val_size: float = 0.1,
    test_size: float = 0.2,
) -> dict:
    """Full pipeline with paper default splits and sklearn + torch probes."""
    pipe = DeceptionTrainingPipeline(device=device, output_dir=output_dir)
    return pipe.run_full_experiment(
        csv_path=csv_path,
        activation_dir=activation_dir,
        split_mode="base_item",
        test_size=test_size,
        val_size=val_size,
        probe_config={
            "epochs": 100,
            "validation_split": 0.2,
            "early_stopping_patience": 10,
        },
        autoencoder_config={
            "epochs": 50,
            "lr": 0.001,
            "l1_coeff": 0.01,
            "bottleneck_dim": 0,
            "tied_weights": True,
            "activation_type": "ReLU",
            "topk_percent": 10,
            "lambda_classify": 1.0,
            "validation_split": 0.2,
            "early_stopping_patience": 5,
        },
    )


def run_gpt4o_dataset(
    api_key: str,
    out_dir: str,
    num_statements: int,
    hf_subject_model: str,
    device: str = "cuda",
) -> dict:
    """
    Generate paired CSV + activations via OpenAI + Hugging Face subject model.
    **Requires OpenAI API key.**
    """
    pipe = ModelIntegrationPipeline(
        gpt4o_api_key=api_key,
        transformer_model_name=hf_subject_model,
        device=device,
    )
    return pipe.create_complete_dataset(
        Path(out_dir),
        num_statements=num_statements,
        scenarios=["poker", "sandbagging", "roleplay", "password_gating"],
    )


def balance_csv_inplace(
    csv_path: str,
    output_path: str,
    stratify_columns: Optional[List[str]] = None,
) -> None:
    df = pd.read_csv(csv_path)
    df = balance_base_items_stratified(
        df, stratify_columns=stratify_columns or ["scenario", "difficulty_bucket"]
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Balanced {len(df)} rows -> {output_path}")


def run_causal_steering_demo(
    hf_model: str,
    prompt: str,
    layer_idx: int,
    device: str = "cuda",
    strengths: Optional[List[float]] = None,
) -> None:
    """Quick dose–response printout (subject LM only, no API)."""
    from deception_circuits.causal_generation import CausalLMInterventionRunner

    strengths = strengths or [0.0, 0.5, 1.0, 2.0]
    runner = CausalLMInterventionRunner(hf_model, device=device)
    n_layers = len(runner._layers())
    if layer_idx < 0:
        layer_idx = n_layers - 1
    layer_idx = min(layer_idx, n_layers - 1)
    dim = runner.model.config.hidden_size
    v = runner.random_direction_control(dim, seed=42)
    print("Causal steering demo (random control direction, matched norm)")
    for a in strengths:
        r = runner.generate(
            prompt,
            max_new_tokens=48,
            layer_idx=layer_idx,
            steering_vector=v,
            strength=a,
            steering_mode="add",
        )
        print(f"  strength={a:.2f} -> {r.text[:200]!r}...")


def run_sae_steering_generation(
    hf_model: str,
    sae_checkpoint: str,
    feature_index: int,
    prompt: str,
    layer_idx: int,
    strength: float = 1.0,
    device: str = "cuda",
    intervention_site: str = "block",
) -> None:
    """Generate with SAE decoder column as steering vector at ``layer_idx``."""
    from deception_circuits.causal_generation import CausalLMInterventionRunner

    vec = load_checkpoint_steering_vector(
        sae_checkpoint, feature_index, device=device, normalize=True
    )
    runner = CausalLMInterventionRunner(
        hf_model, device=device, intervention_site=intervention_site
    )
    n_layers = len(runner._layers())
    if layer_idx < 0:
        layer_idx = n_layers - 1
    layer_idx = min(layer_idx, n_layers - 1)
    base = runner.generate(prompt, max_new_tokens=64, strength=0.0)
    steered = runner.generate(
        prompt,
        max_new_tokens=64,
        layer_idx=layer_idx,
        steering_vector=vec,
        strength=strength,
        steering_mode="add",
    )
    print("--- baseline ---\n", base.text)
    print("--- sae steered ---\n", steered.text)


def _load_env_file() -> None:
    """Load repo-root ``.env`` into ``os.environ`` (e.g. OPENAI_API_KEY)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(env_path, override=False)


def main() -> None:
    _load_env_file()
    # Avoid defaulting to cuda when PyTorch is CPU-only (common on Windows).
    _torch_device_default = "cuda" if torch.cuda.is_available() else "cpu"
    p = argparse.ArgumentParser(description="Deception circuits paper experiments")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_act = sub.add_parser("collect-activations", help="HF model -> activation .pt files")
    p_act.add_argument("--csv", required=True)
    p_act.add_argument("--model", required=True, help="HuggingFace model id")
    p_act.add_argument("--out", required=True, help="Activation output directory")
    p_act.add_argument("--device", default=_torch_device_default)
    p_act.add_argument("--pooling", default="last", choices=["last", "first", "mean"])
    p_act.add_argument("--batch-size", type=int, default=8)
    p_act.add_argument(
        "--activation-site",
        default="block",
        choices=["block", "attn", "mlp"],
        help="HF submodule to hook for activations",
    )

    p_bal = sub.add_parser("balance-csv", help="Balance base items per stratum -> new CSV")
    p_bal.add_argument("--csv", required=True)
    p_bal.add_argument("--out", required=True)
    p_bal.add_argument(
        "--stratify",
        nargs="*",
        default=None,
        help="Columns (default: scenario difficulty_bucket)",
    )

    p_full = sub.add_parser(
        "full-suite",
        help="Balance + probes + optional cross-context steering + causal controls",
    )
    p_full.add_argument("--csv", required=True)
    p_full.add_argument("--activations", default=None)
    p_full.add_argument("--output", default="paper_suite_out")
    p_full.add_argument("--device", default="cpu")
    p_full.add_argument("--hf-model", default=None, help="Subject LM for causal stages")
    p_full.add_argument("--no-balance", action="store_true")
    p_full.add_argument("--no-probes", action="store_true")
    p_full.add_argument("--cross-steering", action="store_true")
    p_full.add_argument("--causal-controls", action="store_true")
    p_full.add_argument("--test-size", type=float, default=0.2)
    p_full.add_argument("--val-size", type=float, default=0.1)
    p_full.add_argument(
        "--intervention-site",
        default="block",
        choices=["block", "attn", "mlp"],
    )

    p_run = sub.add_parser("run-probes", help="Train probes + SAEs on CSV + activations")
    p_run.add_argument("--csv", required=True)
    p_run.add_argument("--activations", default=None)
    p_run.add_argument("--output", default="paper_results")
    p_run.add_argument("--device", default="cpu")
    p_run.add_argument("--test-size", type=float, default=0.2)
    p_run.add_argument("--val-size", type=float, default=0.1)
    p_run.add_argument(
        "--plot-dir",
        default=None,
        help="If set, write probe AUROC/accuracy PNGs from experiment_results.json",
    )

    p_gpt = sub.add_parser("gpt4o-dataset", help="Generate dataset (needs API key)")
    p_gpt.add_argument(
        "--out",
        default=None,
        help=f"Output directory (default: data/openai_runs/gpt4o_<timestamp>)",
    )
    p_gpt.add_argument("--num-statements", type=int, default=50)
    p_gpt.add_argument(
        "--hf-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF subject model for activation extraction (override if needed)",
    )
    p_gpt.add_argument("--device", default=_torch_device_default)
    p_gpt.add_argument(
        "--api-key",
        default=None,
        help="Or set OPENAI_API_KEY",
    )

    p_c = sub.add_parser("causal-demo", help="Steering sweep demo on one prompt")
    p_c.add_argument("--model", required=True)
    p_c.add_argument("--prompt", required=True)
    p_c.add_argument("--layer", type=int, default=-1)
    p_c.add_argument("--device", default=_torch_device_default)

    p_plot = sub.add_parser(
        "plot-probes", help="PNG layer curves from experiment_results.json"
    )
    p_plot.add_argument("--json", dest="json_path", required=True)
    p_plot.add_argument("--out", required=True, help="Output directory for PNGs")

    p_sae_e = sub.add_parser(
        "sae-extract-steering", help="Save SAE decoder column as steering vector .pt"
    )
    p_sae_e.add_argument("--checkpoint", required=True)
    p_sae_e.add_argument("--feature", type=int, required=True)
    p_sae_e.add_argument("--out", required=True)
    p_sae_e.add_argument("--device", default="cpu")

    p_sae_r = sub.add_parser(
        "sae-rank-features",
        help="Rank SAE latents by |corr| with labels (needs activations)",
    )
    p_sae_r.add_argument("--checkpoint", required=True)
    p_sae_r.add_argument("--csv", required=True)
    p_sae_r.add_argument("--activations", required=True)
    p_sae_r.add_argument("--layer", type=int, required=True)
    p_sae_r.add_argument("--device", default="cpu")
    p_sae_r.add_argument("--top-k", type=int, default=15)

    p_sae_g = sub.add_parser(
        "sae-steer-generate",
        help="One prompt: baseline vs SAE-feature steering on subject LM",
    )
    p_sae_g.add_argument("--hf-model", required=True)
    p_sae_g.add_argument("--checkpoint", required=True)
    p_sae_g.add_argument("--feature", type=int, required=True)
    p_sae_g.add_argument("--prompt", required=True)
    p_sae_g.add_argument("--layer", type=int, default=-1)
    p_sae_g.add_argument("--strength", type=float, default=1.0)
    p_sae_g.add_argument("--device", default=_torch_device_default)
    p_sae_g.add_argument(
        "--intervention-site",
        default="block",
        choices=["block", "attn", "mlp"],
    )

    p_judge = sub.add_parser(
        "llm-judge", help="OpenAI judge: add deception_score column to CSV"
    )
    p_judge.add_argument("--csv", required=True)
    p_judge.add_argument(
        "--out",
        default=None,
        help="Judged CSV path (default: data/openai_runs/llm_judge_outputs/<stem>_judged.csv)",
    )
    p_judge.add_argument("--context-col", required=True)
    p_judge.add_argument("--response-col", required=True)
    p_judge.add_argument("--api-key", default=None)
    p_judge.add_argument(
        "--model",
        default=None,
        help="Judge model id (default: gpt-4o-mini, or openai/gpt-4o-mini if using OpenRouter)",
    )
    p_judge.add_argument("--max-rows", type=int, default=None)
    p_judge.add_argument("--sleep", type=float, default=0.2)

    p_refresh = sub.add_parser(
        "refresh-paper-figures",
        help="Rebuild paper_figures/ from an experiment output directory",
    )
    p_refresh.add_argument(
        "--experiment-dir",
        required=True,
        help="Directory containing experiment_results.json (e.g. paper_results or paper_suite_out/probes)",
    )
    p_refresh.add_argument(
        "--out",
        default=None,
        help="Override paper figures directory (default: repo paper_figures/)",
    )
    p_refresh.add_argument(
        "--auxiliary-dir",
        default=None,
        help="Directory with causal_control_suite.json / cross_context_steering.json (default: infer from experiment dir)",
    )

    args = p.parse_args()
    if args.cmd == "collect-activations":
        collect_activations_from_csv(
            args.csv,
            args.model,
            args.out,
            device=args.device,
            pooling=args.pooling,
            batch_size=args.batch_size,
            activation_site=args.activation_site,
        )
    elif args.cmd == "balance-csv":
        balance_csv_inplace(
            args.csv,
            args.out,
            stratify_columns=args.stratify,
        )
    elif args.cmd == "full-suite":
        if (args.cross_steering or args.causal_controls) and not args.hf_model:
            raise SystemExit("--hf-model required when using --cross-steering or --causal-controls")
        res = run_paper_full_suite(
            csv_path=args.csv,
            output_dir=args.output,
            activation_dir=args.activations,
            device=args.device,
            hf_subject_model=args.hf_model,
            balance_dataset=not args.no_balance,
            run_probes=not args.no_probes,
            run_cross_context_steering=args.cross_steering,
            run_causal_controls=args.causal_controls,
            test_size=args.test_size,
            val_size=args.val_size,
            intervention_site=args.intervention_site,
        )
        print(json.dumps(res, indent=2, default=str))
    elif args.cmd == "run-probes":
        run_probe_experiment(
            args.csv,
            args.activations,
            args.output,
            device=args.device,
            val_size=args.val_size,
            test_size=args.test_size,
        )
        build_paper_figures_from_experiment_dir(args.output)
        print("paper_figures/ updated (see visualization_tracking.json)")
        if args.plot_dir:
            jp = Path(args.output) / "experiment_results.json"
            if jp.exists():
                plot_probe_curves_from_json(jp, args.plot_dir)
                print("Additional probe plots ->", args.plot_dir)
    elif args.cmd == "gpt4o-dataset":
        key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Set OPENAI_API_KEY or pass --api-key")
        out_dir = args.out or default_gpt4o_output_dir()
        stats = run_gpt4o_dataset(
            key, out_dir, args.num_statements, args.hf_model, args.device
        )
        print(json.dumps(stats, indent=2, default=str))
        print(f"\nDataset directory -> {out_dir}")
    elif args.cmd == "causal-demo":
        run_causal_steering_demo(
            args.model,
            args.prompt,
            args.layer,
            device=args.device,
        )
    elif args.cmd == "plot-probes":
        paths = plot_probe_curves_from_json(args.json_path, args.out)
        print("Wrote:", paths)
    elif args.cmd == "sae-extract-steering":
        v = load_checkpoint_steering_vector(
            args.checkpoint, args.feature, device=args.device
        )
        torch.save(v.cpu(), args.out)
        print(f"Saved steering vector shape {tuple(v.shape)} -> {args.out}")
    elif args.cmd == "sae-rank-features":
        dl = DeceptionDataLoader(device=args.device)
        dl.load_csv(args.csv, activation_dir=args.activations)
        X = dl.get_activations_tensor(dl.data)
        if args.layer < 0 or args.layer >= X.shape[1]:
            raise SystemExit(f"--layer must be in [0, {X.shape[1] - 1}]")
        X_layer = X[:, args.layer, :]
        y = dl.get_labels_tensor(dl.data)
        ranked = suggest_deception_feature_indices(
            args.checkpoint,
            X_layer,
            y,
            device=args.device,
            top_k=args.top_k,
        )
        print(json.dumps(ranked, indent=2))
    elif args.cmd == "sae-steer-generate":
        run_sae_steering_generation(
            args.hf_model,
            args.checkpoint,
            args.feature,
            args.prompt,
            args.layer,
            strength=args.strength,
            device=args.device,
            intervention_site=args.intervention_site,
        )
    elif args.cmd == "llm-judge":
        key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Set OPENAI_API_KEY or pass --api-key")
        out_csv = args.out or default_llm_judge_output_path(args.csv)
        judge_csv_column(
            args.csv,
            out_csv,
            context_column=args.context_col,
            response_column=args.response_col,
            api_key=key,
            judge_model=args.model,  # None -> openai_compat.default_judge_model()
            rate_limit_s=args.sleep,
            max_rows=args.max_rows,
        )
        print("Done ->", out_csv)
    elif args.cmd == "refresh-paper-figures":
        odir = Path(args.out) if args.out else None
        aux = Path(args.auxiliary_dir) if args.auxiliary_dir else None
        info = build_paper_figures_from_experiment_dir(
            args.experiment_dir,
            paper_figures_dir=odir,
            auxiliary_dir=aux,
        )
        print(json.dumps(info, indent=2, default=str))


if __name__ == "__main__":
    main()
