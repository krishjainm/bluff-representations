#!/usr/bin/env python3
"""Regenerate the headline results table from committed run artifacts.

Reads results/poker_bluff_<variant>_<model>/{probe,confound,sae,causal}_results.json
and prints the tables in docs/RESULTS_SUMMARY.md. Every number in the paper should
be traceable to this script plus the artifacts, not to a transcription.

    python scripts/summarize_run.py
"""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
PAIRS = [(v, m) for v in ("judge_question", "neutral_state")
         for m in ("llama31_8b", "mistral7b_v03")]


def load(run: Path, name: str):
    path = run / f"{name}_results.json"
    return json.loads(path.read_text()) if path.is_file() else None


def selected_layer(probe: dict):
    runs = probe.get("runs") or []
    layers = [r.get("selected_layer") for r in runs if r.get("selected_layer") is not None]
    if not layers:
        return None
    return max(set(layers), key=layers.count)


def main() -> None:
    print("## Probe vs nuisance baseline (frozen test partition)\n")
    hdr = f"{'variant':<15}{'model':<15}{'layer':>6}{'AUROC':>9}{'PR-AUC':>9}{'nuis':>9}{'delta':>9}{'resid':>9}{'mh-match':>10}"
    print(hdr)
    print("-" * len(hdr))
    for variant, model in PAIRS:
        run = RESULTS / f"poker_bluff_{variant}_{model}"
        probe = load(run, "probe")
        if probe is None:
            print(f"{variant:<15}{model:<15}{'(no probe_results.json)':>52}")
            continue
        nuis = probe["baselines"]["nuisance_only"]["auroc"]
        auroc = probe["test_auroc_mean"]
        conf = load(run, "confound") or {}
        cp = conf.get("controlled_probe") or {}
        resid = (cp.get("residualized") or {}).get("auroc")
        mh = (((conf.get("subsets") or {}).get("made_hand_matched") or {}).get("probe") or {}).get("auroc")
        print(f"{variant:<15}{model:<15}{str(selected_layer(probe)):>6}{auroc:>9.4f}"
              f"{probe['test_pr_auc_mean']:>9.4f}{nuis:>9.4f}{auroc - nuis:>+9.4f}"
              f"{(f'{resid:.4f}' if resid is not None else 'n/a'):>9}"
              f"{(f'{mh:.4f}' if mh is not None else 'n/a'):>10}")

    print("\n## Instruction confound (judge_question minus neutral_state)\n")
    for model in ("llama31_8b", "mistral7b_v03"):
        a = load(RESULTS / f"poker_bluff_judge_question_{model}", "probe")
        b = load(RESULTS / f"poker_bluff_neutral_state_{model}", "probe")
        if a and b:
            print(f"  {model:<15} judge {a['test_auroc_mean']:.4f} vs neutral "
                  f"{b['test_auroc_mean']:.4f}  -> {a['test_auroc_mean'] - b['test_auroc_mean']:+.4f}")

    print("\n## Causal suite: effect on P(positive) at the largest dose\n")
    for variant, model in PAIRS:
        causal = load(RESULTS / f"poker_bluff_{variant}_{model}", "causal")
        if causal is None:
            continue
        guard = causal.get("baseline_choice_mass_check", {})
        print(f"\n{model} / {variant} - layer {causal['layer']}, n={causal['n_prompts']}, "
              f"endpoint={causal['primary_endpoint']}")
        print(f"  baseline forced-choice mass {guard.get('observed', float('nan')):.4f} "
              f"(floor {guard.get('floor')})")
        hdr2 = f"  {'condition':<42}{'rel_mag':>9}{'delta':>10}{'ci_low':>10}{'ci_high':>10}{'pos_rate':>10}"
        print(hdr2)
        for cond, val in causal["effects"].items():
            by = val.get("by_strength") or {}
            top = max(by, key=lambda s: float(s)) if by else None
            if top is None:
                continue
            r = by[top]
            print(f"  {cond:<42}{r.get('mean_relative_magnitude', float('nan')):>9.4f}"
                  f"{r['mean_difference']:>10.5f}{r['ci_lower']:>10.5f}{r['ci_upper']:>10.5f}"
                  f"{r.get('intervened_chose_positive_rate', float('nan')):>10.3f}")

    print("\n## SAE diagnostics (held-out, frozen before evaluation)\n")
    h3 = f"{'variant':<15}{'model':<15}{'var_expl':>10}{'L0':>8}{'dead':>8}{'seed_stab':>11}"
    print(h3)
    print("-" * len(h3))
    for variant, model in PAIRS:
        sae = load(RESULTS / f"poker_bluff_{variant}_{model}", "sae")
        if sae is None:
            continue
        d = (sae.get("diagnostics") or {}).get("test") or {}
        stab = sae.get("seed_stability") or {}
        stab_v = stab.get("mean_max_cosine") if isinstance(stab, dict) else None
        print(f"{variant:<15}{model:<15}{d.get('fraction_variance_explained', float('nan')):>10.4f}"
              f"{d.get('l0_mean', float('nan')):>8.2f}{d.get('dead_feature_fraction', float('nan')):>8.3f}"
              f"{(f'{stab_v:.4f}' if stab_v is not None else 'n/a'):>11}"
              f"   rankable {sae['sae_config']['n_features'] - sae.get('n_features_too_rare_to_rank', 0)}"
              f"/{sae['sae_config']['n_features']}")


if __name__ == "__main__":
    main()
