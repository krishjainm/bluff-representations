"""
Paper-ready figures for deception-circuit experiments.

Reads ``experiment_results.json`` (and optional probe checkpoints under
``models/``) and writes PNG + PDF into `paper_figures/` at the **repository
root** by default. Called automatically after ``DeceptionTrainingPipeline.run_full_experiment``
when results are saved.

Figures produced (aligned with the paper):
  - ``layer_performance`` — AUROC, accuracy, ECE vs layer
  - ``cross_scenario_generalization`` — heatmap (train scenario × test scenario AUROC)
  - ``feature_importance`` — top-|weights| from best probe checkpoint (if present)
  - ``autoencoder_reconstruction`` — unsupervised reconstruction loss vs layer (if present)
  - ``sklearn_logistic_auroc`` — sklearn vs held-out metrics (if present)
  - ``probe_training_history`` — val loss / AUROC for best layer (if history in JSON)
  - ``roc_curves`` — ROC for best-layer probe (needs ``test_predictions_best_probe.npz``)
  - ``activation_distributions`` — L2 norm of activations at best layer by label (same npz)
  - ``statistical_comparisons`` — bootstrap AUROC / accuracy CIs (``bootstrap_confidence.json``)
  - ``causal_control_response_lengths`` — bar chart from ``causal_control_suite.json`` (optional ``auxiliary_dir``)
  - ``cross_context_response_quality`` — baseline vs steered token counts from ``cross_context_steering.json``

Not generated without extra runs: model-size sweeps (multi-checkpoint comparison).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PAPER_FIG_DIR = REPO_ROOT / "paper_figures"


def _paper_figures_dir() -> Path:
    import os

    return Path(os.environ.get("PAPER_FIGURES_DIR", str(DEFAULT_PAPER_FIG_DIR)))


def _use_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_fig(fig, path_base: Path, dpi: int = 200) -> List[Path]:
    out = []
    for ext in (".png", ".pdf"):
        p = path_base.with_suffix(ext)
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        out.append(p)
    return out


def _fig_layer_performance(data: Dict[str, Any], plt) -> Optional[Any]:
    from .paper_results_plots import extract_probe_curves

    layers, aucs, accs, eces = extract_probe_curves(data)
    if not layers:
        return None
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    axes[0].plot(layers, aucs, "b-o", markersize=4)
    axes[0].set_ylabel("AUROC")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Linear probe performance by layer")
    axes[1].plot(layers, accs, "g-s", markersize=4)
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0, 1)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(layers, eces, "r-^", markersize=4)
    axes[2].set_xlabel("Layer index")
    axes[2].set_ylabel("ECE")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_cross_scenario(data: Dict[str, Any], plt) -> Optional[Any]:
    cs = data.get("cross_scenario_results") or {}
    if not cs:
        return None
    train_scenarios = sorted(cs.keys())
    test_scenarios = set()
    grid: Dict[Tuple[str, str], float] = {}
    for tr_s, block in cs.items():
        tests = block.get("test_results") or {}
        for te_s, te_m in tests.items():
            test_scenarios.add(te_s)
            if isinstance(te_m, dict) and "auc" in te_m:
                grid[(tr_s, te_s)] = float(te_m["auc"])
    test_scenarios = sorted(test_scenarios)
    if not grid or not train_scenarios or not test_scenarios:
        return None
    mat = np.full((len(train_scenarios), len(test_scenarios)), np.nan)
    for i, tr in enumerate(train_scenarios):
        for j, te in enumerate(test_scenarios):
            mat[i, j] = grid.get((tr, te), np.nan)
    fig, ax = plt.subplots(figsize=(max(6, len(test_scenarios)), max(4, len(train_scenarios))))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(test_scenarios)))
    ax.set_xticklabels(test_scenarios, rotation=45, ha="right")
    ax.set_yticks(range(len(train_scenarios)))
    ax.set_yticklabels(train_scenarios)
    ax.set_xlabel("Test scenario")
    ax.set_ylabel("Train scenario")
    ax.set_title("Cross-scenario probe AUROC (train on row, test on column)")
    plt.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    return fig


def _fig_feature_importance(probe_path: Path, plt, top_k: int = 40) -> Optional[Any]:
    from .linear_probe import LinearProbeTrainer

    probe = LinearProbeTrainer(device="cpu").load_probe(probe_path)
    w = probe.get_weights().detach().cpu().numpy()
    abs_w = np.abs(w)
    k = min(top_k, len(abs_w))
    idx = np.argsort(-abs_w)[:k]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(k), abs_w[idx], color="steelblue")
    ax.set_xlabel("Feature index (ranked)")
    ax.set_ylabel("|Probe weight|")
    ax.set_title(f"Top {k} probe weights (best layer)")
    fig.tight_layout()
    return fig


def _fig_autoencoder_loss(data: Dict[str, Any], plt) -> Optional[Any]:
    ar = data.get("autoencoder_results") or {}
    uns = ar.get("unsupervised_results") or {}
    if not uns:
        return None
    layers = []
    losses = []
    for name in sorted(uns.keys(), key=lambda x: int(x.split("_")[-1])):
        fm = uns[name].get("final_metrics") or {}
        if "reconstruction_loss" in fm:
            layers.append(int(name.split("_")[-1]))
            losses.append(float(fm["reconstruction_loss"]))
    if not layers:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, losses, "purple", marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Reconstruction loss (MSE)")
    ax.set_title("Unsupervised SAE reconstruction by layer")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_sklearn_auroc(data: Dict[str, Any], plt) -> Optional[Any]:
    pr = data.get("probe_results") or {}
    sk = pr.get("sklearn_logistic") or {}
    if not sk or "layer_0" not in sk:
        return None
    layers = []
    aucs = []
    for k in sorted(sk.keys(), key=lambda x: int(x.split("_")[-1]) if x.startswith("layer_") else 999):
        if not k.startswith("layer_") or k == "best_layer":
            continue
        v = sk[k]
        if isinstance(v, dict) and "auc" in v:
            layers.append(int(k.split("_")[-1]))
            aucs.append(float(v["auc"]))
    if not layers:
        return None
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, aucs, "c-o", markersize=4, label="Sklearn logistic (test)")
    tr = pr.get("test_results") or {}
    if tr:
        from .paper_results_plots import _layer_sort_key

        names = sorted(tr.keys(), key=_layer_sort_key)
        auc2 = [float(tr[n]["auc"]) for n in names]
        li = [_layer_sort_key(n) for n in names]
        ax.plot(li, auc2, "m--s", markersize=3, alpha=0.8, label="Torch probe (test)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("Sklearn vs torch linear probe (test set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    return fig


def _fig_probe_history(data: Dict[str, Any], plt) -> Optional[Any]:
    pr = data.get("probe_results") or {}
    best = pr.get("best_layer") or {}
    layer_name = best.get("layer_name")
    if not layer_name:
        return None
    lr = (pr.get("layer_results") or {}).get(layer_name, {})
    hist = (lr.get("results") or {}).get("history") or {}
    if not hist or "val_loss" not in hist:
        return None
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(hist["val_loss"], label="Val loss", color="tab:red")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val loss", color="tab:red")
    if "val_auc" in hist:
        ax2 = ax1.twinx()
        ax2.plot(hist["val_auc"], label="Val AUROC", color="tab:blue")
        ax2.set_ylabel("Val AUROC", color="tab:blue")
        ax2.set_ylim(0, 1)
    ax1.set_title(f"Probe training ({layer_name})")
    fig.legend(loc="upper right")
    fig.tight_layout()
    return fig


def _resolve_auxiliary_dir(
    experiment_dir: Path, auxiliary_dir: Optional[Path]
) -> Optional[Path]:
    if auxiliary_dir is not None:
        return Path(auxiliary_dir)
    names = ("causal_control_suite.json", "cross_context_steering.json")
    exp = Path(experiment_dir)
    for base in (exp, exp.parent):
        if base.is_dir() and any((base / n).exists() for n in names):
            return base
    return None


def _fig_roc_curves(npz_path: Path, plt) -> Optional[Any]:
    z = np.load(npz_path, allow_pickle=True)
    y = z["labels"].astype(int).ravel()
    s = z["probabilities"].astype(float).ravel()
    if len(np.unique(y)) < 2:
        return None
    try:
        from sklearn.metrics import auc, roc_curve
    except ImportError:
        return None
    fpr, tpr, _ = roc_curve(y, s)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — best linear probe (held-out test)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_activation_distributions(npz_path: Path, plt) -> Optional[Any]:
    z = np.load(npz_path, allow_pickle=True)
    norms = z["activation_l2_norm"].astype(float).ravel()
    y = z["labels"].astype(int).ravel()
    if norms.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, color, name in (
        (0, "tab:blue", "Truthful (label=0)"),
        (1, "tab:red", "Deceptive (label=1)"),
    ):
        m = y == lab
        if m.any():
            ax.hist(
                norms[m],
                bins=30,
                alpha=0.5,
                color=color,
                label=name,
                density=True,
            )
    ax.set_xlabel(r"$\|\mathbf{h}\|_2$ at best probe layer")
    ax.set_ylabel("Density")
    ax.set_title("Activation norm distribution (test set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_statistical_comparisons(json_path: Path, plt) -> Optional[Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    rows = []
    for label, key in (("AUROC", "auroc"), ("Accuracy", "accuracy")):
        block = d.get(key) or {}
        if block.get("error") or "ci_low" not in block:
            continue
        pe = float(block.get("point_estimate", block.get("bootstrap_mean", 0.0)))
        lo = float(block["ci_low"])
        hi = float(block["ci_high"])
        rows.append((label, pe, pe - lo, hi - pe))
    if not rows:
        return None
    labels, centers, el, eh = zip(*rows)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(5, 4))
    colors = ["steelblue", "seagreen"][: len(rows)]
    ax.bar(
        x,
        centers,
        yerr=[list(el), list(eh)],
        capsize=6,
        color=colors,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value (bootstrap 95% CI)")
    ax.set_title("Held-out test: point estimate with bootstrap CI")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_causal_control_response_lengths(json_path: Path, plt) -> Optional[Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or not rows:
        return None
    conds = [r.get("condition", "?") for r in rows]
    lengths = [len(str(r.get("text", ""))) for r in rows]
    fig, ax = plt.subplots(figsize=(max(8, len(conds) * 0.35), 4))
    ax.bar(range(len(conds)), lengths, color="slategray")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(conds, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Response length (chars)")
    ax.set_title("Causal control suite — generation length by condition")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def _fig_cross_context_response_quality(json_path: Path, plt) -> Optional[Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    b_toks: List[float] = []
    s_toks: List[float] = []
    if not isinstance(data, dict):
        return None
    for _src, block in data.items():
        targets = (block or {}).get("targets") or {}
        for _tgt, rows in targets.items():
            if not isinstance(rows, list):
                continue
            for r in rows:
                qb = r.get("quality_baseline") or {}
                qs = r.get("quality_steered") or {}
                if "num_tokens" in qb and "num_tokens" in qs:
                    b_toks.append(float(qb["num_tokens"]))
                    s_toks.append(float(qs["num_tokens"]))
    if not b_toks:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    means = [float(np.mean(b_toks)), float(np.mean(s_toks))]
    sem = [
        float(np.std(b_toks) / max(np.sqrt(len(b_toks)), 1)),
        float(np.std(s_toks) / max(np.sqrt(len(s_toks)), 1)),
    ]
    ax.bar([0, 1], means, yerr=sem, capsize=6, color=["tab:blue", "tab:orange"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Baseline", "Probe-steered"])
    ax.set_ylabel("Mean num_tokens (± SEM)")
    ax.set_title("Cross-context steering — response length")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def build_paper_figures_from_experiment_dir(
    experiment_output_dir: Union[str, Path],
    paper_figures_dir: Optional[Path] = None,
    auxiliary_dir: Optional[Path] = None,
    dpi: int = 200,
) -> Dict[str, Any]:
    """
    Build all figures from ``experiment_output_dir/experiment_results.json``.

    ``auxiliary_dir`` (optional) is the directory that holds
    ``causal_control_suite.json`` / ``cross_context_steering.json`` when probes
    were saved under ``.../probes/``; if omitted, the parent of
    ``experiment_output_dir`` is tried when those files exist there.

    Returns a summary dict with `figures` list of written paths and `tracking` path.
    """
    exp = Path(experiment_output_dir)
    json_path = exp / "experiment_results.json"
    out_dir = paper_figures_dir or _paper_figures_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not json_path.exists():
        logger.warning("No experiment_results.json at %s — skip figures", json_path)
        return {"error": "missing experiment_results.json", "output_dir": str(out_dir)}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    plt = _use_matplotlib()
    written: List[str] = []
    builders = [
        ("layer_performance", _fig_layer_performance),
        ("cross_scenario_generalization", _fig_cross_scenario),
        ("autoencoder_reconstruction", _fig_autoencoder_loss),
        ("sklearn_logistic_auroc", _fig_sklearn_auroc),
        ("probe_training_history", _fig_probe_history),
    ]

    for stem, fn in builders:
        try:
            fig = fn(data, plt)
            if fig is None:
                continue
            paths = _save_fig(fig, out_dir / stem, dpi=dpi)
            plt.close(fig)
            written.extend(str(p) for p in paths)
        except Exception as e:
            logger.warning("Figure %s failed: %s", stem, e)

    # Feature importance (needs checkpoint)
    models_dir = exp / "models"
    if models_dir.exists():
        probes = list(models_dir.glob("best_probe_*.pt"))
        if probes:
            try:
                fig = _fig_feature_importance(probes[0], plt)
                if fig is not None:
                    paths = _save_fig(fig, out_dir / "feature_importance", dpi=dpi)
                    plt.close(fig)
                    written.extend(str(p) for p in paths)
            except Exception as e:
                logger.warning("feature_importance failed: %s", e)

    npz_path = exp / "test_predictions_best_probe.npz"
    if npz_path.exists():
        for stem, fn in (
            ("roc_curves", lambda: _fig_roc_curves(npz_path, plt)),
            ("activation_distributions", lambda: _fig_activation_distributions(npz_path, plt)),
        ):
            try:
                fig = fn()
                if fig is None:
                    continue
                paths = _save_fig(fig, out_dir / stem, dpi=dpi)
                plt.close(fig)
                written.extend(str(p) for p in paths)
            except Exception as e:
                logger.warning("Figure %s failed: %s", stem, e)

    boot_path = exp / "bootstrap_confidence.json"
    if boot_path.exists():
        try:
            fig = _fig_statistical_comparisons(boot_path, plt)
            if fig is not None:
                paths = _save_fig(fig, out_dir / "statistical_comparisons", dpi=dpi)
                plt.close(fig)
                written.extend(str(p) for p in paths)
        except Exception as e:
            logger.warning("statistical_comparisons failed: %s", e)

    aux_base = _resolve_auxiliary_dir(exp, auxiliary_dir)
    if aux_base is not None:
        causal_path = aux_base / "causal_control_suite.json"
        if causal_path.exists():
            try:
                fig = _fig_causal_control_response_lengths(causal_path, plt)
                if fig is not None:
                    paths = _save_fig(
                        fig, out_dir / "causal_control_response_lengths", dpi=dpi
                    )
                    plt.close(fig)
                    written.extend(str(p) for p in paths)
            except Exception as e:
                logger.warning("causal_control_response_lengths failed: %s", e)
        steer_path = aux_base / "cross_context_steering.json"
        if steer_path.exists():
            try:
                fig = _fig_cross_context_response_quality(steer_path, plt)
                if fig is not None:
                    paths = _save_fig(
                        fig, out_dir / "cross_context_response_quality", dpi=dpi
                    )
                    plt.close(fig)
                    written.extend(str(p) for p in paths)
            except Exception as e:
                logger.warning("cross_context_response_quality failed: %s", e)

    tracking = {
        "source": str(json_path.resolve()),
        "experiment_output_dir": str(exp.resolve()),
        "paper_figures_dir": str(out_dir.resolve()),
        "auxiliary_dir": str(aux_base.resolve()) if aux_base else None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figures": written,
    }
    track_path = out_dir / "visualization_tracking.json"
    with open(track_path, "w", encoding="utf-8") as f:
        json.dump(tracking, f, indent=2)

    tracking["tracking_path"] = str(track_path)
    return tracking
