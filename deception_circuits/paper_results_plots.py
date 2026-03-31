"""
Generate **paper-style figures** from ``experiment_results.json`` (probe curves).

Requires matplotlib (already in project requirements).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


def _layer_sort_key(name: str) -> int:
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return 0


def extract_probe_curves(
    experiment_results: Dict[str, Any],
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """Returns (layer_indices, auc, accuracy, ece) from ``probe_results.test_results``."""
    pr = experiment_results.get("probe_results") or {}
    tr = pr.get("test_results") or {}
    if not tr:
        return [], [], [], []
    names = sorted(tr.keys(), key=_layer_sort_key)
    layers, aucs, accs, eces = [], [], [], []
    for name in names:
        layers.append(_layer_sort_key(name))
        m = tr[name]
        aucs.append(float(m.get("auc", 0.0)))
        accs.append(float(m.get("accuracy", 0.0)))
        eces.append(float(m.get("ece", 0.0)) if m.get("ece") is not None else 0.0)
    return layers, aucs, accs, eces


def plot_probe_curves_from_json(
    json_path: Union[str, Path],
    output_dir: Union[str, Path],
    prefix: str = "probe_layers",
    dpi: int = 150,
) -> List[Path]:
    """
    Read ``experiment_results.json`` and write:
      - ``{prefix}_auc.png``
      - ``{prefix}_accuracy_ece.png`` (twin metrics)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    json_path = Path(json_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    layers, aucs, accs, eces = extract_probe_curves(data)
    written: List[Path] = []
    if not layers:
        return written

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(layers, aucs, "b-o", markersize=4, label="AUROC")
    ax.set_xlabel("Layer")
    ax.set_ylabel("AUROC")
    ax.set_title("Linear probe: deception AUROC by layer")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    p1 = out / f"{prefix}_auc.png"
    fig.savefig(p1, dpi=dpi)
    plt.close(fig)
    written.append(p1)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(layers, accs, "g-s", markersize=4, label="Accuracy")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Accuracy", color="g")
    ax1.tick_params(axis="y", labelcolor="g")
    ax2 = ax1.twinx()
    ax2.plot(layers, eces, "r-^", markersize=4, label="ECE")
    ax2.set_ylabel("ECE", color="r")
    ax2.tick_params(axis="y", labelcolor="r")
    ax1.set_title("Linear probe: accuracy & calibration by layer")
    fig.tight_layout()
    p2 = out / f"{prefix}_accuracy_ece.png"
    fig.savefig(p2, dpi=dpi)
    plt.close(fig)
    written.append(p2)

    return written
