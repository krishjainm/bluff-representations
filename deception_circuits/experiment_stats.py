"""
Bootstrap confidence intervals for paper statistics (held-out test set).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Percentile bootstrap CI for AUROC (binary labels)."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {"error": "sklearn required"}

    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return {"error": "insufficient_class_diversity", "n": n}

    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(float(roc_auc_score(yt, ys)))
    if not aucs:
        return {"error": "no_valid_bootstrap_samples"}
    aucs_arr = np.array(aucs)
    q_lo, q_hi = np.quantile(aucs_arr, [alpha / 2, 1.0 - alpha / 2])
    return {
        "metric": "auroc",
        "point_estimate": float(roc_auc_score(y_true, y_score)),
        "bootstrap_mean": float(aucs_arr.mean()),
        "ci_low": float(q_lo),
        "ci_high": float(q_hi),
        "n_bootstrap_valid": int(len(aucs)),
        "alpha": alpha,
    }


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    n_bootstrap: int = 2000,
    seed: int = 43,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    preds = (y_score >= threshold).astype(int)
    n = len(y_true)
    if n < 1:
        return {"error": "empty"}

    rng = np.random.default_rng(seed)
    accs: List[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        accs.append(float((preds[idx] == y_true[idx]).mean()))
    accs_arr = np.array(accs)
    q_lo, q_hi = np.quantile(accs_arr, [alpha / 2, 1.0 - alpha / 2])
    return {
        "metric": "accuracy_at_0.5",
        "point_estimate": float((preds == y_true).mean()),
        "bootstrap_mean": float(accs_arr.mean()),
        "ci_low": float(q_lo),
        "ci_high": float(q_hi),
        "n_bootstrap_valid": int(len(accs)),
        "alpha": alpha,
    }


def build_bootstrap_summary(
    y_true: np.ndarray, y_score: np.ndarray, **kwargs
) -> Dict[str, Any]:
    out = {
        "auroc": bootstrap_auroc_ci(y_true, y_score, **kwargs),
        "accuracy": bootstrap_accuracy_ci(y_true, y_score, **kwargs),
    }
    return out
