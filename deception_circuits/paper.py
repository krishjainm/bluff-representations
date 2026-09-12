"""Strict, reproducible paper-experiment primitives.

This module is deliberately separate from the repository's legacy/demo
pipeline.  It never creates synthetic activations and it never uses the test
partition to select a layer or hyperparameter.
"""
from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit

REQUIRED_COLUMNS = {"sample_id", "base_item_id", "statement", "response", "label", "scenario"}


class ResearchIntegrityError(ValueError):
    """Raised when an input cannot support a real research experiment."""


@dataclass(frozen=True)
class PaperConfig:
    experiment_name: str
    dataset_path: str
    activation_dir: str
    output_dir: str
    seed: int = 2026
    split_manifest_path: str | None = None
    group_column: str = "split_group_id"
    test_fraction: float = 0.20
    validation_fraction: float = 0.10
    probe_c: float = 1.0
    n_seeds: int = 5
    bootstrap_resamples: int = 1000
    activation_site: str = "residual_stream"
    activation_mode: str = "prompt_end"
    subject_model: str = "UNSPECIFIED"
    model_revision: str | None = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PaperConfig":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        allowed = set(cls.__dataclass_fields__)
        unknown = set(raw) - allowed
        if unknown:
            raise ResearchIntegrityError(f"Unknown config fields: {sorted(unknown)}")
        missing = {"experiment_name", "dataset_path", "activation_dir", "output_dir"} - set(raw)
        if missing:
            raise ResearchIntegrityError(f"Missing config fields: {sorted(missing)}")
        return cls(**raw)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(path: str | Path) -> pd.DataFrame:
    """Load and strictly validate the canonical CSV schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ResearchIntegrityError(f"Dataset missing required columns: {sorted(missing)}")
    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ResearchIntegrityError("Required dataset fields contain missing values")
    if not df["sample_id"].is_unique:
        raise ResearchIntegrityError("sample_id values must be unique")
    labels = set(pd.to_numeric(df["label"], errors="coerce").dropna().astype(int))
    if labels - {0, 1} or len(labels) != 2:
        raise ResearchIntegrityError("label must contain both binary classes {0, 1}")
    group = "split_group_id" if "split_group_id" in df else "base_item_id"
    # A base item may be repeated but must not map to multiple explicit groups.
    if "split_group_id" in df and df.groupby("base_item_id")[group].nunique().gt(1).any():
        raise ResearchIntegrityError("Each base_item_id must map to exactly one split_group_id")
    return df


def load_activations(df: pd.DataFrame, activation_dir: str | Path) -> np.ndarray:
    """Load one real [layers, hidden] tensor per sample, failing on any defect."""
    root = Path(activation_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Activation directory is missing: {root}")
    tensors: list[np.ndarray] = []
    shape: tuple[int, ...] | None = None
    for sid in df["sample_id"]:
        path = root / f"sample_{sid}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing activation for sample_id={sid}: {path}")
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ResearchIntegrityError(f"{path} is not a tensor")
        arr = value.detach().cpu().numpy()
        if arr.ndim != 2 or not np.isfinite(arr).all():
            raise ResearchIntegrityError(f"{path} must be a finite [layers, hidden] tensor")
        if shape is None:
            shape = arr.shape
        elif arr.shape != shape:
            raise ResearchIntegrityError(f"Activation shape mismatch: expected {shape}, got {arr.shape} in {path}")
        tensors.append(arr.astype(np.float32, copy=False))
    return np.stack(tensors)


def make_split_manifest(df: pd.DataFrame, config: PaperConfig) -> dict[str, Any]:
    """Create a deterministic group-safe train/validation/test manifest."""
    group_col = config.group_column if config.group_column in df else "base_item_id"
    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 6:
        raise ResearchIntegrityError("At least six independent groups are required for paper splits")
    # GroupShuffleSplit deliberately avoids row-level leakage.  Its deterministic
    # split is preserved verbatim rather than recalculated at later stages.
    outer = GroupShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=config.seed)
    trainval_idx, test_idx = next(outer.split(df, groups=groups))
    trainval = df.iloc[trainval_idx]
    rel_val = config.validation_fraction / (1.0 - config.test_fraction)
    inner = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=config.seed + 1)
    train_idx_local, val_idx_local = next(inner.split(trainval, groups=trainval[group_col].astype(str)))
    train_idx = trainval_idx[train_idx_local]
    val_idx = trainval_idx[val_idx_local]
    result = {
        "schema_version": 1, "dataset_sha256": _sha256(Path(config.dataset_path)),
        "group_column": group_col, "seed": config.seed,
        "train": df.iloc[train_idx]["sample_id"].tolist(),
        "validation": df.iloc[val_idx]["sample_id"].tolist(),
        "test": df.iloc[test_idx]["sample_id"].tolist(),
    }
    assert_split_integrity(df, result)
    return result


def assert_split_integrity(df: pd.DataFrame, manifest: dict[str, Any]) -> None:
    parts = {k: set(manifest[k]) for k in ("train", "validation", "test")}
    if any(parts[a] & parts[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ResearchIntegrityError("Split manifest has overlapping sample IDs")
    if set().union(*parts.values()) != set(df.sample_id):
        raise ResearchIntegrityError("Split manifest does not cover the dataset exactly")
    group_col = manifest.get("group_column", "base_item_id")
    assignments = df[["sample_id", group_col]].copy()
    assignments["partition"] = assignments.sample_id.map({sid: p for p, ids in parts.items() for sid in ids})
    if assignments.groupby(group_col).partition.nunique().gt(1).any():
        raise ResearchIntegrityError("Group leakage detected across partitions")


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _metrics(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    if len(np.unique(y)) != 2:
        raise ResearchIntegrityError("Metrics require both label classes")
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def run_probe_experiment(df: pd.DataFrame, activations: np.ndarray, manifest: dict[str, Any], config: PaperConfig) -> dict[str, Any]:
    """Select layers solely on validation, then evaluate frozen choices on test."""
    index = {sid: i for i, sid in enumerate(df.sample_id)}
    def take(part: str):
        ids = manifest[part]; ii = np.array([index[s] for s in ids])
        return activations[ii], df.iloc[ii].label.to_numpy(dtype=int)
    xtr, ytr = take("train"); xva, yva = take("validation"); xte, yte = take("test")
    if any(len(np.unique(y)) != 2 for y in (ytr, yva, yte)):
        raise ResearchIntegrityError("Every partition needs both labels for a paper probe run")
    runs = []
    for seed in range(config.seed, config.seed + config.n_seeds):
        val_scores, models = [], []
        for layer in range(activations.shape[1]):
            clf = LogisticRegression(C=config.probe_c, class_weight="balanced", max_iter=5000, random_state=seed)
            clf.fit(xtr[:, layer], ytr)
            val_scores.append(_metrics(yva, clf.predict_proba(xva[:, layer])[:, 1])["auroc"]); models.append(clf)
        selected = int(np.argmax(val_scores))
        test_prob = models[selected].predict_proba(xte[:, selected])[:, 1]
        runs.append({"seed": seed, "selected_layer": selected, "validation_auroc": val_scores[selected], "test": _metrics(yte, test_prob)})
    return {"selection_partition": "validation", "test_partition_used_for_selection": False,
            "n_train": len(ytr), "n_validation": len(yva), "n_test": len(yte), "runs": runs,
            "test_auroc_mean": float(np.mean([r["test"]["auroc"] for r in runs])),
            "test_pr_auc_mean": float(np.mean([r["test"]["pr_auc"] for r in runs]))}


def write_run_metadata(config: PaperConfig, output_dir: Path) -> None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unavailable"
    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "config": asdict(config)}
    (output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def audit_run(output_dir: str | Path) -> list[str]:
    root = Path(output_dir); failures = []
    for name in ("resolved_config.yaml", "run_metadata.json", "split_manifest.json", "probe_results.json"):
        if not (root / name).is_file(): failures.append(f"missing {name}")
    if not failures:
        result = json.loads((root / "probe_results.json").read_text())
        if result.get("selection_partition") != "validation" or result.get("test_partition_used_for_selection"):
            failures.append("probe selection is not validation-only")
    return failures
