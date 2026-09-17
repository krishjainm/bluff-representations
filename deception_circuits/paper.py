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
from dataclasses import asdict, dataclass, field
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
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder

REQUIRED_COLUMNS = {"sample_id", "base_item_id", "statement", "response", "label", "scenario"}


class ResearchIntegrityError(ValueError):
    """Raised when an input cannot support a real research experiment."""


def render_predecision_prompt(statement: str, template: str = "{statement}\nAction:") -> tuple[str, int]:
    """Render a controlled decision prompt and return its character boundary.

    The response is intentionally not an argument: this makes accidental
    response-token probing impossible in the primary prompt-end path.
    Tokenizers must record the corresponding final-token index at extraction.
    """
    if "{statement}" not in template:
        raise ResearchIntegrityError("Prompt template must contain {statement}")
    prompt = template.format(statement=str(statement))
    if not prompt.strip().endswith("Action:"):
        raise ResearchIntegrityError("Pre-decision prompt template must end at an explicit 'Action:' boundary")
    return prompt, len(prompt)


@dataclass(frozen=True)
class PaperConfig:
    experiment_name: str
    dataset_path: str
    activation_dir: str
    output_dir: str
    seed: int = 2026
    split_manifest_path: str | None = None
    group_column: str = "split_group_id"
    split_strategy: str = "grouped_stratified"
    stratify_columns: list[str] = field(default_factory=lambda: ["label"])
    test_fraction: float = 0.20
    validation_fraction: float = 0.10
    probe_c: float = 1.0
    n_seeds: int = 5
    bootstrap_resamples: int = 1000
    activation_site: str = "residual_stream"
    activation_mode: str = "prompt_end"
    subject_model: str = "UNSPECIFIED"
    model_revision: str | None = None
    tokenizer_revision: str | None = None
    activation_layers: str = "all"
    max_sequence_length: int | None = None
    prompt_template_id: str = "unspecified"
    decision_boundary_marker: str = "Action:"
    # None keeps the mode-appropriate default ("{statement}\nAction:"). Set this
    # when the prompt needs a different scaffold, and keep it consistent with
    # decision_boundary_marker: the rendered prompt must end at that marker.
    prompt_template: str | None = None
    # Subject-model weight dtype: float32 | float16 | bfloat16. Activations are
    # always stored float32, so this trades model memory and speed only.
    torch_dtype: str = "float32"
    probe_type: str = "logistic_regression"
    sae_config: dict[str, Any] = field(default_factory=dict)
    sae_top_n_features: int = 5
    sae_stability_seeds: list[int] = field(default_factory=list)
    steering_direction_source: str | None = None
    intervention_layer: int | None = None
    intervention_site: str | None = None
    intervention_token_policy: str | None = None
    intervention_strengths: list[float] = field(default_factory=list)
    # Causal evaluation size is configurable rather than the reviewers' criticised 100.
    causal_eval_size: int = 300
    endpoint_options: list[str] = field(default_factory=list)
    endpoint_positive_option: str | None = None
    causal_nuisance_column: str | None = None
    decoding_settings: dict[str, Any] = field(default_factory=dict)
    nuisance_columns: list[str] = field(default_factory=list)
    learning_curve_sizes: list[int] = field(default_factory=list)
    learning_curve_subsamples: int = 5

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


def load_activations(
    df: pd.DataFrame,
    activation_dir: str | Path,
    config: "PaperConfig | None" = None,
    *,
    require_manifest: bool = True,
) -> np.ndarray:
    """Load one real [layers, hidden] tensor per sample, failing on any defect.

    By default an extraction provenance manifest must exist and cover every
    dataset row.  When ``config`` is supplied the manifest is additionally
    checked against it, so a ``prompt_end`` run cannot silently consume
    ``response_token`` artifacts.  ``require_manifest=False`` exists only for
    unit tests of the tensor-level checks themselves.
    """
    root = Path(activation_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Activation directory is missing: {root}")
    if require_manifest:
        # Imported here: paper_extraction depends on this module.
        from .paper_extraction import verify_extraction_manifest
        verify_extraction_manifest(df, root, config)
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


def _resolve_stratify_columns(df: pd.DataFrame, config: PaperConfig, n_folds: int) -> list[str]:
    """Pick the richest stratification the group count can actually support.

    Stratifying on more columns is better balanced but needs more groups per
    stratum.  Rather than failing on a small dataset, drop the optional columns
    and record what was actually used in the manifest.
    """
    requested = [c for c in config.stratify_columns if c in df.columns]
    if "label" not in requested:
        requested = ["label"] + requested
    group_col = config.group_column if config.group_column in df else "base_item_id"
    while len(requested) > 1:
        key = df[requested].astype(str).agg("|".join, axis=1)
        groups_per_stratum = df.assign(_k=key).groupby("_k")[group_col].nunique()
        if groups_per_stratum.min() >= n_folds:
            return requested
        requested = requested[:-1]
    return requested


def _stratified_group_split(
    df: pd.DataFrame, groups: np.ndarray, strata: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """One group-safe, label-stratified holdout of approximately ``fraction``."""
    n_folds = int(max(2, min(round(1.0 / fraction), len(np.unique(groups)))))
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    keep_idx, hold_idx = next(splitter.split(df, y=strata, groups=groups))
    return keep_idx, hold_idx


def make_split_manifest(df: pd.DataFrame, config: PaperConfig) -> dict[str, Any]:
    """Create a deterministic group-safe train/validation/test manifest.

    ``split_strategy='grouped_stratified'`` (the default) keeps groups intact
    *and* balances the label across partitions, so a small dataset cannot land a
    partition with only one class.  ``'grouped_random'`` is the earlier
    group-only behaviour, kept for comparison.
    """
    group_col = config.group_column if config.group_column in df else "base_item_id"
    groups = df[group_col].astype(str).to_numpy()
    unique_groups = np.unique(groups)
    if len(unique_groups) < 6:
        raise ResearchIntegrityError("At least six independent groups are required for paper splits")
    if config.split_strategy not in ("grouped_stratified", "grouped_random"):
        raise ResearchIntegrityError(
            f"split_strategy must be grouped_stratified or grouped_random, got {config.split_strategy!r}")

    rel_val = config.validation_fraction / (1.0 - config.test_fraction)
    if config.split_strategy == "grouped_stratified":
        n_folds = int(max(2, round(1.0 / config.test_fraction)))
        stratify_columns = _resolve_stratify_columns(df, config, n_folds)
        strata = df[stratify_columns].astype(str).agg("|".join, axis=1).to_numpy()
        trainval_idx, test_idx = _stratified_group_split(
            df, groups, strata, config.test_fraction, config.seed)
        trainval = df.iloc[trainval_idx]
        train_local, val_local = _stratified_group_split(
            trainval, groups[trainval_idx], strata[trainval_idx], rel_val, config.seed + 1)
        train_idx, val_idx = trainval_idx[train_local], trainval_idx[val_local]
    else:
        stratify_columns = []
        # GroupShuffleSplit avoids row-level leakage but does not balance labels.
        outer = GroupShuffleSplit(n_splits=1, test_size=config.test_fraction, random_state=config.seed)
        trainval_idx, test_idx = next(outer.split(df, groups=groups))
        trainval = df.iloc[trainval_idx]
        inner = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=config.seed + 1)
        train_local, val_local = next(inner.split(trainval, groups=trainval[group_col].astype(str)))
        train_idx, val_idx = trainval_idx[train_local], trainval_idx[val_local]

    result = {
        "schema_version": 2, "dataset_sha256": _sha256(Path(config.dataset_path)),
        "group_column": group_col, "seed": config.seed,
        "split_strategy": config.split_strategy,
        "stratify_columns": stratify_columns,
        "train": df.iloc[train_idx]["sample_id"].tolist(),
        "validation": df.iloc[val_idx]["sample_id"].tolist(),
        "test": df.iloc[test_idx]["sample_id"].tolist(),
    }
    result["partition_summary"] = summarize_partitions(df, result)
    assert_split_integrity(df, result)
    return result


def summarize_partitions(df: pd.DataFrame, manifest: dict[str, Any]) -> dict[str, Any]:
    """Per-partition row/group counts and label balance, for the manifest and audit."""
    group_col = manifest.get("group_column", "base_item_id")
    summary: dict[str, Any] = {}
    for name in ("train", "validation", "test"):
        rows = df[df.sample_id.isin(set(manifest[name]))]
        counts = rows.label.value_counts().sort_index()
        summary[name] = {
            "n_rows": int(len(rows)),
            "n_groups": int(rows[group_col].nunique()) if group_col in rows.columns else None,
            "label_counts": {str(k): int(v) for k, v in counts.items()},
            "positive_rate": float(rows.label.astype(float).mean()) if len(rows) else None,
        }
    return summary


def assert_partition_label_availability(df: pd.DataFrame, manifest: dict[str, Any]) -> None:
    """Every partition must carry both classes, or no metric there is meaningful."""
    summary = manifest.get("partition_summary") or summarize_partitions(df, manifest)
    for name, info in summary.items():
        present = {k for k, v in info["label_counts"].items() if v > 0}
        if present != {"0", "1"}:
            raise ResearchIntegrityError(
                f"{name} partition does not contain both label classes "
                f"(counts: {info['label_counts']}). Increase the dataset, rebalance it, or "
                "use split_strategy=grouped_stratified."
            )


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
    assert_partition_label_availability(df, manifest)


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_split_manifest(
    path: str | Path,
    df: pd.DataFrame,
    config: PaperConfig,
) -> dict[str, Any]:
    """Load a split manifest only if it belongs to the current dataset."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Split manifest is missing: {path}")

    manifest = json.loads(path.read_text(encoding="utf-8"))

    recorded_hash = manifest.get("dataset_sha256")
    if not recorded_hash:
        raise ResearchIntegrityError(
            "Split manifest is missing dataset_sha256; recreate the split manifest "
            "before running research analyses."
        )

    current_hash = _sha256(Path(config.dataset_path))
    if recorded_hash != current_hash:
        raise ResearchIntegrityError(
            "Split manifest dataset SHA-256 does not match the current dataset. "
            f"manifest={recorded_hash}, current={current_hash}. "
            "The dataset changed after the split was created; recreate the split "
            "manifest before continuing."
        )

    assert_split_integrity(df, manifest)
    return manifest


def binary_ece(probabilities: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:

    """Equal-width expected calibration error for binary labels.

    This is the canonical implementation for the strict path;
    ``linear_probe.binary_ece`` re-exports it so the two cannot drift apart.
    Unlike the legacy version it raises on empty input rather than returning a
    calibrated-looking 0.0.
    """

    probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    labels = np.asarray(labels, dtype=np.float64).ravel()
    if len(probabilities) == 0:
        raise ResearchIntegrityError("Cannot compute ECE on an empty prediction set")
    if len(probabilities) != len(labels):
        raise ResearchIntegrityError("ECE inputs must have equal length")
    if n_bins < 1:
        raise ResearchIntegrityError("n_bins must be >= 1")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.clip(np.digitize(probabilities, edges, right=False) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        ece += (mask.sum() / len(probabilities)) * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(ece)


def compute_binary_metrics(y: np.ndarray, p: np.ndarray, *, ece_bins: int = 15) -> dict[str, Any]:
    """Full binary classification metric set, including calibration error.

    Raises rather than degrading to a chance-level score when a partition lacks
    both classes: a silent 0.5 would be indistinguishable from a real result.
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=np.float64)
    if len(np.unique(y)) != 2:
        raise ResearchIntegrityError("Metrics require both label classes")
    pred = (p >= .5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auroc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "accuracy": float(accuracy_score(y, pred)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)), "mcc": float(matthews_corrcoef(y, pred)),
            "ece": binary_ece(p, y, n_bins=ece_bins), "n": int(len(y)),
            "positive_rate": float(np.mean(np.asarray(y, dtype=float))),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


# Internal alias kept so existing call sites read compactly.
_metrics = compute_binary_metrics


def grouped_bootstrap_ci(
    y: np.ndarray, probabilities: np.ndarray, groups: Iterable[Any], *, seed: int,
    n_resamples: int = 1000, metric: str = "auroc",
) -> dict[str, Any]:
    """Grouped percentile CI for AUROC/PR-AUC, rejecting invalid resamples.

    Sampling groups rather than rows avoids presenting clustered poker hands or
    paired items as independent observations.
    """
    y, probabilities = np.asarray(y), np.asarray(probabilities)
    groups = np.asarray(list(groups))
    if not (len(y) == len(probabilities) == len(groups)):
        raise ResearchIntegrityError("Bootstrap inputs must have equal length")
    score = {"auroc": roc_auc_score, "pr_auc": average_precision_score}.get(metric)
    if score is None:
        raise ValueError("metric must be auroc or pr_auc")
    unique = np.unique(groups)
    # Precompute group -> row indices once. Recomputing `groups == group` inside the
    # resample loop is O(n_groups * n_rows) per resample, which on a test partition
    # of ~9k near-singleton groups dominates the entire probe stage.
    members = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_resamples):
        selected = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([members[group] for group in selected])
        if len(np.unique(y[idx])) == 2:
            values.append(float(score(y[idx], probabilities[idx])))
    if not values:
        raise ResearchIntegrityError("No valid grouped bootstrap resamples; check class/group distribution")
    return {"metric": metric, "lower": float(np.quantile(values, .025)),
            "upper": float(np.quantile(values, .975)), "valid_resamples": len(values),
            "requested_resamples": n_resamples}


def _text_baseline(train_text: pd.Series, y_train: np.ndarray, test_text: pd.Series, y_test: np.ndarray, seed: int) -> dict[str, Any]:
    classifier = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("logistic", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=seed)),
    ])
    classifier.fit(train_text.astype(str), y_train)
    return _metrics(y_test, classifier.predict_proba(test_text.astype(str))[:, 1])


def run_baselines(df: pd.DataFrame, manifest: dict[str, Any], config: PaperConfig) -> dict[str, Any]:
    """Cheap leakage/confound baselines, fit only on training rows."""
    train = df.set_index("sample_id").loc[manifest["train"]]
    test = df.set_index("sample_id").loc[manifest["test"]]
    y_train, y_test = train.label.to_numpy(int), test.label.to_numpy(int)
    result: dict[str, Any] = {"majority_class_accuracy": float(max(np.mean(y_test == 0), np.mean(y_test == 1)))}
    print("[baselines] fitting prompt-text tf-idf", flush=True)
    result["prompt_text"] = _text_baseline(train.statement, y_train, test.statement, y_test, config.seed)
    print("[baselines] fitting response-text diagnostic tf-idf", flush=True)
    # This is a deliberately diagnostic baseline, never a pre-decision claim.
    result["response_text_diagnostic"] = _text_baseline(train.response, y_train, test.response, y_test, config.seed)
    columns = [c for c in config.nuisance_columns if c in df.columns]
    if columns:
        # String coercion makes categorical nuisance variables explicit and avoids
        # interpreting arbitrary IDs as ordinal measurements.
        encoder = OneHotEncoder(handle_unknown="ignore")
        x_train = encoder.fit_transform(train[columns].fillna("__MISSING__").astype(str))
        x_test = encoder.transform(test[columns].fillna("__MISSING__").astype(str))
        model = LogisticRegression(class_weight="balanced", max_iter=5000, random_state=config.seed)
        model.fit(x_train, y_train)
        result["nuisance_only"] = {"columns": columns, **_metrics(y_test, model.predict_proba(x_test)[:, 1])}
    else:
        result["nuisance_only"] = {"status": "not_run", "reason": "no configured nuisance columns available"}
    return result

def run_probe_experiment(
    df: pd.DataFrame,
    activations: np.ndarray,
    manifest: dict[str, Any],
    config: PaperConfig,
    *,
    layer_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Select layers solely on validation, then evaluate frozen choices on test."""
    if layer_indices is None:
        layer_indices = list(range(activations.shape[1]))

    layer_indices = [int(layer) for layer in layer_indices]

    if len(layer_indices) != activations.shape[1]:
        raise ResearchIntegrityError(
            "Physical layer mapping length must match the activation tensor layer dimension."
        )

    if len(set(layer_indices)) != len(layer_indices) or any(layer < 0 for layer in layer_indices):
        raise ResearchIntegrityError(
            "Physical layer mapping must contain unique, non-negative transformer layers."
        )

    index = {sid: i for i, sid in enumerate(df.sample_id)}

    def take(part: str):
        ids = manifest[part]; ii = np.array([index[s] for s in ids])
        return activations[ii], df.iloc[ii].label.to_numpy(dtype=int)
    xtr, ytr = take("train"); xva, yva = take("validation"); xte, yte = take("test")
    if any(len(np.unique(y)) != 2 for y in (ytr, yva, yte)):
        raise ResearchIntegrityError("Every partition needs both labels for a paper probe run")
    group_col = manifest.get("group_column", "base_item_id")
    test_groups = df.iloc[[index[s] for s in manifest["test"]]][group_col].to_numpy()
    runs = []
    n_layers = activations.shape[1]
    print(f"[probe] {config.n_seeds} seeds x {n_layers} layers on "
          f"{len(ytr)} train / {len(yva)} val / {len(yte)} test rows", flush=True)
    for index, seed in enumerate(range(config.seed, config.seed + config.n_seeds), start=1):
        started = datetime.now(timezone.utc)

        selected_axis, val_scores, model = _select_layer_on_validation(
            xtr,
            ytr,
            xva,
            yva,
            config,
            seed,
        )
        selected_physical_layer = layer_indices[selected_axis]

        test_prob = model.predict_proba(xte[:, selected_axis])[:, 1]

        runs.append({
            "seed": seed,
            # Transitional compatibility field. Downstream code will be migrated
            # to the explicit axis/physical-layer fields in the next steps.
            "selected_layer": selected_axis,
            "selected_layer_index": selected_axis,
            "selected_physical_layer": selected_physical_layer,
            "validation_auroc": val_scores[selected_axis],
            "validation_auroc_by_layer": [float(v) for v in val_scores],
            "test": _metrics(yte, test_prob),
            "test_auroc_grouped_ci": grouped_bootstrap_ci(
                yte,
                test_prob,
                test_groups,
                seed=seed,
                n_resamples=config.bootstrap_resamples,
                metric="auroc",
            ),
            "test_pr_auc_grouped_ci": grouped_bootstrap_ci(
                yte,
                test_prob,
                test_groups,
                seed=seed,
                n_resamples=config.bootstrap_resamples,
                metric="pr_auc",
            ),
        })

        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        print(
            f"[probe] seed {index}/{config.n_seeds} done in {elapsed:.0f}s: "
            f"tensor axis {selected_axis} -> model layer {selected_physical_layer}, "
            f"val AUROC {val_scores[selected_axis]:.4f}, "
            f"test AUROC {runs[-1]['test']['auroc']:.4f}",
            flush=True,
        )

    result = {
        "selection_partition": "validation",
        "test_partition_used_for_selection": False,
        "activation_layer_indices": [int(layer) for layer in layer_indices],
        "n_train": len(ytr),
        "n_validation": len(yva),
        "n_test": len(yte),
        "runs": runs,
        "test_auroc_mean": float(
            np.mean([r["test"]["auroc"] for r in runs])
        ),
        "test_pr_auc_mean": float(
            np.mean([r["test"]["pr_auc"] for r in runs])
        ),
        "seed_summary": summarize_across_seeds(runs),
        "baselines": run_baselines(df, manifest, config),
    }

    if config.learning_curve_sizes:
        # The curve reuses the activation tensor axis selected by the main probe
        # loop. The physical transformer layer is carried separately.
        modal_layer_index = result["seed_summary"]["selected_layer_index"]["modal"]
        result["learning_curve"] = run_learning_curve(
            df,
            activations,
            manifest,
            config,
            layer=modal_layer_index,
            layer_indices=layer_indices,
            layer_source=(
                f"modal validation-selected layer across {config.n_seeds} seeds"
            ),
        )
    return result


def _select_layer_on_validation(
    xtr: np.ndarray, ytr: np.ndarray, xva: np.ndarray, yva: np.ndarray,
    config: PaperConfig, seed: int,
) -> tuple[int, list[float], LogisticRegression]:
    """Fit one probe per layer on train, return the best-by-validation layer.

    The returned model is the one fitted for the selected layer, so reported test
    metrics always come from the model that validation actually chose.
    """
    val_scores: list[float] = []
    models: list[LogisticRegression] = []
    for layer in range(xtr.shape[1]):
        clf = LogisticRegression(C=config.probe_c, class_weight="balanced", max_iter=5000, random_state=seed)
        clf.fit(xtr[:, layer], ytr)
        val_scores.append(_metrics(yva, clf.predict_proba(xva[:, layer])[:, 1])["auroc"])
        models.append(clf)
    selected = int(np.argmax(val_scores))
    return selected, val_scores, models[selected]


def summarize_across_seeds(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Seed-level variability, reported alongside the within-seed grouped CIs.

    These are different sources of uncertainty: the grouped bootstrap covers
    sampling of poker hands, this covers probe-fitting randomness.  Reviewer B
    asked for both.
    """
    def spread(values: list[float]) -> dict[str, Any]:
        arr = np.asarray(values, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "median": float(np.median(arr)),
                "iqr": [float(np.quantile(arr, .25)), float(np.quantile(arr, .75))],
                "min": float(arr.min()), "max": float(arr.max()), "n_seeds": int(len(arr))}
    layer_indices = [
        int(r.get("selected_layer_index", r["selected_layer"]))
        for r in runs
    ]
    physical_layers = [
        int(r.get("selected_physical_layer", r["selected_layer"]))
        for r in runs
    ]

    def layer_summary(values: list[int]) -> dict[str, Any]:
        return {
            "values": values,
            "modal": max(set(values), key=values.count),
            "n_distinct": len(set(values)),
        }

    return {
        "test_auroc": spread([r["test"]["auroc"] for r in runs]),
        "test_pr_auc": spread([r["test"]["pr_auc"] for r in runs]),
        "test_ece": spread([r["test"]["ece"] for r in runs]),
        # Compatibility alias until all downstream consumers are migrated.
        "selected_layer": layer_summary(layer_indices),
        "selected_layer_index": layer_summary(layer_indices),
        "selected_physical_layer": layer_summary(physical_layers),
    }


def run_learning_curve(
    df: pd.DataFrame,
    activations: np.ndarray,
    manifest: dict[str, Any],
    config: PaperConfig,
    *,
    layer: int | None = None,
    layer_indices: list[int] | None = None,
    layer_source: str = "unspecified",
) -> dict[str, Any]:
    """Sample-size curve using multiple *independent* training subsamples per size.

    Reviewer B's complaint was that sample-size comparisons used one arbitrary
    subset with no uncertainty.  Each size here is evaluated over
    ``learning_curve_subsamples`` independent draws, subsampled at **group**
    level so a smaller training set is genuinely fewer poker hands rather than
    fewer rows from the same hands.

    ``layer`` controls the dominant cost. With a fixed layer the curve asks "how
    much data does this probe need, holding the layer choice fixed", which is one
    fit per draw. With ``layer=None`` it re-selects across every layer inside
    each draw, which is ``n_layers`` times more fits and answers a noisier
    question -- on a 32-layer model that is 800 fits instead of 25. The policy
    and the layer's provenance are recorded either way.
    """
    if layer_indices is None:
        layer_indices = list(range(activations.shape[1]))

    layer_indices = [int(value) for value in layer_indices]

    if len(layer_indices) != activations.shape[1]:
        raise ResearchIntegrityError(
            "Physical layer mapping length must match the activation tensor layer dimension."
        )

    if len(set(layer_indices)) != len(layer_indices) or any(
        value < 0 for value in layer_indices
    ):
        raise ResearchIntegrityError(
            "Physical layer mapping must contain unique, non-negative transformer layers."
        )

    if layer is not None and not 0 <= int(layer) < activations.shape[1]:
        raise ResearchIntegrityError(
            f"Learning-curve layer index {layer} is outside the stored activation "
            f"tensor with {activations.shape[1]} layer axes."
        )

    index = {sid: i for i, sid in enumerate(df.sample_id)}
    group_col = manifest.get("group_column", "base_item_id")
    labels = df.label.to_numpy(int)
    train_ids = list(manifest["train"])
    train_groups = df.set_index("sample_id").loc[train_ids, group_col].astype(str)
    unique_groups = np.unique(train_groups.to_numpy())

    val_pos = np.array([index[s] for s in manifest["validation"]])
    test_pos = np.array([index[s] for s in manifest["test"]])
    xva, yva = activations[val_pos], labels[val_pos]
    xte, yte = activations[test_pos], labels[test_pos]

    points: list[dict[str, Any]] = []
    sizes = sorted({int(s) for s in config.learning_curve_sizes})
    print(f"[curve] {len(sizes)} sizes x {config.learning_curve_subsamples} draws"
          + (f" at fixed layer {layer}" if layer is not None else " re-selecting every layer"),
          flush=True)
    for size in sizes:
        if size < 2 or size > len(unique_groups):
            points.append({"n_train_groups": size, "status": "not_run",
                           "reason": f"requested {size} training groups but {len(unique_groups)} are available"})
            continue
        replicates: list[dict[str, Any]] = []
        for replicate in range(config.learning_curve_subsamples):
            rng = np.random.default_rng(config.seed + 1000 * size + replicate)
            chosen = rng.choice(unique_groups, size=size, replace=False)
            ids = train_groups.index[train_groups.isin(set(chosen))].tolist()
            pos = np.array([index[s] for s in ids])
            if len(np.unique(labels[pos])) < 2:
                continue
            if layer is None:
                selected, val_scores, model = _select_layer_on_validation(
                    activations[pos], labels[pos], xva, yva, config, config.seed + replicate)
            else:
                # Fixed layer: one fit per draw instead of one per layer per draw.
                selected = int(layer)
                model = LogisticRegression(C=config.probe_c, class_weight="balanced",
                                           max_iter=5000, random_state=config.seed + replicate)
                model.fit(activations[pos][:, selected], labels[pos])
                val_scores = [float("nan")] * activations.shape[1]
                val_scores[selected] = _metrics(
                    yva, model.predict_proba(xva[:, selected])[:, 1])["auroc"]
            selected_physical_layer = layer_indices[selected]

            replicates.append({
                "replicate": replicate,
                "n_train_rows": int(len(pos)),
                "n_train_groups": int(len(chosen)),
                # Compatibility alias: this remains the activation tensor axis.
                "selected_layer": selected,
                "selected_layer_index": selected,
                "selected_physical_layer": selected_physical_layer,
                # Identifies the draw without inlining the id list, so the curve
                # stays auditable and reproducible without bloating the JSON.
                "train_groups_sha256": hashlib.sha256(
                    "|".join(sorted(map(str, chosen))).encode("utf-8")
                ).hexdigest()[:16],
                "validation_auroc": float(val_scores[selected]),
                "test_auroc": float(
                    _metrics(
                        yte,
                        model.predict_proba(xte[:, selected])[:, 1],
                    )["auroc"]
                ),
            })
        if not replicates:
            points.append({"n_train_groups": size, "status": "not_run",
                           "reason": "no subsample of this size contained both label classes"})
            continue
        aurocs = np.asarray([r["test_auroc"] for r in replicates])
        print(f"[curve] {size} groups: {len(replicates)} draws, "
              f"test AUROC {aurocs.mean():.4f}", flush=True)
        points.append({
            "n_train_groups": size, "status": "ok", "n_replicates": len(replicates),
            "test_auroc_mean": float(aurocs.mean()),
            "test_auroc_std": float(aurocs.std(ddof=1)) if len(aurocs) > 1 else 0.0,
            "test_auroc_median": float(np.median(aurocs)),
            "test_auroc_range": [float(aurocs.min()), float(aurocs.max())],
            "replicates": replicates,
        })
    fixed_layer_index = None if layer is None else int(layer)
    fixed_physical_layer = (
        None if fixed_layer_index is None else layer_indices[fixed_layer_index]
    )

    return {
        "subsample_unit": "group",
        "requested_subsamples_per_size": config.learning_curve_subsamples,
        "layer_policy": "fixed" if layer is not None else "reselected_per_subsample",
        # Compatibility alias: this remains the activation tensor axis.
        "layer": fixed_layer_index,
        "layer_index": fixed_layer_index,
        "physical_layer": fixed_physical_layer,
        "layer_source": layer_source,
        "n_available_train_groups": int(len(unique_groups)),
        "points": points,
    }


def write_run_metadata(config: PaperConfig, output_dir: Path) -> None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unavailable"
    meta = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "git_commit": commit,
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "config": asdict(config)}
    (output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


def audit_run(output_dir: str | Path, df: pd.DataFrame | None = None, config: PaperConfig | None = None) -> list[str]:
    """Return every reason this experiment folder cannot support a paper claim."""
    root = Path(output_dir); failures = []
    if df is not None and config is not None:
        from .paper_extraction import audit_activation_provenance
        failures.extend(audit_activation_provenance(df, config.activation_dir, config))
    manifest_file = Path(config.split_manifest_path) if config and config.split_manifest_path else root / "split_manifest.json"
    for name in ("resolved_config.yaml", "run_metadata.json", "probe_results.json"):
        if not (root / name).is_file(): failures.append(f"missing {name}")
    if not manifest_file.is_file():
        failures.append(f"missing split manifest: {manifest_file}")
    if not failures:
        result = json.loads((root / "probe_results.json").read_text())
        if result.get("selection_partition") != "validation" or result.get("test_partition_used_for_selection"):
            failures.append("probe selection is not validation-only")
        if config is not None and len(result.get("runs", [])) != config.n_seeds:
            failures.append("number of completed probe seeds does not match config")
        if any("test_auroc_grouped_ci" not in run for run in result.get("runs", [])):
            failures.append("grouped AUROC confidence intervals are missing")
        if "baselines" not in result:
            failures.append("baseline results are missing")
        if "seed_summary" not in result:
            failures.append("seed-level variability summary is missing")
        if config is not None and config.learning_curve_sizes and "learning_curve" not in result:
            failures.append(
                "learning_curve_sizes is configured but no learning curve was produced"
            )

        if df is not None:
            from .paper_transfer import (
                audit_cross_context_transfer,
                available_contexts,
            )

            contexts = available_contexts(df)
            transfer_path = root / "cross_context_results.json"

            if len(contexts) >= 2:
                if not transfer_path.is_file():
                    failures.append(
                        "dataset contains multiple contexts but "
                        "cross_context_results.json is missing; run run-transfer"
                    )
                else:
                    transfer_result = json.loads(
                        transfer_path.read_text(encoding="utf-8")
                    )
                    failures.extend(
                        audit_cross_context_transfer(transfer_result)
                    )
            elif transfer_path.is_file():
                # Even when the current dataset has only one usable context,
                # any transfer artifact that exists must still be internally valid.
                transfer_result = json.loads(
                    transfer_path.read_text(encoding="utf-8")
                )
                failures.extend(
                    audit_cross_context_transfer(transfer_result)
                )

        if config is not None and config.nuisance_columns:
            confounds = root / "confound_results.json"
            if not confounds.is_file():
                failures.append(
                    "nuisance_columns are configured but confound_results.json is missing; "
                    "run analyze-confounds")
            elif "metadata_availability" not in json.loads(confounds.read_text()):
                failures.append("confound_results.json has no metadata availability report")
        if config is not None and config.sae_config:
            sae = root / "sae_results.json"
            if not sae.is_file():
                failures.append("sae_config is set but sae_results.json is missing; run train-sae")
            else:
                sae_result = json.loads(sae.read_text())
                if sae_result.get("unsupervised") is not True:
                    failures.append("SAE results do not declare unsupervised training")
                if sae_result.get("training", {}).get("labels_used_in_training") is not False:
                    failures.append("SAE results do not assert that training used no labels")
                if not sae_result.get("sae_frozen_before_evaluation"):
                    failures.append("SAE was not frozen before held-out evaluation")
                if not sae_result.get("sae_unchanged_after_evaluation"):
                    failures.append("SAE parameters changed during evaluation")
                diagnostics = sae_result.get("diagnostics") or {}
                missing_partitions = {"train", "validation", "test"} - set(diagnostics)
                if missing_partitions:
                    failures.append(f"SAE diagnostics missing partitions: {sorted(missing_partitions)}")
                for partition, entry in diagnostics.items():
                    absent = {"reconstruction_mse", "fraction_variance_explained", "l0_mean",
                              "dead_feature_fraction", "activation_frequency"} - set(entry)
                    if absent:
                        failures.append(f"SAE {partition} diagnostics missing {sorted(absent)}")
                        break
                held_out = sae_result.get("held_out_feature_evaluation") or {}
                if held_out.get("evaluation_partition") != "test":
                    failures.append("SAE features were not evaluated on the test partition")
                if not sae_result.get("feature_examples"):
                    failures.append("SAE results carry no top-activating example export")
        if config is not None and config.intervention_strengths:
            causal = root / "causal_results.json"
            if not causal.is_file():
                failures.append(
                    "intervention_strengths are configured but causal_results.json is missing; "
                    "run run-interventions")
            else:
                causal_result = json.loads(causal.read_text())
                if causal_result.get("primary_endpoint_is_probe_derived") is not False:
                    failures.append("causal results do not declare a probe-independent primary endpoint")
                required_conditions = {
                    "baseline", "positive_steering", "negative_steering", "random_matched_norm",
                    "orthogonal", "shuffled_label", "wrong_layer", "wrong_token_position",
                    "activation_patch_mismatched_prompt"}
                present = set(causal_result.get("conditions") or [])
                absent = sorted(required_conditions - present)
                if absent:
                    failures.append(f"causal control conditions are missing: {absent}")
                if not causal_result.get("quality_controls"):
                    failures.append("causal results carry no generation/format quality controls")
                for name, entry in (causal_result.get("effects") or {}).items():
                    for strength, stats in (entry.get("by_strength") or {}).items():
                        if "ci_lower" not in stats:
                            failures.append(f"causal effect {name}@{strength} has no bootstrap CI")
                            break
        manifest = json.loads(manifest_file.read_text())
        if df is not None:
            try:
                assert_split_integrity(df, manifest)
            except ResearchIntegrityError as exc:
                failures.append(f"split integrity failure: {exc}")
    return failures


def audit_notes(output_dir: str | Path, config: PaperConfig | None = None) -> list[str]:
    """Informational findings that are not audit failures.

    A control that could not run for a genuine data reason is a result to report,
    not a defect to block on; keeping these out of :func:`audit_run` means a PASS
    still means "nothing is wrong" rather than "nothing is missing".
    """
    root = Path(output_dir)
    notes: list[str] = []
    confounds = root / "confound_results.json"
    if confounds.is_file():
        result = json.loads(confounds.read_text())
        for name in result.get("unavailable_analyses") or []:
            reason = (result.get("subsets", {}).get(name) or {}).get("reason", "unspecified")
            notes.append(f"confound control {name!r} did not run: {reason}")
        blocked = result.get("metadata_availability", {}).get("unavailable_analyses") or []
        if blocked:
            notes.append("analyses blocked by missing metadata: " + ", ".join(blocked))
    probe = root / "probe_results.json"
    if probe.is_file():
        curve = json.loads(probe.read_text()).get("learning_curve") or {}
        for point in curve.get("points", []):
            if point.get("status") == "not_run":
                notes.append(f"learning-curve size {point['n_train_groups']} did not run: {point['reason']}")
    return notes
