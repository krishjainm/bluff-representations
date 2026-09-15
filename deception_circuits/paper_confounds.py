"""Poker confound controls and nuisance baselines for the strict paper path.

The scientific question this module exists to answer is the brief's central one:
after controlling for poker strategy, game state, and action type, is there still
an internal representation associated with bluff-like deception?

Three complementary attacks on that question live here:

1. **Matched subsets** — hold the nuisance variables fixed by construction and
   re-run the probe (``build_matched_subset``, ``bluff_vs_value_subset``).
2. **Nuisance decodability** — ask how well the same hidden states predict the
   nuisance variables themselves, which bounds how much of the probe's apparent
   performance could be strategy decoding (``nuisance_decodability``).
3. **Residualisation** — linearly remove the nuisance subspace from the
   activations and re-run the probe (``residualize_activations``).

Nothing here invents metadata.  Every analysis reports ``status: not_run`` with a
reason when the columns it needs are absent, so a missing control is visible in
the artifact instead of quietly dropping out of the results.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder

from .paper import (PaperConfig, ResearchIntegrityError, compute_binary_metrics,
                    grouped_bootstrap_ci)

# Aggressive actions are the ones a bluff can hide inside: you cannot bluff by
# checking or folding.  Matching on action category is what separates "bluff-like
# deception" from "aggression".
DEFAULT_AGGRESSIVE_ACTIONS = ("bet", "raise", "reraise", "3bet", "4bet", "allin", "all-in", "shove")
# Columns treated as continuous and therefore binned before matching.
DEFAULT_CONTINUOUS_NUISANCE = ("hand_strength", "bet_size", "pot_size", "equity", "bet_to_pot")

MISSING = "__MISSING__"


def _normalize_action(value: Any) -> str:
    return str(value).strip().lower().replace(" ", "").replace("_", "")


def _is_continuous(df: pd.DataFrame, column: str) -> bool:
    if column in DEFAULT_CONTINUOUS_NUISANCE:
        return pd.api.types.is_numeric_dtype(df[column])
    return pd.api.types.is_numeric_dtype(df[column]) and df[column].nunique(dropna=True) > 10


# --------------------------------------------------------------------------- #
# Metadata availability
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ColumnAvailability:
    column: str
    present: bool
    coverage: float
    n_unique: int
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return {"column": self.column, "present": self.present, "coverage": round(self.coverage, 6),
                "n_unique": self.n_unique, "kind": self.kind}


# Which analyses each column makes possible.  Used to explain, in the artifact,
# exactly why a control could not be run.
ANALYSIS_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "bluff_vs_value": ("action",),
    "action_matched": ("action",),
    "street_matched": ("action", "street"),
    "equity_matched": ("action", "hand_strength"),
    "bet_size_matched": ("action", "bet_size", "pot_size"),
    "position_matched": ("action", "position"),
    "board_matched": ("action", "board_texture"),
}


def describe_metadata_availability(
    df: pd.DataFrame, columns: Iterable[str], *, min_coverage: float = 0.95
) -> dict[str, Any]:
    """Report which nuisance columns exist and which analyses they unlock.

    ``min_coverage`` is the non-missing fraction below which a column is treated
    as unusable; partial metadata silently shrinking an analysis is worse than
    the analysis being reported as unavailable.
    """
    availability: dict[str, ColumnAvailability] = {}
    for column in dict.fromkeys(columns):
        if column not in df.columns:
            availability[column] = ColumnAvailability(column, False, 0.0, 0, "absent")
            continue
        series = df[column]
        coverage = float(series.notna().mean())
        kind = "continuous" if _is_continuous(df, column) else "categorical"
        availability[column] = ColumnAvailability(
            column, True, coverage, int(series.nunique(dropna=True)), kind)

    usable = {c for c, a in availability.items() if a.present and a.coverage >= min_coverage}
    analyses: dict[str, Any] = {}
    for name, required in ANALYSIS_REQUIREMENTS.items():
        blocked = [c for c in required if c not in usable]
        analyses[name] = ({"available": True, "requires": list(required)} if not blocked
                          else {"available": False, "requires": list(required), "blocked_by": blocked})
    return {
        "min_coverage": min_coverage,
        "columns": {c: a.to_dict() for c, a in availability.items()},
        "usable_columns": sorted(usable),
        "analyses": analyses,
        "unavailable_analyses": sorted(n for n, v in analyses.items() if not v["available"]),
    }


def validate_nuisance_metadata(
    df: pd.DataFrame, columns: Iterable[str], *, min_coverage: float = 0.95
) -> dict[str, Any]:
    """Fail loudly when a *configured* nuisance column is absent or mostly empty.

    Configuring a column is a claim that the analysis will run.  If the data
    cannot support it, that is a data problem to fix, not a result to omit.
    """
    report = describe_metadata_availability(df, columns, min_coverage=min_coverage)
    problems: list[str] = []
    for column, info in report["columns"].items():
        if not info["present"]:
            problems.append(f"{column}: absent from dataset")
        elif info["coverage"] < min_coverage:
            problems.append(f"{column}: coverage {info['coverage']:.3f} below required {min_coverage}")
    if problems:
        raise ResearchIntegrityError(
            "Configured nuisance columns are unusable: " + "; ".join(problems)
        )
    return report


# --------------------------------------------------------------------------- #
# Matched subsets
# --------------------------------------------------------------------------- #

def bluff_vs_value_subset(
    df: pd.DataFrame, *, action_column: str = "action",
    aggressive_actions: Sequence[str] = DEFAULT_AGGRESSIVE_ACTIONS,
) -> pd.DataFrame:
    """Restrict to aggressive actions, where bluff and value are both possible.

    This is the comparison both reviewers implicitly asked for: bluff raise vs
    value raise, rather than deception vs passivity.
    """
    if action_column not in df.columns:
        raise ResearchIntegrityError(
            f"bluff_vs_value requires an {action_column!r} column; it is absent from the dataset"
        )
    allowed = {_normalize_action(a) for a in aggressive_actions}
    subset = df[df[action_column].map(_normalize_action).isin(allowed)]
    if subset.empty:
        raise ResearchIntegrityError(
            f"No rows have an aggressive {action_column!r} value; observed: "
            f"{sorted(df[action_column].dropna().astype(str).unique())[:10]}"
        )
    return subset.copy()


def _stratum_keys(
    df: pd.DataFrame, exact_columns: Sequence[str], binned_columns: Sequence[str], n_bins: int
) -> pd.Series:
    parts: list[pd.Series] = []
    for column in exact_columns:
        if column not in df.columns:
            raise ResearchIntegrityError(f"Matching column {column!r} is absent from the dataset")
        values = df[column].map(_normalize_action) if column == "action" else df[column].fillna(MISSING).astype(str)
        parts.append(column + "=" + values.astype(str))
    for column in binned_columns:
        if column not in df.columns:
            raise ResearchIntegrityError(f"Matching column {column!r} is absent from the dataset")
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.isna().all():
            raise ResearchIntegrityError(f"Binned matching column {column!r} has no numeric values")
        # Quantile bins adapt to the observed distribution; duplicate edges are
        # dropped so a degenerate column collapses to fewer bins instead of raising.
        codes = pd.qcut(numeric, q=min(n_bins, max(1, numeric.nunique())), duplicates="drop")
        parts.append(column + "=" + codes.astype(str))
    if not parts:
        raise ResearchIntegrityError("Matching requires at least one exact or binned column")
    key = parts[0]
    for part in parts[1:]:
        key = key + "|" + part
    return key


def build_matched_subset(
    df: pd.DataFrame,
    *,
    exact_columns: Sequence[str] = ("action",),
    binned_columns: Sequence[str] = (),
    n_bins: int = 4,
    seed: int = 2026,
    group_column: str = "split_group_id",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Keep equal numbers of each label inside every nuisance stratum.

    Within a stratum the two classes are downsampled to the smaller count, so the
    returned subset has matched nuisance distributions by construction.  Strata
    that contain only one class are dropped and counted in the report, because
    they carry no within-stratum contrast.

    Sampling is deterministic given ``seed``.  ``group_column`` is only used for
    reporting how many independent groups survive; matching is row-level because
    a bluff/value contrast normally lives *inside* a group.
    """
    if "label" not in df.columns:
        raise ResearchIntegrityError("Matched subsets require a label column")
    work = df.copy()
    work["_stratum"] = _stratum_keys(work, list(exact_columns), list(binned_columns), n_bins)
    rng = np.random.default_rng(seed)
    kept: list[pd.DataFrame] = []
    dropped_single_class = 0
    for stratum, block in work.groupby("_stratum", sort=True):
        counts = block.label.value_counts()
        if len(counts) < 2:
            dropped_single_class += 1
            continue
        take = int(counts.min())
        for label_value, rows in block.groupby("label", sort=True):
            if len(rows) == take:
                kept.append(rows)
            else:
                # Sort first so the sample depends only on the seed, not row order.
                ordered = rows.sort_values("sample_id")
                picks = rng.choice(len(ordered), size=take, replace=False)
                kept.append(ordered.iloc[np.sort(picks)])
    if not kept:
        raise ResearchIntegrityError(
            f"Matching on {list(exact_columns) + list(binned_columns)} left no stratum containing both "
            f"labels ({dropped_single_class} single-class strata dropped)"
        )
    matched = pd.concat(kept).sort_values("sample_id").drop(columns="_stratum").reset_index(drop=True)
    report = {
        "exact_columns": list(exact_columns), "binned_columns": list(binned_columns),
        "n_bins": n_bins, "seed": seed,
        "n_input_rows": int(len(df)), "n_matched_rows": int(len(matched)),
        "n_strata_used": int(work.loc[work.sample_id.isin(matched.sample_id), "_stratum"].nunique()),
        "n_strata_dropped_single_class": dropped_single_class,
        "label_balance": {str(k): int(v) for k, v in matched.label.value_counts().sort_index().items()},
        "n_groups": int(matched[group_column].nunique()) if group_column in matched.columns else None,
    }
    return matched, report


# --------------------------------------------------------------------------- #
# Nuisance decodability and residualisation
# --------------------------------------------------------------------------- #

def _one_hot(train: pd.DataFrame, other: pd.DataFrame, columns: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    fitted = encoder.fit_transform(train[list(columns)].fillna(MISSING).astype(str))
    return fitted, encoder.transform(other[list(columns)].fillna(MISSING).astype(str))


def _decode_categorical(
    df: pd.DataFrame, column: str, train_pos: np.ndarray, test_pos: np.ndarray,
    x_train: np.ndarray, x_test: np.ndarray, config: PaperConfig,
) -> dict[str, Any]:
    target = df[column].fillna(MISSING).astype(str)
    y_train, y_test = target.iloc[train_pos].to_numpy(), target.iloc[test_pos].to_numpy()
    classes = np.unique(y_train)
    if len(classes) < 2:
        return {"status": "not_run", "reason": "only one value present in the training split"}
    if len(classes) > len(y_train) / 2:
        return {"status": "not_run", "target_type": "categorical", "n_classes": int(len(classes)),
                "reason": f"{len(classes)} classes for {len(y_train)} training rows is not a "
                          "meaningful classification target; treat this column as continuous"}
    model = LogisticRegression(C=config.probe_c, class_weight="balanced", max_iter=5000,
                               random_state=config.seed)
    model.fit(x_train, y_train)
    majority = classes[int(np.argmax([np.sum(y_train == v) for v in classes]))]
    return {
        "status": "ok", "target_type": "categorical", "n_classes": int(len(classes)),
        "accuracy": float(np.mean(model.predict(x_test) == y_test)),
        "majority_baseline_accuracy": float(np.mean(y_test == majority)),
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
    }


def _decode_continuous(
    df: pd.DataFrame, column: str, train_pos: np.ndarray, test_pos: np.ndarray,
    x_train: np.ndarray, x_test: np.ndarray, config: PaperConfig,
) -> dict[str, Any]:
    """Regress a continuous nuisance variable; R^2 against a train-mean predictor.

    Stringifying a continuous variable into hundreds of classes would report a
    meaningless accuracy, so continuous columns get a regression instead.
    """
    numeric = pd.to_numeric(df[column], errors="coerce")
    y_train = numeric.iloc[train_pos].to_numpy(dtype=float)
    y_test = numeric.iloc[test_pos].to_numpy(dtype=float)
    finite_train, finite_test = np.isfinite(y_train), np.isfinite(y_test)
    if finite_train.sum() < 4 or finite_test.sum() < 2:
        return {"status": "not_run", "target_type": "continuous",
                "reason": "too few finite numeric values to regress"}
    if np.std(y_train[finite_train]) == 0:
        return {"status": "not_run", "target_type": "continuous",
                "reason": "constant in the training split"}
    model = Ridge(alpha=1.0, random_state=config.seed)
    model.fit(x_train[finite_train], y_train[finite_train])
    predicted = model.predict(x_test[finite_test])
    truth = y_test[finite_test]
    residual = float(np.sum((truth - predicted) ** 2))
    # Baseline is the training mean, so R^2 can go negative and that is informative.
    baseline = float(np.sum((truth - np.mean(y_train[finite_train])) ** 2))
    return {
        "status": "ok", "target_type": "continuous",
        "r2_vs_train_mean": float(1.0 - residual / baseline) if baseline > 0 else None,
        "rmse": float(np.sqrt(residual / len(truth))),
        "baseline_rmse": float(np.sqrt(baseline / len(truth))),
        "n_train": int(finite_train.sum()), "n_test": int(finite_test.sum()),
    }


def nuisance_decodability(
    activations: np.ndarray, df: pd.DataFrame, manifest: dict[str, Any],
    columns: Sequence[str], config: PaperConfig, *, layer: int | None = None,
) -> dict[str, Any]:
    """How well do the same hidden states predict each nuisance variable?

    A probe that reads bluff labels at AUROC 0.9 while also reading ``action`` at
    accuracy 0.95 is plausibly decoding strategy.  Reporting both is what lets a
    reader judge that, so this is a diagnostic to publish rather than a number to
    beat.  Each column is scored against its own majority-class baseline.
    """
    index = {sid: i for i, sid in enumerate(df.sample_id)}
    train_pos = np.array([index[s] for s in manifest["train"]])
    test_pos = np.array([index[s] for s in manifest["test"]])
    chosen = int(layer if layer is not None else activations.shape[1] - 1)
    x_train, x_test = activations[train_pos, chosen], activations[test_pos, chosen]
    results: dict[str, Any] = {}
    for column in columns:
        if column not in df.columns:
            results[column] = {"status": "not_run", "reason": "column absent from dataset"}
            continue
        if _is_continuous(df, column):
            results[column] = _decode_continuous(df, column, train_pos, test_pos, x_train, x_test, config)
        else:
            results[column] = _decode_categorical(df, column, train_pos, test_pos, x_train, x_test, config)
        results[column]["layer"] = chosen
    return results


def residualize_activations(
    train: np.ndarray, evaluate: np.ndarray, nuisance_train: np.ndarray, nuisance_eval: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly remove the nuisance subspace, fitting the removal on train only.

    Solves ``train ~ nuisance_train`` by least squares and subtracts the fitted
    component from both partitions using the *training* coefficients, so no
    evaluation information leaks into the control.
    """
    if train.ndim != 2 or evaluate.ndim != 2:
        raise ResearchIntegrityError("Residualisation expects 2-D [samples, features] activations")
    design = np.hstack([nuisance_train, np.ones((len(nuisance_train), 1))])
    design_eval = np.hstack([nuisance_eval, np.ones((len(nuisance_eval), 1))])
    coefficients, *_ = np.linalg.lstsq(design, train, rcond=None)
    return train - design @ coefficients, evaluate - design_eval @ coefficients


def controlled_probe_analysis(
    activations: np.ndarray, df: pd.DataFrame, manifest: dict[str, Any],
    config: PaperConfig, columns: Sequence[str], *, layer: int,
) -> dict[str, Any]:
    """Probe performance before and after removing the nuisance subspace.

    Both numbers are reported.  A large drop means the apparent bluff signal was
    substantially strategy information; a small drop means it was not.  Neither
    outcome is treated as the expected one.
    """
    usable = [c for c in columns if c in df.columns]
    index = {sid: i for i, sid in enumerate(df.sample_id)}
    train_pos = np.array([index[s] for s in manifest["train"]])
    test_pos = np.array([index[s] for s in manifest["test"]])
    labels = df.label.to_numpy(int)
    y_train, y_test = labels[train_pos], labels[test_pos]
    x_train, x_test = activations[train_pos, layer], activations[test_pos, layer]

    def fit_and_score(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
        model = LogisticRegression(C=config.probe_c, class_weight="balanced", max_iter=5000,
                                   random_state=config.seed)
        model.fit(a, y_train)
        return compute_binary_metrics(y_test, model.predict_proba(b)[:, 1])

    result: dict[str, Any] = {"layer": int(layer), "unadjusted": fit_and_score(x_train, x_test)}
    if not usable:
        result["residualized"] = {"status": "not_run", "reason": "no configured nuisance columns are present"}
        return result
    train_rows, test_rows = df.iloc[train_pos], df.iloc[test_pos]
    nuisance_train, nuisance_test = _one_hot(train_rows, test_rows, usable)
    r_train, r_test = residualize_activations(x_train, x_test, nuisance_train, nuisance_test)
    result["residualized"] = {"columns": usable, **fit_and_score(r_train, r_test)}
    result["auroc_drop_after_control"] = round(
        result["unadjusted"]["auroc"] - result["residualized"]["auroc"], 6)
    return result


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def _probe_on_subset(
    activations: np.ndarray, df: pd.DataFrame, manifest: dict[str, Any],
    subset_ids: Sequence[Any], config: PaperConfig, *, layer: int,
) -> dict[str, Any]:
    """Refit on a subset, keeping the original partition assignment intact."""
    index = {sid: i for i, sid in enumerate(df.sample_id)}
    keep = set(map(str, subset_ids))
    part = {name: [s for s in manifest[name] if str(s) in keep] for name in ("train", "test")}
    labels = df.label.to_numpy(int)
    for name, ids in part.items():
        if len(ids) < 4:
            return {"status": "not_run", "reason": f"only {len(ids)} {name} rows survive matching"}
        if len(np.unique(labels[[index[s] for s in ids]])) < 2:
            return {"status": "not_run", "reason": f"{name} partition lost a label class after matching"}
    train_pos = np.array([index[s] for s in part["train"]])
    test_pos = np.array([index[s] for s in part["test"]])
    model = LogisticRegression(C=config.probe_c, class_weight="balanced", max_iter=5000,
                               random_state=config.seed)
    model.fit(activations[train_pos, layer], labels[train_pos])
    probabilities = model.predict_proba(activations[test_pos, layer])[:, 1]
    y_test = labels[test_pos]
    group_col = manifest.get("group_column", "base_item_id")
    out: dict[str, Any] = {"status": "ok", "layer": int(layer),
                           "n_train": int(len(train_pos)), "n_test": int(len(test_pos)),
                           **compute_binary_metrics(y_test, probabilities)}
    if group_col in df.columns:
        out["test_auroc_grouped_ci"] = grouped_bootstrap_ci(
            y_test, probabilities, df.iloc[test_pos][group_col].to_numpy(),
            seed=config.seed, n_resamples=config.bootstrap_resamples, metric="auroc")
    return out


def run_confound_suite(
    activations: np.ndarray, df: pd.DataFrame, manifest: dict[str, Any],
    config: PaperConfig, *, layer: int,
) -> dict[str, Any]:
    """Run every confound control the available metadata supports.

    ``layer`` must have been selected on validation data before this is called;
    this suite never selects a layer, so it cannot leak test information into the
    choice.
    """
    columns = list(config.nuisance_columns)
    availability = describe_metadata_availability(df, columns)
    result: dict[str, Any] = {
        "selected_layer": int(layer),
        "layer_selected_on": "validation (upstream)",
        "metadata_availability": availability,
        "nuisance_decodability": nuisance_decodability(activations, df, manifest, columns, config, layer=layer),
        "controlled_probe": controlled_probe_analysis(activations, df, manifest, config, columns, layer=layer),
        "subsets": {},
    }

    def attempt(name: str, build) -> None:
        required = ANALYSIS_REQUIREMENTS.get(name, ())
        blocked = [c for c in required if c not in availability["usable_columns"]]
        if blocked:
            result["subsets"][name] = {"status": "not_run",
                                       "reason": f"requires usable {blocked}", "requires": list(required)}
            return
        try:
            subset, report = build()
        except ResearchIntegrityError as exc:
            result["subsets"][name] = {"status": "not_run", "reason": str(exc), "requires": list(required)}
            return
        result["subsets"][name] = {
            "matching": report,
            "probe": _probe_on_subset(activations, df, manifest, subset.sample_id.tolist(), config, layer=layer),
        }

    def _bluff_vs_value():
        subset = bluff_vs_value_subset(df)
        return subset, {
            "description": "aggressive actions only; bluff raise vs value raise",
            "n_rows": int(len(subset)),
            "label_balance": {str(k): int(v) for k, v in subset.label.value_counts().sort_index().items()},
        }

    attempt("bluff_vs_value", _bluff_vs_value)
    attempt("action_matched", lambda: build_matched_subset(
        df, exact_columns=("action",), seed=config.seed, group_column=manifest.get("group_column", "base_item_id")))
    attempt("street_matched", lambda: build_matched_subset(
        bluff_vs_value_subset(df), exact_columns=("action", "street"), seed=config.seed,
        group_column=manifest.get("group_column", "base_item_id")))
    attempt("equity_matched", lambda: build_matched_subset(
        bluff_vs_value_subset(df), exact_columns=("action",), binned_columns=("hand_strength",),
        seed=config.seed, group_column=manifest.get("group_column", "base_item_id")))
    attempt("position_matched", lambda: build_matched_subset(
        bluff_vs_value_subset(df), exact_columns=("action", "position"), seed=config.seed,
        group_column=manifest.get("group_column", "base_item_id")))
    attempt("bet_size_matched", lambda: build_matched_subset(
        bluff_vs_value_subset(df), exact_columns=("action",), binned_columns=("bet_size", "pot_size"),
        seed=config.seed, group_column=manifest.get("group_column", "base_item_id")))
    attempt("board_matched", lambda: build_matched_subset(
        bluff_vs_value_subset(df), exact_columns=("action", "board_texture"), seed=config.seed,
        group_column=manifest.get("group_column", "base_item_id")))
    result["unavailable_analyses"] = sorted(
        name for name, v in result["subsets"].items() if v.get("status") == "not_run")
    return result


__all__ = [
    "ANALYSIS_REQUIREMENTS", "DEFAULT_AGGRESSIVE_ACTIONS", "ColumnAvailability",
    "bluff_vs_value_subset", "build_matched_subset", "controlled_probe_analysis",
    "describe_metadata_availability", "nuisance_decodability",
    "residualize_activations", "run_confound_suite", "validate_nuisance_metadata",
]
