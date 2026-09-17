"""Strict cross-context transfer analysis for the paper pipeline.

Layer selection is performed using the source context's training and validation
rows only. The target context is used only for held-out evaluation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .paper import (
    PaperConfig,
    ResearchIntegrityError,
    compute_binary_metrics,
    grouped_bootstrap_ci,
)


def available_contexts(
    df: pd.DataFrame,
    *,
    context_column: str = "scenario",
) -> list[str]:
    """Return sorted usable context labels."""
    if context_column not in df.columns:
        return []

    values = (
        df[context_column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    return sorted(value for value in values.unique() if value)


def run_cross_context_transfer(
    activations: np.ndarray,
    df: pd.DataFrame,
    manifest: dict[str, Any],
    config: PaperConfig,
    *,
    layer_indices: list[int] | None = None,
    context_column: str = "scenario",
) -> dict[str, Any]:
    """Train/select on one context and evaluate on another held-out context.

    For each source context:
      1. fit one probe per activation tensor axis on source-train rows;
      2. select the axis using source-validation AUROC only;
      3. freeze that probe and layer;
      4. evaluate on target-context rows from the global test partition only.

    No target-context test label is used for layer or model selection.
    """
    if activations.ndim != 3:
        raise ResearchIntegrityError(
            "Cross-context transfer requires activations shaped "
            "[samples, layers, hidden]."
        )

    if len(df) != activations.shape[0]:
        raise ResearchIntegrityError(
            "Activation sample count does not match the dataset."
        )

    if layer_indices is None:
        layer_indices = list(range(activations.shape[1]))

    layer_indices = [int(layer) for layer in layer_indices]

    if len(layer_indices) != activations.shape[1]:
        raise ResearchIntegrityError(
            "Physical layer mapping length must match the activation tensor "
            "layer dimension."
        )

    if len(set(layer_indices)) != len(layer_indices) or any(
        layer < 0 for layer in layer_indices
    ):
        raise ResearchIntegrityError(
            "Physical layer mapping must contain unique, non-negative "
            "transformer layers."
        )

    contexts = available_contexts(
        df,
        context_column=context_column,
    )

    if len(contexts) < 2:
        return {
            "status": "not_run",
            "reason": (
                "cross-context transfer requires at least two usable contexts "
                f"in column {context_column!r}; found {contexts}"
            ),
            "context_column": context_column,
            "contexts": contexts,
            "selection_partition": "source validation only",
            "target_test_used_for_selection": False,
            "cells": [],
        }

    required_parts = {"train", "validation", "test"}
    if not required_parts.issubset(manifest):
        raise ResearchIntegrityError(
            "Cross-context transfer requires train, validation, and test "
            "partitions in the split manifest."
        )

    sample_to_position = {
        sample_id: position
        for position, sample_id in enumerate(df["sample_id"])
    }

    labels = df["label"].to_numpy(dtype=int)
    context_values = df[context_column].astype(str).to_numpy()
    group_column = manifest.get("group_column", "base_item_id")

    if group_column not in df.columns:
        raise ResearchIntegrityError(
            f"Transfer group column {group_column!r} is missing from the dataset."
        )

    partition_positions: dict[str, np.ndarray] = {}
    for partition in ("train", "validation", "test"):
        try:
            partition_positions[partition] = np.array(
                [sample_to_position[sid] for sid in manifest[partition]],
                dtype=int,
            )
        except KeyError as exc:
            raise ResearchIntegrityError(
                f"Split manifest references unknown sample_id {exc.args[0]!r}."
            ) from exc

    cells: list[dict[str, Any]] = []

    for source_context in contexts:
        source_train = partition_positions["train"][
            context_values[partition_positions["train"]] == source_context
        ]
        source_validation = partition_positions["validation"][
            context_values[partition_positions["validation"]] == source_context
        ]

        if (
            len(source_train) == 0
            or len(source_validation) == 0
            or len(np.unique(labels[source_train])) < 2
            or len(np.unique(labels[source_validation])) < 2
        ):
            for target_context in contexts:
                if target_context == source_context:
                    continue
                cells.append({
                    "source_context": source_context,
                    "target_context": target_context,
                    "status": "not_run",
                    "reason": (
                        "source context does not contain both label classes in "
                        "both train and validation partitions"
                    ),
                })
            continue

        validation_scores: list[float] = []
        models: list[LogisticRegression] = []

        for layer_index in range(activations.shape[1]):
            model = LogisticRegression(
                C=config.probe_c,
                class_weight="balanced",
                max_iter=5000,
                random_state=config.seed,
            )
            model.fit(
                activations[source_train, layer_index],
                labels[source_train],
            )

            validation_probability = model.predict_proba(
                activations[source_validation, layer_index]
            )[:, 1]

            validation_metrics = compute_binary_metrics(
                labels[source_validation],
                validation_probability,
            )
            validation_scores.append(
                float(validation_metrics["auroc"])
            )
            models.append(model)

        selected_layer_index = int(np.argmax(validation_scores))
        selected_physical_layer = layer_indices[selected_layer_index]
        selected_model = models[selected_layer_index]

        for target_context in contexts:
            if target_context == source_context:
                continue

            target_test = partition_positions["test"][
                context_values[partition_positions["test"]] == target_context
            ]

            if (
                len(target_test) == 0
                or len(np.unique(labels[target_test])) < 2
            ):
                cells.append({
                    "source_context": source_context,
                    "target_context": target_context,
                    "status": "not_run",
                    "reason": (
                        "target context does not contain both label classes "
                        "in the held-out test partition"
                    ),
                    "selected_layer_index": selected_layer_index,
                    "selected_physical_layer": selected_physical_layer,
                })
                continue

            target_probability = selected_model.predict_proba(
                activations[target_test, selected_layer_index]
            )[:, 1]

            target_labels = labels[target_test]
            target_groups = df.iloc[target_test][group_column].to_numpy()

            cells.append({
                "source_context": source_context,
                "target_context": target_context,
                "status": "ok",
                "selected_layer_index": selected_layer_index,
                "selected_physical_layer": selected_physical_layer,
                "source_validation_auroc": validation_scores[
                    selected_layer_index
                ],
                "source_validation_auroc_by_layer": [
                    float(value) for value in validation_scores
                ],
                "n_source_train": int(len(source_train)),
                "n_source_validation": int(len(source_validation)),
                "n_target_test": int(len(target_test)),
                "target_test": compute_binary_metrics(
                    target_labels,
                    target_probability,
                ),
                "target_test_auroc_grouped_ci": grouped_bootstrap_ci(
                    target_labels,
                    target_probability,
                    target_groups,
                    seed=config.seed,
                    n_resamples=config.bootstrap_resamples,
                    metric="auroc",
                ),
            })

    return {
        "status": "ok",
        "context_column": context_column,
        "contexts": contexts,
        "activation_layer_indices": layer_indices,
        "selection_partition": "source validation only",
        "target_test_used_for_selection": False,
        "cells": cells,
    }
def audit_cross_context_transfer(
    result: dict[str, Any],
) -> list[str]:
    """Return integrity failures for a cross-context transfer artifact."""
    failures: list[str] = []

    status = result.get("status")

    if status == "not_run":
        if not result.get("reason"):
            failures.append(
                "cross-context transfer is not_run but records no reason"
            )
        if result.get("target_test_used_for_selection") is not False:
            failures.append(
                "cross-context transfer does not explicitly assert that "
                "target test data was excluded from selection"
            )
        return failures

    if status != "ok":
        return [
            f"cross-context transfer has unknown status {status!r}"
        ]

    if result.get("selection_partition") != "source validation only":
        failures.append(
            "cross-context layer selection is not source-validation-only"
        )

    if result.get("target_test_used_for_selection") is not False:
        failures.append(
            "cross-context transfer used or may have used target test data "
            "for selection"
        )

    contexts = [
        str(value)
        for value in (result.get("contexts") or [])
    ]

    if len(contexts) < 2:
        failures.append(
            "completed cross-context transfer records fewer than two contexts"
        )

    if len(set(contexts)) != len(contexts):
        failures.append(
            "cross-context transfer context list contains duplicates"
        )

    raw_layer_indices = result.get("activation_layer_indices") or []

    try:
        layer_indices = [
            int(value)
            for value in raw_layer_indices
        ]
    except (TypeError, ValueError):
        layer_indices = []
        failures.append(
            "cross-context activation layer mapping is not integer-valued"
        )

    if not layer_indices:
        failures.append(
            "cross-context transfer records no activation layer mapping"
        )
    elif (
        len(set(layer_indices)) != len(layer_indices)
        or any(value < 0 for value in layer_indices)
    ):
        failures.append(
            "cross-context activation layer mapping must contain unique, "
            "non-negative physical layers"
        )

    expected_pairs = {
        (source, target)
        for source in contexts
        for target in contexts
        if source != target
    }

    seen_pairs: set[tuple[str, str]] = set()

    for cell in result.get("cells") or []:
        source = str(cell.get("source_context"))
        target = str(cell.get("target_context"))
        pair = (source, target)

        if source not in contexts or target not in contexts:
            failures.append(
                f"cross-context cell {pair} references an unknown context"
            )
            continue

        if source == target:
            failures.append(
                f"cross-context cell {pair} is a within-context cell"
            )
            continue

        if pair in seen_pairs:
            failures.append(
                f"cross-context cell {pair} is duplicated"
            )
            continue

        seen_pairs.add(pair)

        cell_status = cell.get("status")

        if cell_status == "not_run":
            if not cell.get("reason"):
                failures.append(
                    f"cross-context cell {pair} is not_run but records no reason"
                )
            continue

        if cell_status != "ok":
            failures.append(
                f"cross-context cell {pair} has unknown status "
                f"{cell_status!r}"
            )
            continue

        try:
            selected_layer_index = int(
                cell["selected_layer_index"]
            )
            selected_physical_layer = int(
                cell["selected_physical_layer"]
            )
        except (KeyError, TypeError, ValueError):
            failures.append(
                f"cross-context cell {pair} has invalid layer provenance"
            )
            continue

        if not 0 <= selected_layer_index < len(layer_indices):
            failures.append(
                f"cross-context cell {pair} selected activation axis "
                f"{selected_layer_index} outside the recorded layer mapping"
            )
        elif (
            layer_indices[selected_layer_index]
            != selected_physical_layer
        ):
            failures.append(
                f"cross-context cell {pair} maps activation axis "
                f"{selected_layer_index} to physical layer "
                f"{selected_physical_layer}, expected "
                f"{layer_indices[selected_layer_index]}"
            )

        validation_curve = (
            cell.get("source_validation_auroc_by_layer") or []
        )

        if len(validation_curve) != len(layer_indices):
            failures.append(
                f"cross-context cell {pair} validation curve length does "
                "not match the activation layer mapping"
            )

        if "source_validation_auroc" not in cell:
            failures.append(
                f"cross-context cell {pair} is missing source validation AUROC"
            )

        target_test = cell.get("target_test") or {}

        missing_metrics = {
            "auroc",
            "pr_auc",
            "ece",
        } - set(target_test)

        if missing_metrics:
            failures.append(
                f"cross-context cell {pair} target-test metrics are missing "
                f"{sorted(missing_metrics)}"
            )

        interval = (
            cell.get("target_test_auroc_grouped_ci") or {}
        )

        if "lower" not in interval or "upper" not in interval:
            failures.append(
                f"cross-context cell {pair} has no grouped AUROC confidence "
                "interval"
            )

    missing_pairs = sorted(expected_pairs - seen_pairs)
    extra_pairs = sorted(seen_pairs - expected_pairs)

    if missing_pairs:
        failures.append(
            f"cross-context transfer is missing source-target cells: "
            f"{missing_pairs}"
        )

    if extra_pairs:
        failures.append(
            f"cross-context transfer has unexpected source-target cells: "
            f"{extra_pairs}"
        )

    return failures