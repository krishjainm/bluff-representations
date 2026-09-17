from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deception_circuits.paper import PaperConfig, make_split_manifest
from deception_circuits.paper_transfer import (
    audit_cross_context_transfer,
    run_cross_context_transfer,
)

def _config(csv: Path, tmp_path: Path) -> PaperConfig:
    return PaperConfig(
        experiment_name="transfer-test",
        dataset_path=str(csv),
        activation_dir=str(tmp_path / "activations"),
        output_dir=str(tmp_path / "out"),
        subject_model="stub/model",
        n_seeds=1,
        bootstrap_resamples=20,
    )


def _two_context_dataset(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    rows = []

    sample = 0
    for context in ("poker", "negotiation"):
        for group in range(30):
            for label in (0, 1):
                rows.append({
                    "sample_id": f"s{sample}",
                    "base_item_id": f"{context}-{group}",
                    "split_group_id": f"{context}-{group}",
                    "statement": f"{context} example {group} label {label}",
                    "response": "x",
                    "label": label,
                    "scenario": context,
                })
                sample += 1

    df = pd.DataFrame(rows)
    csv = tmp_path / "dataset.csv"
    df.to_csv(csv, index=False)
    return csv, df


def test_single_context_returns_not_run(tmp_path):
    rows = []
    for group in range(20):
        for label in (0, 1):
            rows.append({
                "sample_id": f"{group}-{label}",
                "base_item_id": group,
                "split_group_id": group,
                "statement": f"hand {group}",
                "response": "x",
                "label": label,
                "scenario": "poker",
            })

    df = pd.DataFrame(rows)
    csv = tmp_path / "dataset.csv"
    df.to_csv(csv, index=False)

    config = _config(csv, tmp_path)
    manifest = make_split_manifest(df, config)

    activations = np.zeros(
        (len(df), 2, 3),
        dtype=np.float32,
    )

    result = run_cross_context_transfer(
        activations,
        df,
        manifest,
        config,
        layer_indices=[1, 3],
    )

    assert result["status"] == "not_run"
    assert result["contexts"] == ["poker"]
    assert result["target_test_used_for_selection"] is False


def test_transfer_preserves_physical_layer_mapping(tmp_path):
    csv, df = _two_context_dataset(tmp_path)
    config = _config(csv, tmp_path)
    manifest = make_split_manifest(df, config)

    activations = np.zeros(
        (len(df), 2, 3),
        dtype=np.float32,
    )

    labels = df["label"].to_numpy(dtype=np.float32)

    # Tensor axis 1 carries the predictive signal, but represents
    # physical transformer layer 8.
    activations[:, 1, 0] = labels

    result = run_cross_context_transfer(
        activations,
        df,
        manifest,
        config,
        layer_indices=[2, 8],
    )

    assert result["status"] == "ok"
    assert result["activation_layer_indices"] == [2, 8]

    ok_cells = [
        cell for cell in result["cells"]
        if cell["status"] == "ok"
    ]

    assert ok_cells

    for cell in ok_cells:
        assert cell["selected_layer_index"] == 1
        assert cell["selected_physical_layer"] == 8


def test_target_test_labels_do_not_change_source_layer_selection(tmp_path):
    csv, df = _two_context_dataset(tmp_path)
    config = _config(csv, tmp_path)
    manifest = make_split_manifest(df, config)

    activations = np.zeros(
        (len(df), 2, 3),
        dtype=np.float32,
    )

    labels = df["label"].to_numpy(dtype=np.float32)

    poker_mask = df["scenario"].to_numpy() == "poker"
    negotiation_mask = df["scenario"].to_numpy() == "negotiation"

    # Poker source signal lives on tensor axis 0.
    activations[poker_mask, 0, 0] = labels[poker_mask]

    # Negotiation source signal lives on tensor axis 1.
    activations[negotiation_mask, 1, 0] = labels[negotiation_mask]

    first = run_cross_context_transfer(
        activations,
        df,
        manifest,
        config,
        layer_indices=[4, 9],
    )

    changed = df.copy()

    test_ids = set(manifest["test"])
    negotiation_test = (
        changed["sample_id"].isin(test_ids)
        & (changed["scenario"] == "negotiation")
    )

    changed.loc[negotiation_test, "label"] = (
        1 - changed.loc[negotiation_test, "label"].astype(int)
    )

    second = run_cross_context_transfer(
        activations,
        changed,
        manifest,
        config,
        layer_indices=[4, 9],
    )

    def source_selection(result, source):
        cells = [
            cell
            for cell in result["cells"]
            if (
                cell["source_context"] == source
                and cell["status"] == "ok"
            )
        ]

        assert cells

        return {
            (
                cell["selected_layer_index"],
                cell["selected_physical_layer"],
                cell["source_validation_auroc"],
            )
            for cell in cells
        }

    assert source_selection(first, "poker") == source_selection(
        second,
        "poker",
    )

    assert first["target_test_used_for_selection"] is False
    assert second["target_test_used_for_selection"] is False

def test_transfer_audit_accepts_complete_result(tmp_path):
    csv, df = _two_context_dataset(tmp_path)
    config = _config(csv, tmp_path)
    manifest = make_split_manifest(df, config)

    activations = np.zeros(
        (len(df), 2, 3),
        dtype=np.float32,
    )
    activations[:, 1, 0] = df["label"].to_numpy(dtype=np.float32)

    result = run_cross_context_transfer(
        activations,
        df,
        manifest,
        config,
        layer_indices=[2, 8],
    )

    assert audit_cross_context_transfer(result) == []


def test_transfer_audit_rejects_missing_source_target_cell(tmp_path):
    csv, df = _two_context_dataset(tmp_path)
    config = _config(csv, tmp_path)
    manifest = make_split_manifest(df, config)

    activations = np.zeros(
        (len(df), 2, 3),
        dtype=np.float32,
    )
    activations[:, 1, 0] = df["label"].to_numpy(dtype=np.float32)

    result = run_cross_context_transfer(
        activations,
        df,
        manifest,
        config,
        layer_indices=[2, 8],
    )

    result["cells"] = result["cells"][:-1]

    failures = audit_cross_context_transfer(result)

    assert any(
        "missing source-target cells" in failure
        for failure in failures
    )


def test_transfer_audit_rejects_wrong_physical_layer_mapping(tmp_path):
    csv, df = _two_context_dataset(tmp_path)
    config = _config(csv, tmp_path)
    manifest = make_split_manifest(df, config)

    activations = np.zeros(
        (len(df), 2, 3),
        dtype=np.float32,
    )
    activations[:, 1, 0] = df["label"].to_numpy(dtype=np.float32)

    result = run_cross_context_transfer(
        activations,
        df,
        manifest,
        config,
        layer_indices=[2, 8],
    )

    ok_cell = next(
        cell
        for cell in result["cells"]
        if cell["status"] == "ok"
    )
    ok_cell["selected_physical_layer"] = 999

    failures = audit_cross_context_transfer(result)

    assert any(
        "expected 8" in failure
        for failure in failures
    )