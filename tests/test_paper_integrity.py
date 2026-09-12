from pathlib import Path
import sys

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deception_circuits.paper import (PaperConfig, ResearchIntegrityError,
                                      load_activations, make_split_manifest,
                                      run_probe_experiment, validate_dataset)


def _fixture(tmp_path: Path):
    activations = tmp_path / "activations"
    activations.mkdir()
    rows = []
    for group in range(20):
        for label in (0, 1):
            sid = f"{group}-{label}"
            rows.append({"sample_id": sid, "base_item_id": group,
                         "split_group_id": group, "statement": f"prompt {group}",
                         "response": "response", "label": label, "scenario": "poker"})
            tensor = torch.randn(3, 6)
            tensor[:, 0] += label
            torch.save(tensor, activations / f"sample_{sid}.pt")
    csv = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv, activations


def test_paper_probe_is_group_safe_and_validation_selected(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = PaperConfig("test", str(csv), str(activation_dir), str(tmp_path / "out"), n_seeds=2)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    split_by_id = {sid: name for name in ("train", "validation", "test") for sid in manifest[name]}
    assert df.assign(split=df.sample_id.map(split_by_id)).groupby("split_group_id").split.nunique().max() == 1
    result = run_probe_experiment(df, load_activations(df, activation_dir), manifest, config)
    assert result["selection_partition"] == "validation"
    assert result["test_partition_used_for_selection"] is False
    assert len(result["runs"]) == 2


def test_missing_activation_is_a_hard_error(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    (activation_dir / "sample_0-0.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Missing activation"):
        load_activations(validate_dataset(csv), activation_dir)


def test_duplicate_sample_id_is_rejected(tmp_path):
    csv, _ = _fixture(tmp_path)
    df = pd.read_csv(csv)
    df.loc[1, "sample_id"] = df.loc[0, "sample_id"]
    df.to_csv(csv, index=False)
    with pytest.raises(ResearchIntegrityError, match="unique"):
        validate_dataset(csv)
