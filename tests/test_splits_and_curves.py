"""Tests for P2 split stratification, calibration, and learning curves."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _stubs import StubHiddenStateProvider

from deception_circuits.paper import (PaperConfig, ResearchIntegrityError,
                                      assert_partition_label_availability, assert_split_integrity,
                                      load_activations, load_split_manifest, make_split_manifest,
                                      run_probe_experiment, save_manifest,
                                      summarize_across_seeds, summarize_partitions,
                                      validate_dataset)
from deception_circuits.paper_extraction import (ExtractionSpec,
                                                 load_activation_layer_indices,
                                                 run_extraction)
STUB_MODEL = "stub/tiny-test-model"


def _dataset(tmp_path: Path, n_groups: int = 30, *, imbalance: float | None = None) -> Path:
    rows = []
    for group in range(n_groups):
        # With imbalance set, only that fraction of groups carries a positive row.
        labels = (0, 1) if imbalance is None or group < int(n_groups * imbalance) else (0,)
        for label in labels:
            rows.append({"sample_id": f"{group}-{label}", "base_item_id": group,
                         "split_group_id": group, "statement": f"hand {group} variant {label}",
                         "response": "raise", "label": label, "scenario": "poker",
                         "difficulty_bucket": "hard" if group % 2 else "easy"})
    csv = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _fixture(tmp_path: Path, **kwargs):
    csv = _dataset(tmp_path, **kwargs)
    df = pd.read_csv(csv)
    activations = tmp_path / "activations"
    run_extraction(df, ExtractionSpec(subject_model=STUB_MODEL, prompt_template_id="poker_action_v1"),
                   activations, StubHiddenStateProvider())
    return csv, activations


def _config(csv: Path, activations: Path, tmp_path: Path, **overrides) -> PaperConfig:
    base = dict(experiment_name="splits", dataset_path=str(csv), activation_dir=str(activations),
                output_dir=str(tmp_path / "out"), subject_model=STUB_MODEL,
                prompt_template_id="poker_action_v1", n_seeds=2, bootstrap_resamples=30)
    base.update(overrides)
    return PaperConfig(**base)


# --- stratified splits ----------------------------------------------------------

def test_stratified_split_is_the_default_and_is_recorded(tmp_path):
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, _config(csv, tmp_path / "a", tmp_path))
    assert manifest["split_strategy"] == "grouped_stratified"
    assert manifest["stratify_columns"] == ["label"]
    assert manifest["schema_version"] == 2


def test_stratified_split_keeps_groups_intact(tmp_path):
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, _config(csv, tmp_path / "a", tmp_path))
    by_id = {sid: name for name in ("train", "validation", "test") for sid in manifest[name]}
    assert df.assign(p=df.sample_id.map(by_id)).groupby("split_group_id").p.nunique().max() == 1


def test_stratified_split_balances_labels_across_partitions(tmp_path):
    """The stratified strategy should hold the positive rate roughly constant."""
    csv = _dataset(tmp_path, n_groups=40, imbalance=0.5)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, _config(csv, tmp_path / "a", tmp_path))
    rates = [manifest["partition_summary"][p]["positive_rate"]
             for p in ("train", "validation", "test")]
    assert max(rates) - min(rates) < 0.20, rates


def test_split_is_deterministic_for_a_given_seed(tmp_path):
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    config = _config(csv, tmp_path / "a", tmp_path)
    assert make_split_manifest(df, config)["test"] == make_split_manifest(df, config)["test"]
    other = make_split_manifest(df, _config(csv, tmp_path / "a", tmp_path, seed=99))
    assert other["test"] != make_split_manifest(df, config)["test"]

def test_load_split_manifest_accepts_the_dataset_it_was_created_from(tmp_path):
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    config = _config(csv, tmp_path / "a", tmp_path)

    manifest = make_split_manifest(df, config)
    path = tmp_path / "split_manifest.json"
    save_manifest(manifest, path)

    loaded = load_split_manifest(path, df, config)

    assert loaded["dataset_sha256"] == manifest["dataset_sha256"]
    assert loaded["train"] == manifest["train"]
    assert loaded["validation"] == manifest["validation"]
    assert loaded["test"] == manifest["test"]


def test_load_split_manifest_rejects_dataset_changed_after_split(tmp_path):
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    config = _config(csv, tmp_path / "a", tmp_path)

    manifest = make_split_manifest(df, config)
    path = tmp_path / "split_manifest.json"
    save_manifest(manifest, path)

    changed = pd.read_csv(csv)
    changed.loc[0, "statement"] = (
        str(changed.loc[0, "statement"]) + " CHANGED_AFTER_SPLIT"
    )
    changed.to_csv(csv, index=False)
    changed_df = validate_dataset(csv)

    with pytest.raises(
        ResearchIntegrityError,
        match="dataset SHA-256 does not match",
    ):
        load_split_manifest(path, changed_df, config)

def test_load_split_manifest_rejects_changed_split_seed(tmp_path):
    """A frozen split must not be reused under a different configured seed."""
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)

    original_config = _config(
        csv,
        tmp_path / "a",
        tmp_path,
        seed=2026,
    )
    manifest = make_split_manifest(df, original_config)

    path = tmp_path / "split_manifest.json"
    save_manifest(manifest, path)

    changed_config = _config(
        csv,
        tmp_path / "a",
        tmp_path,
        seed=9999,
    )

    with pytest.raises(
        ResearchIntegrityError,
        match="split",
    ):
        load_split_manifest(
            path,
            df,
            changed_config,
        )


def test_load_split_manifest_rejects_changed_split_strategy(tmp_path):
    """A frozen split must not be reused under a different split strategy."""
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)

    original_config = _config(
        csv,
        tmp_path / "a",
        tmp_path,
        split_strategy="grouped_stratified",
    )
    manifest = make_split_manifest(df, original_config)

    path = tmp_path / "split_manifest.json"
    save_manifest(manifest, path)

    changed_config = _config(
        csv,
        tmp_path / "a",
        tmp_path,
        split_strategy="grouped_random",
    )

    with pytest.raises(
        ResearchIntegrityError,
        match="split",
    ):
        load_split_manifest(
            path,
            df,
            changed_config,
        )

def test_grouped_random_strategy_remains_available(tmp_path):
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, _config(csv, tmp_path / "a", tmp_path,
                                               split_strategy="grouped_random"))
    assert manifest["split_strategy"] == "grouped_random"
    assert manifest["stratify_columns"] == []
    assert_split_integrity(df, manifest)


def test_unknown_split_strategy_is_rejected(tmp_path):
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    with pytest.raises(ResearchIntegrityError, match="split_strategy must be"):
        make_split_manifest(df, _config(csv, tmp_path / "a", tmp_path, split_strategy="random"))


def test_stratification_degrades_gracefully_when_groups_are_scarce(tmp_path):
    """Requesting a rich stratification on few groups drops columns, not the run."""
    csv = _dataset(tmp_path, n_groups=8)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, _config(csv, tmp_path / "a", tmp_path,
                                               stratify_columns=["label", "difficulty_bucket"]))
    # Falls back to label-only rather than raising on a stratum with too few groups.
    assert manifest["stratify_columns"] == ["label"]
    assert_split_integrity(df, manifest)


def test_partition_summary_reports_rows_groups_and_balance(tmp_path):
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, _config(csv, tmp_path / "a", tmp_path))
    summary = summarize_partitions(df, manifest)
    assert sum(s["n_rows"] for s in summary.values()) == len(df)
    assert sum(s["n_groups"] for s in summary.values()) == df.split_group_id.nunique()
    for part in summary.values():
        assert set(part["label_counts"]) == {"0", "1"}


def test_single_class_partition_is_a_hard_error(tmp_path):
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, _config(csv, tmp_path / "a", tmp_path))
    # Force every positive row out of the test partition.
    positives = set(df[df.label == 1].sample_id.astype(str))
    manifest["test"] = [s for s in manifest["test"] if str(s) not in positives]
    manifest["train"] = manifest["train"] + [s for s in df.sample_id if str(s) in positives
                                             and s not in manifest["validation"]]
    manifest.pop("partition_summary", None)
    with pytest.raises(ResearchIntegrityError, match="does not contain both label classes"):
        assert_partition_label_availability(df, manifest)


# --- seed-level summary ---------------------------------------------------------

def test_seed_summary_reports_spread_and_layer_stability():
    runs = [{"seed": 1, "selected_layer": 2, "test": {"auroc": 0.8, "pr_auc": 0.7, "ece": 0.1}},
            {"seed": 2, "selected_layer": 2, "test": {"auroc": 0.9, "pr_auc": 0.8, "ece": 0.2}},
            {"seed": 3, "selected_layer": 3, "test": {"auroc": 0.7, "pr_auc": 0.6, "ece": 0.3}}]
    summary = summarize_across_seeds(runs)
    assert summary["test_auroc"]["mean"] == pytest.approx(0.8)
    assert summary["test_auroc"]["n_seeds"] == 3
    assert summary["test_auroc"]["std"] > 0
    assert summary["selected_layer"]["modal"] == 2
    assert summary["selected_layer"]["n_distinct"] == 2
    assert "test_ece" in summary


def test_probe_experiment_emits_a_seed_summary(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    result = run_probe_experiment(
        df,
        load_activations(df, activation_dir, config),
        manifest,
        config,
    )
    assert result["seed_summary"]["test_auroc"]["n_seeds"] == 2
    assert "ece" in result["runs"][0]["test"]
    assert len(result["runs"][0]["validation_auroc_by_layer"]) == 4


def test_probe_result_distinguishes_tensor_axis_from_physical_layer(tmp_path):
    """A selected activation axis must report its true transformer layer."""
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    activation_dir = tmp_path / "activations"

    run_extraction(
        df,
        ExtractionSpec(
            subject_model=STUB_MODEL,
            prompt_template_id="poker_action_v1",
            layer_indices=(1, 3),
        ),
        activation_dir,
        StubHiddenStateProvider(),
    )

    layer_indices = load_activation_layer_indices(activation_dir)
    assert layer_indices == [1, 3]

    activations = np.zeros((len(df), 2, 3), dtype=np.float32)
    activations[:, 1, 0] = df["label"].to_numpy(dtype=np.float32)

    config = _config(csv, activation_dir, tmp_path)
    manifest = make_split_manifest(df, config)

    result = run_probe_experiment(
        df,
        activations,
        manifest,
        config,
        layer_indices=layer_indices,
    )

    assert result["activation_layer_indices"] == [1, 3]

    for run in result["runs"]:
        assert run["selected_layer_index"] == 1
        assert run["selected_physical_layer"] == 3
        assert run["selected_layer"] == 1

def test_learning_curve_reports_physical_layer_for_subset_extraction(tmp_path):
    """Learning curves must distinguish stored tensor axes from model layers."""
    csv = _dataset(tmp_path)
    df = validate_dataset(csv)
    activation_dir = tmp_path / "activations"

    run_extraction(
        df,
        ExtractionSpec(
            subject_model=STUB_MODEL,
            prompt_template_id="poker_action_v1",
            layer_indices=(1, 3),
        ),
        activation_dir,
        StubHiddenStateProvider(),
    )

    layer_indices = load_activation_layer_indices(activation_dir)
    assert layer_indices == [1, 3]

    # Tensor axis 1 contains the decodable signal, but that axis represents
    # physical transformer layer 3.
    activations = np.zeros((len(df), 2, 3), dtype=np.float32)
    activations[:, 1, 0] = df["label"].to_numpy(dtype=np.float32)

    config = _config(
        csv,
        activation_dir,
        tmp_path,
        learning_curve_sizes=[6],
        learning_curve_subsamples=2,
    )
    manifest = make_split_manifest(df, config)

    result = run_probe_experiment(
        df,
        activations,
        manifest,
        config,
        layer_indices=layer_indices,
    )

    curve = result["learning_curve"]

    assert curve["layer"] == 1
    assert curve["layer_index"] == 1
    assert curve["physical_layer"] == 3

    for point in curve["points"]:
        if point.get("status") != "ok":
            continue

        for replicate in point["replicates"]:
            assert replicate["selected_layer"] == 1
            assert replicate["selected_layer_index"] == 1
            assert replicate["selected_physical_layer"] == 3

# --- learning curves

def test_learning_curve_uses_multiple_independent_subsamples(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path, learning_curve_sizes=[4, 8, 12],
                     learning_curve_subsamples=3)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    result = run_probe_experiment(df, load_activations(df, activation_dir, config), manifest, config)
    curve = result["learning_curve"]
    assert curve["subsample_unit"] == "group"
    ok = [p for p in curve["points"] if p["status"] == "ok"]
    assert len(ok) == 3
    for point in ok:
        # Independent replicates, each with its own uncertainty contribution.
        assert point["n_replicates"] == 3
        assert len({r["n_train_rows"] for r in point["replicates"]}) >= 1
        assert point["test_auroc_std"] >= 0.0
        assert point["test_auroc_range"][0] <= point["test_auroc_mean"] <= point["test_auroc_range"][1]
    assert [p["n_train_groups"] for p in ok] == [4, 8, 12]


def test_learning_curve_subsamples_are_distinct_draws(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path, learning_curve_sizes=[6],
                     learning_curve_subsamples=4)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    result = run_probe_experiment(df, load_activations(df, activation_dir, config), manifest, config)
    point = result["learning_curve"]["points"][0]
    # Assert on the draws themselves: a fixture with no learnable signal can give
    # identical scores across genuinely different subsamples, so equal AUROCs
    # would not prove the replicates collapsed.
    draws = {r["train_groups_sha256"] for r in point["replicates"]}
    assert len(draws) == 4
    assert all(r["n_train_groups"] == 6 for r in point["replicates"])


def test_learning_curve_reports_an_oversized_request_as_not_run(tmp_path):
    csv, activation_dir = _fixture(tmp_path, n_groups=12)
    config = _config(csv, activation_dir, tmp_path, learning_curve_sizes=[4, 9999],
                     learning_curve_subsamples=2)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    result = run_probe_experiment(df, load_activations(df, activation_dir, config), manifest, config)
    points = {p["n_train_groups"]: p for p in result["learning_curve"]["points"]}
    assert points[9999]["status"] == "not_run"
    assert "available" in points[9999]["reason"]
    assert points[4]["status"] == "ok"


def test_learning_curve_is_absent_unless_configured(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    result = run_probe_experiment(df, load_activations(df, activation_dir, config), manifest, config)
    assert "learning_curve" not in result


def test_learning_curve_is_reproducible(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path, learning_curve_sizes=[8],
                     learning_curve_subsamples=3)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    first = run_probe_experiment(df, activations, manifest, config)["learning_curve"]
    second = run_probe_experiment(df, activations, manifest, config)["learning_curve"]
    assert first == second


# --- learning-curve layer policy ------------------------------------------------

def test_curve_defaults_to_the_validation_selected_layer(tmp_path):
    """Re-searching all layers per subsample is n_layers x more fits and asks a
    noisier question, so the pipeline fixes the layer and records its source."""
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path, learning_curve_sizes=[6, 12],
                     learning_curve_subsamples=2)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    result = run_probe_experiment(df, load_activations(df, activation_dir, config), manifest, config)
    curve = result["learning_curve"]
    assert curve["layer_policy"] == "fixed"
    assert curve["layer"] == result["seed_summary"]["selected_layer"]["modal"]
    assert "modal validation-selected layer" in curve["layer_source"]
    # Every replicate reports the fixed layer, so the artifact shape is unchanged.
    for point in curve["points"]:
        if point.get("status") == "ok":
            assert {r["selected_layer"] for r in point["replicates"]} == {curve["layer"]}


def test_curve_can_still_reselect_per_subsample_when_asked(tmp_path):
    from deception_circuits.paper import run_learning_curve

    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path, learning_curve_sizes=[8],
                     learning_curve_subsamples=2)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    curve = run_learning_curve(df, activations, manifest, config, layer=None)
    assert curve["layer_policy"] == "reselected_per_subsample"
    assert curve["layer"] is None


def test_fixing_the_layer_does_not_change_which_groups_are_drawn(tmp_path):
    """The cost saving must come from fewer fits, not from a different sample."""
    from deception_circuits.paper import run_learning_curve

    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path, learning_curve_sizes=[8],
                     learning_curve_subsamples=3)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    fixed = run_learning_curve(df, activations, manifest, config, layer=1)
    searched = run_learning_curve(df, activations, manifest, config, layer=None)

    def draws(curve):
        return [r["train_groups_sha256"] for p in curve["points"]
                if p.get("status") == "ok" for r in p["replicates"]]

    assert draws(fixed) == draws(searched)
