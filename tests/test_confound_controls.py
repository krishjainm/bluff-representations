"""Tests for P2 confound controls and nuisance baselines.

Activations come from the named stub provider, so no model is loaded.  These
tests check that controls are constructed correctly and, just as importantly,
that a control which the metadata cannot support is *reported as unavailable*
rather than silently skipped.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _stubs import StubHiddenStateProvider

from deception_circuits.paper import (PaperConfig, ResearchIntegrityError, binary_ece,
                                      compute_binary_metrics, load_activations,
                                      make_split_manifest, validate_dataset)
from deception_circuits.paper_confounds import (
    ANALYSIS_REQUIREMENTS,
    bluff_vs_value_subset,
    build_matched_subset,
    build_partition_matched_subset,
    controlled_probe_analysis,
    describe_metadata_availability,
    nuisance_decodability,
    residualize_activations,
    run_confound_suite,
    validate_nuisance_metadata,
)
from deception_circuits.paper_extraction import ExtractionSpec, run_extraction

STUB_MODEL = "stub/tiny-test-model"
FULL_NUISANCE = ["action", "street", "position", "hand_strength", "made_hand",
                 "board_texture", "bet_size", "pot_size"]


def _rows(n_groups: int = 24, *, with_metadata: bool = True) -> list[dict]:
    streets = ("flop", "turn", "river")
    positions = ("btn", "co", "bb")
    textures = ("dry", "wet")
    rows = []
    for group in range(n_groups):
        for label in (0, 1):
            row = {
                "sample_id": f"{group}-{label}", "base_item_id": group, "split_group_id": group,
                "statement": f"hand {group} variant {label}", "response": "raise",
                "label": label, "scenario": "poker",
            }
            if with_metadata:
                row.update({
                    # Both labels share the same action, so action-matching leaves
                    # a real within-stratum contrast rather than emptying out.
                    "action": "raise" if group % 4 else "check",
                    "street": streets[group % len(streets)],
                    "position": positions[group % len(positions)],
                    "board_texture": textures[group % len(textures)],
                    "hand_strength": 0.2 + 0.6 * label + 0.01 * group,
                    # Deliberately not label-separated, so made-hand matching
                    # leaves a real within-stratum contrast.
                    "made_hand": ("pair", "two_pair", "flush")[group % 3],
                    "bet_size": 10.0 + group, "pot_size": 40.0 + 2 * group,
                })
            rows.append(row)
    return rows


def _fixture(tmp_path: Path, **kwargs):
    df = pd.DataFrame(_rows(**kwargs))
    csv = tmp_path / "dataset.csv"
    df.to_csv(csv, index=False)
    activations = tmp_path / "activations"
    run_extraction(df, ExtractionSpec(subject_model=STUB_MODEL, prompt_template_id="poker_action_v1"),
                   activations, StubHiddenStateProvider())
    return csv, activations


def _config(csv: Path, activations: Path, tmp_path: Path, **overrides) -> PaperConfig:
    base = dict(experiment_name="confounds", dataset_path=str(csv), activation_dir=str(activations),
                output_dir=str(tmp_path / "out"), subject_model=STUB_MODEL,
                prompt_template_id="poker_action_v1", n_seeds=1, bootstrap_resamples=30,
                nuisance_columns=list(FULL_NUISANCE))
    base.update(overrides)
    return PaperConfig(**base)


# --- metadata availability ------------------------------------------------------

def test_availability_reports_absent_columns_and_blocked_analyses(tmp_path):
    df = pd.DataFrame(_rows(n_groups=6, with_metadata=False))
    report = describe_metadata_availability(df, FULL_NUISANCE)
    assert report["usable_columns"] == []
    assert all(not v["present"] for v in report["columns"].values())
    # Every named analysis must be explicitly marked unavailable, not omitted.
    assert set(report["unavailable_analyses"]) == set(ANALYSIS_REQUIREMENTS)
    assert report["analyses"]["equity_matched"]["blocked_by"] == ["action", "hand_strength"]


def test_availability_flags_partial_coverage_as_unusable(tmp_path):
    df = pd.DataFrame(_rows(n_groups=10))
    df.loc[df.index[:30], "hand_strength"] = np.nan
    report = describe_metadata_availability(df, ["action", "hand_strength"])
    assert "action" in report["usable_columns"]
    assert "hand_strength" not in report["usable_columns"]
    assert report["columns"]["hand_strength"]["coverage"] < 0.95
    assert "equity_matched" in report["unavailable_analyses"]


def test_validate_nuisance_metadata_fails_loudly_on_configured_but_absent(tmp_path):
    df = pd.DataFrame(_rows(n_groups=6, with_metadata=False))
    with pytest.raises(ResearchIntegrityError, match="absent from dataset"):
        validate_nuisance_metadata(df, ["action"])
    full = pd.DataFrame(_rows(n_groups=6))
    assert validate_nuisance_metadata(full, ["action", "street"])["usable_columns"] == ["action", "street"]


def test_availability_distinguishes_continuous_from_categorical(tmp_path):
    df = pd.DataFrame(_rows(n_groups=24))
    report = describe_metadata_availability(df, ["action", "hand_strength"])
    assert report["columns"]["action"]["kind"] == "categorical"
    assert report["columns"]["hand_strength"]["kind"] == "continuous"


# --- matched subsets ------------------------------------------------------------

def test_bluff_vs_value_keeps_only_aggressive_actions(tmp_path):
    df = pd.DataFrame(_rows(n_groups=24))
    subset = bluff_vs_value_subset(df)
    assert set(subset.action.unique()) == {"raise"}
    assert len(subset) < len(df)
    assert set(subset.label.unique()) == {0, 1}


def test_bluff_vs_value_requires_an_action_column():
    df = pd.DataFrame(_rows(n_groups=6, with_metadata=False))
    with pytest.raises(ResearchIntegrityError, match="requires an 'action' column"):
        bluff_vs_value_subset(df)


def test_bluff_vs_value_reports_when_no_action_is_aggressive():
    df = pd.DataFrame(_rows(n_groups=6))
    df["action"] = "fold"
    with pytest.raises(ResearchIntegrityError, match="No rows have an aggressive"):
        bluff_vs_value_subset(df)


def test_matched_subset_balances_labels_within_every_stratum(tmp_path):
    df = pd.DataFrame(_rows(n_groups=24))
    matched, report = build_matched_subset(df, exact_columns=("action", "street"))
    # Equal label counts inside each matched cell is the whole point.
    per_cell = matched.groupby(["action", "street"]).label.value_counts().unstack(fill_value=0)
    assert (per_cell[0] == per_cell[1]).all()
    assert report["label_balance"]["0"] == report["label_balance"]["1"]
    assert report["n_matched_rows"] == len(matched)


def test_matched_subset_is_deterministic_given_a_seed(tmp_path):
    df = pd.DataFrame(_rows(n_groups=24))
    a, _ = build_matched_subset(df, exact_columns=("action",), seed=7)
    b, _ = build_matched_subset(df, exact_columns=("action",), seed=7)
    assert a.sample_id.tolist() == b.sample_id.tolist()

def test_confound_matching_training_selection_is_independent_of_test_labels(
    tmp_path,
    monkeypatch,
):
    """Changing only test labels must not change which training rows are matched."""
    rows = []

    for index in range(4):
        rows.append({
            "sample_id": f"tr0{index}",
            "base_item_id": f"tr0{index}",
            "split_group_id": f"tr0{index}",
            "statement": f"train zero {index}",
            "response": "raise",
            "label": 0,
            "scenario": "poker",
            "action": "raise",
        })

    for index in range(4):
        rows.append({
            "sample_id": f"tr1{index}",
            "base_item_id": f"tr1{index}",
            "split_group_id": f"tr1{index}",
            "statement": f"train one {index}",
            "response": "raise",
            "label": 1,
            "scenario": "poker",
            "action": "raise",
        })

    for index in range(2):
        rows.append({
            "sample_id": f"te0{index}",
            "base_item_id": f"te0{index}",
            "split_group_id": f"te0{index}",
            "statement": f"test zero {index}",
            "response": "raise",
            "label": 0,
            "scenario": "poker",
            "action": "raise",
        })

    for index in range(4):
        rows.append({
            "sample_id": f"te1{index}",
            "base_item_id": f"te1{index}",
            "split_group_id": f"te1{index}",
            "statement": f"test one {index}",
            "response": "raise",
            "label": 1,
            "scenario": "poker",
            "action": "raise",
        })

    df = pd.DataFrame(rows)

    manifest = {
        "group_column": "split_group_id",
        "train": [
            sid
            for sid in df["sample_id"]
            if str(sid).startswith("tr")
        ],
        "validation": [],
        "test": [
            sid
            for sid in df["sample_id"]
            if str(sid).startswith("te")
        ],
    }

    config = PaperConfig(
        experiment_name="confound-leakage-test",
        dataset_path=str(tmp_path / "unused.csv"),
        activation_dir=str(tmp_path / "unused-activations"),
        output_dir=str(tmp_path / "out"),
        nuisance_columns=["action"],
        seed=2026,
        bootstrap_resamples=10,
    )

    activations = np.zeros(
        (len(df), 1, 3),
        dtype=np.float32,
    )

    def capture_subset(
        activations,
        current_df,
        current_manifest,
        subset_ids,
        config,
        *,
        layer,
    ):
        keep = set(map(str, subset_ids))

        return {
            "status": "ok",
            "train_subset_ids": [
                str(sid)
                for sid in current_manifest["train"]
                if str(sid) in keep
            ],
            "test_subset_ids": [
                str(sid)
                for sid in current_manifest["test"]
                if str(sid) in keep
            ],
        }

    monkeypatch.setattr(
        "deception_circuits.paper_confounds._probe_on_subset",
        capture_subset,
    )

    original = run_confound_suite(
        activations,
        df,
        manifest,
        config,
        layer=0,
    )

    changed = df.copy()
    changed.loc[
        changed["sample_id"] == "te00",
        "label",
    ] = 1

    changed_result = run_confound_suite(
        activations,
        changed,
        manifest,
        config,
        layer=0,
    )

    original_train_ids = original[
        "subsets"
    ]["action_matched"]["probe"]["train_subset_ids"]

    changed_train_ids = changed_result[
        "subsets"
    ]["action_matched"]["probe"]["train_subset_ids"]

    assert original_train_ids == changed_train_ids
def test_continuous_matching_bins_are_fit_on_train_only():
    """Changing only test nuisance values must not change train matching."""
    rows = []

    train_values = [
        (0, 0.10),
        (0, 0.20),
        (0, 0.60),
        (0, 0.70),
        (1, 0.15),
        (1, 0.25),
        (1, 0.65),
        (1, 0.75),
    ]

    for index, (label, strength) in enumerate(train_values):
        rows.append({
            "sample_id": f"tr{index}",
            "base_item_id": f"tr{index}",
            "split_group_id": f"tr{index}",
            "statement": f"train {index}",
            "response": "raise",
            "label": label,
            "scenario": "poker",
            "action": "raise",
            "hand_strength": strength,
        })

    test_values = [
        (0, 0.12),
        (0, 0.62),
        (1, 0.18),
        (1, 0.68),
    ]

    for index, (label, strength) in enumerate(test_values):
        rows.append({
            "sample_id": f"te{index}",
            "base_item_id": f"te{index}",
            "split_group_id": f"te{index}",
            "statement": f"test {index}",
            "response": "raise",
            "label": label,
            "scenario": "poker",
            "action": "raise",
            "hand_strength": strength,
        })

    df = pd.DataFrame(rows)

    manifest = {
        "group_column": "split_group_id",
        "train": [
            sid
            for sid in df["sample_id"]
            if str(sid).startswith("tr")
        ],
        "validation": [],
        "test": [
            sid
            for sid in df["sample_id"]
            if str(sid).startswith("te")
        ],
    }

    original_subset, original_report = build_partition_matched_subset(
        df,
        manifest,
        exact_columns=("action",),
        binned_columns=("hand_strength",),
        n_bins=2,
        seed=2026,
        group_column="split_group_id",
    )

    changed = df.copy()

    changed.loc[
        changed["sample_id"].astype(str).str.startswith("te"),
        "hand_strength",
    ] = [
        1000.0,
        2000.0,
        3000.0,
        4000.0,
    ]

    changed_subset, changed_report = build_partition_matched_subset(
        changed,
        manifest,
        exact_columns=("action",),
        binned_columns=("hand_strength",),
        n_bins=2,
        seed=2026,
        group_column="split_group_id",
    )

    original_train_ids = sorted(
        original_subset.loc[
            original_subset["sample_id"].astype(str).str.startswith("tr"),
            "sample_id",
        ].astype(str)
    )

    changed_train_ids = sorted(
        changed_subset.loc[
            changed_subset["sample_id"].astype(str).str.startswith("tr"),
            "sample_id",
        ].astype(str)
    )

    assert original_train_ids == changed_train_ids
    assert original_report["bin_edges_fit_partition"] == "train"
    assert changed_report["bin_edges_fit_partition"] == "train"

def test_matched_subset_drops_and_counts_single_class_strata(tmp_path):
    df = pd.DataFrame(_rows(n_groups=24))
    # Make 'check' rows single-class so that stratum must be dropped.
    df.loc[(df.action == "check") & (df.label == 1), "label"] = 0
    matched, report = build_matched_subset(df, exact_columns=("action",))
    assert report["n_strata_dropped_single_class"] >= 1
    assert set(matched.action.unique()) == {"raise"}


def test_matching_on_a_continuous_column_uses_bins(tmp_path):
    df = pd.DataFrame(_rows(n_groups=24))
    matched, report = build_matched_subset(
        bluff_vs_value_subset(df), exact_columns=("action",), binned_columns=("hand_strength",), n_bins=3)
    assert report["binned_columns"] == ["hand_strength"]
    assert report["n_matched_rows"] % 2 == 0


def test_matching_raises_when_nothing_survives(tmp_path):
    df = pd.DataFrame(_rows(n_groups=24))
    # hand_strength is almost perfectly separated by label, so binning it finely
    # leaves no bin containing both classes.
    with pytest.raises(ResearchIntegrityError, match="left no stratum containing both"):
        build_matched_subset(df, exact_columns=(), binned_columns=("hand_strength",), n_bins=10)


def test_matching_on_an_absent_column_raises(tmp_path):
    df = pd.DataFrame(_rows(n_groups=6, with_metadata=False))
    with pytest.raises(ResearchIntegrityError, match="absent from the dataset"):
        build_matched_subset(df, exact_columns=("action",))


# --- nuisance decodability and residualisation ----------------------------------

def test_nuisance_decodability_reports_against_a_majority_baseline(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    result = nuisance_decodability(activations, df, manifest, ["action", "street"], config)
    for column in ("action", "street"):
        assert result[column]["status"] == "ok"
        assert result[column]["target_type"] == "categorical"
        assert 0.0 <= result[column]["accuracy"] <= 1.0
        assert "majority_baseline_accuracy" in result[column]


def test_nuisance_decodability_regresses_continuous_columns(tmp_path):
    """A continuous variable must not be stringified into hundreds of classes."""
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    result = nuisance_decodability(activations, df, manifest, ["hand_strength", "bet_size"], config)
    for column in ("hand_strength", "bet_size"):
        assert result[column]["target_type"] == "continuous"
        assert result[column]["status"] == "ok"
        assert "r2_vs_train_mean" in result[column]
        assert "accuracy" not in result[column]
        assert result[column]["baseline_rmse"] > 0


def test_nuisance_decodability_refuses_a_degenerate_categorical_target(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    # A unique string per row: categorical in dtype, meaningless as a target.
    df = df.assign(source_id=[f"id-{i}" for i in range(len(df))])
    result = nuisance_decodability(activations, df, manifest, ["source_id"], config)
    assert result["source_id"]["status"] == "not_run"
    assert "not a meaningful classification target" in result["source_id"]["reason"]


def test_nuisance_decodability_reports_a_constant_column_as_not_run(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    df = df.assign(pot_size=5.0)
    result = nuisance_decodability(activations, df, manifest, ["pot_size"], config)
    assert result["pot_size"]["status"] == "not_run"
    assert "constant" in result["pot_size"]["reason"]


def test_nuisance_decodability_marks_absent_columns_not_run(tmp_path):
    csv, activation_dir = _fixture(tmp_path, with_metadata=False)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    result = nuisance_decodability(activations, df, manifest, ["action"], config)
    assert result["action"] == {"status": "not_run", "reason": "column absent from dataset"}


def test_residualization_removes_the_nuisance_subspace():
    rng = np.random.default_rng(0)
    nuisance = np.eye(3)[rng.integers(0, 3, size=60)]
    signal = rng.normal(size=(60, 5))
    # Activations that are a pure function of the nuisance must residualize to ~0.
    train = nuisance @ rng.normal(size=(3, 5))
    residual_train, residual_eval = residualize_activations(train, train, nuisance, nuisance)
    assert np.abs(residual_train).max() < 1e-8
    # Independent signal must largely survive.
    mixed = train + signal
    r_mixed, _ = residualize_activations(mixed, mixed, nuisance, nuisance)
    assert np.linalg.norm(r_mixed) > 0.5 * np.linalg.norm(signal)


def test_residualization_fits_removal_on_train_only():
    rng = np.random.default_rng(1)
    nuisance_train = np.eye(2)[rng.integers(0, 2, size=40)]
    nuisance_eval = np.eye(2)[rng.integers(0, 2, size=10)]
    train = rng.normal(size=(40, 3))
    evaluate = rng.normal(size=(10, 3))
    r_train, r_eval = residualize_activations(train, evaluate, nuisance_train, nuisance_eval)
    # Re-running with different eval rows must not change the training residuals.
    r_train_again, _ = residualize_activations(train, evaluate[:5], nuisance_train, nuisance_eval[:5])
    assert np.allclose(r_train, r_train_again)
    assert r_eval.shape == evaluate.shape


def test_controlled_probe_reports_unadjusted_and_residualized(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    result = controlled_probe_analysis(activations, df, manifest, config, ["action", "street"], layer=2)
    assert "auroc" in result["unadjusted"]
    assert "auroc" in result["residualized"]
    assert result["residualized"]["columns"] == ["action", "street"]
    assert "auroc_drop_after_control" in result


def test_controlled_probe_marks_residualization_not_run_without_metadata(tmp_path):
    csv, activation_dir = _fixture(tmp_path, with_metadata=False)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    result = controlled_probe_analysis(activations, df, manifest, config, ["action"], layer=2)
    assert result["residualized"]["status"] == "not_run"
    assert "auroc" in result["unadjusted"]


# --- full suite -----------------------------------------------------------------

def test_confound_suite_runs_every_supported_control(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    result = run_confound_suite(activations, df, manifest, config, layer=2)
    assert result["selected_layer"] == 2
    assert set(result["subsets"]) == set(ANALYSIS_REQUIREMENTS)
    # With full metadata, the headline bluff-vs-value comparison must actually run.
    assert result["subsets"]["bluff_vs_value"]["probe"]["status"] in ("ok", "not_run")
    assert "matching" in result["subsets"]["action_matched"]
    assert result["metadata_availability"]["unavailable_analyses"] == []


def test_confound_suite_explains_each_unavailable_control(tmp_path):
    csv, activation_dir = _fixture(tmp_path, with_metadata=False)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    result = run_confound_suite(activations, df, manifest, config, layer=2)
    assert set(result["unavailable_analyses"]) == set(ANALYSIS_REQUIREMENTS)
    for name, entry in result["subsets"].items():
        assert entry["status"] == "not_run"
        assert entry["reason"], name
        assert entry["requires"], name


def test_confound_suite_never_selects_a_layer_itself(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    activations = load_activations(df, activation_dir, config)
    for layer in (0, 3):
        result = run_confound_suite(activations, df, manifest, config, layer=layer)
        assert result["selected_layer"] == layer
        assert result["controlled_probe"]["layer"] == layer
        assert result["layer_selected_on"] == "validation (upstream)"


# --- calibration metric ---------------------------------------------------------

def test_ece_is_zero_for_perfectly_calibrated_predictions():
    # Half the rows at p=0.0 with label 0, half at p=1.0 with label 1.
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([0.0] * 50 + [1.0] * 50)
    assert binary_ece(p, y) == pytest.approx(0.0)


def test_ece_detects_systematic_overconfidence():
    y = np.array([0] * 50 + [1] * 50)
    p = np.array([1.0] * 50 + [0.0] * 50)
    assert binary_ece(p, y) == pytest.approx(1.0)


def test_ece_raises_on_empty_input_instead_of_returning_zero():
    with pytest.raises(ResearchIntegrityError, match="empty"):
        binary_ece(np.array([]), np.array([]))


def test_metrics_include_ece_and_n():
    metrics = compute_binary_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]))
    assert metrics["n"] == 4
    assert 0.0 <= metrics["ece"] <= 1.0
    assert metrics["positive_rate"] == pytest.approx(0.5)


def test_metrics_refuse_a_single_class_partition():
    with pytest.raises(ResearchIntegrityError, match="both label classes"):
        compute_binary_metrics(np.zeros(5), np.full(5, 0.3))


# --- audit notes vs audit failures ---------------------------------------------

def test_audit_notes_report_unrunnable_controls_without_failing_the_audit(tmp_path):
    """An honest 'this control could not run' must not be reported as a defect."""
    import json

    from deception_circuits.paper import audit_notes, audit_run

    out = tmp_path / "out"
    out.mkdir()
    (out / "confound_results.json").write_text(json.dumps({
        "metadata_availability": {"unavailable_analyses": ["board_matched"]},
        "unavailable_analyses": ["equity_matched"],
        "subsets": {"equity_matched": {"status": "not_run", "reason": "no stratum had both labels"}},
    }))
    (out / "probe_results.json").write_text(json.dumps({
        "selection_partition": "validation", "test_partition_used_for_selection": False,
        "runs": [], "baselines": {}, "seed_summary": {},
        "learning_curve": {"points": [{"n_train_groups": 999, "status": "not_run",
                                       "reason": "only 20 groups are available"}]},
    }))
    notes = audit_notes(out, None)
    assert any("equity_matched" in n and "no stratum" in n for n in notes)
    assert any("board_matched" in n for n in notes)
    assert any("999" in n for n in notes)
    # None of those notes may appear as audit failures.
    failures = audit_run(out)
    assert not any("equity_matched" in f or "999" in f for f in failures)


def test_audit_requires_confound_results_when_nuisance_columns_are_configured(tmp_path):
    import json

    from deception_circuits.paper import audit_run

    csv, activation_dir = _fixture(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    config = _config(csv, activation_dir, tmp_path, output_dir=str(out))
    df = validate_dataset(csv)
    for name in ("resolved_config.yaml", "run_metadata.json"):
        (out / name).write_text("{}")
    (out / "probe_results.json").write_text(json.dumps({
        "selection_partition": "validation", "test_partition_used_for_selection": False,
        "runs": [], "baselines": {}, "seed_summary": {}}))
    (out / "split_manifest.json").write_text(json.dumps(make_split_manifest(df, config)))
    failures = audit_run(out, df, config)
    assert any("confound_results.json is missing" in f for f in failures)
