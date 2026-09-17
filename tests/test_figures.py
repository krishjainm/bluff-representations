"""Tests for the figure suite.

The property that matters most: a figure is never produced from absent data, so
a publication directory cannot fill with plausible-looking plots that correspond
to nothing. The JSON artifacts below are minimal named test fixtures.
"""
from pathlib import Path
import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deception_circuits.paper import ResearchIntegrityError
from deception_circuits.paper_figures import (FIGURE_BUILDERS, audit_figures, build_all_figures,
                                              build_figure_context, load_result_artifacts)


def _probe_results(*, with_curve: bool = True) -> dict:
    runs = [
        {
            "seed": s,
            "selected_layer": 2,
            "selected_layer_index": 2,
            "selected_physical_layer": 8,
            "validation_auroc": 0.8,
            "validation_auroc_by_layer": [0.55, 0.62, 0.81, 0.74],
            "test": {"auroc": 0.78, "pr_auc": 0.6, "ece": 0.08},
        }
        for s in (1, 2, 3)
    ]
    result = {
        "selection_partition": "validation",
        "test_partition_used_for_selection": False,
        "activation_layer_indices": [1, 3, 8, 12],
        "n_train": 300,
        "n_validation": 60,
        "n_test": 90,
        "runs": runs,
        "test_auroc_mean": 0.78, "test_pr_auc_mean": 0.6,
        "seed_summary": {"test_auroc": {"mean": 0.78, "n_seeds": 3}},
        "baselines": {
            "majority_class_accuracy": 0.83,
            "prompt_text": {"auroc": 0.61}, "nuisance_only": {"auroc": 0.72},
            "response_text_diagnostic": {"auroc": 0.999},
        },
    }
    if with_curve:
        result["learning_curve"] = {
            "subsample_unit": "group", "n_available_train_groups": 50,
            "points": [
                {"n_train_groups": 10, "status": "ok", "n_replicates": 4,
                 "test_auroc_mean": 0.66, "test_auroc_std": 0.04,
                 "test_auroc_range": [0.60, 0.71]},
                {"n_train_groups": 30, "status": "ok", "n_replicates": 4,
                 "test_auroc_mean": 0.75, "test_auroc_std": 0.02,
                 "test_auroc_range": [0.72, 0.78]},
                {"n_train_groups": 9999, "status": "not_run",
                 "reason": "requested 9999 training groups but 50 are available"},
            ],
        }
    return result


def _confound_results() -> dict:
    return {
        "selected_layer": 2,
        "selected_layer_index": 2,
        "selected_physical_layer": 8,
        "layer_selected_on": "validation (upstream)",
        "metadata_availability": {
            "usable_columns": ["action"],
            "unavailable_analyses": [],
        },
        "controlled_probe": {"layer": 2, "unadjusted": {"auroc": 0.78},
                             "residualized": {"auroc": 0.63}, "auroc_drop_after_control": 0.15},
        "subsets": {
            "action_matched": {"matching": {"n_matched_rows": 200},
                               "probe": {"status": "ok", "auroc": 0.71, "n_test": 60,
                                         "test_auroc_grouped_ci": {"lower": 0.63, "upper": 0.79}}},
            "equity_matched": {"status": "not_run", "reason": "requires usable ['hand_strength']"},
        },
        "unavailable_analyses": ["equity_matched"],
    }

def _transfer_results() -> dict:
    return {
        "status": "ok",
        "context_column": "scenario",
        "contexts": ["negotiation", "poker"],
        "activation_layer_indices": [1, 3, 8, 12],
        "selection_partition": "source validation only",
        "target_test_used_for_selection": False,
        "cells": [
            {
                "source_context": "negotiation",
                "target_context": "poker",
                "status": "ok",
                "selected_layer_index": 2,
                "selected_physical_layer": 8,
                "source_validation_auroc": 0.82,
                "source_validation_auroc_by_layer": [0.55, 0.64, 0.82, 0.71],
                "n_source_train": 120,
                "n_source_validation": 30,
                "n_target_test": 45,
                "target_test": {
                    "auroc": 0.73,
                    "pr_auc": 0.61,
                    "ece": 0.09,
                },
                "target_test_auroc_grouped_ci": {
                    "lower": 0.64,
                    "upper": 0.81,
                },
            },
            {
                "source_context": "poker",
                "target_context": "negotiation",
                "status": "ok",
                "selected_layer_index": 1,
                "selected_physical_layer": 3,
                "source_validation_auroc": 0.79,
                "source_validation_auroc_by_layer": [0.58, 0.79, 0.72, 0.69],
                "n_source_train": 120,
                "n_source_validation": 30,
                "n_target_test": 45,
                "target_test": {
                    "auroc": 0.68,
                    "pr_auc": 0.57,
                    "ece": 0.11,
                },
                "target_test_auroc_grouped_ci": {
                    "lower": 0.59,
                    "upper": 0.77,
                },
            },
        ],
    }

def _causal_results() -> dict:
    def effect(mean):
        return {"mean_difference": mean, "ci_lower": mean - 0.05, "ci_upper": mean + 0.05,
                "n_pairs": 90, "n_groups": 45, "effect_size_dz": mean / 0.1,
                "signed_consistency_rate": 0.9, "mean_relative_magnitude": 0.12}
    return {
        "primary_endpoint": "forced_choice_positive_probability",
        "primary_endpoint_is_probe_derived": False,
        "probe_score_role": "diagnostic only; never the primary endpoint",
        "decoding": "deterministic single forward pass (no sampling)",
        "layer": 2, "n_prompts": 90, "n_groups": 45, "strengths": [1.0, 2.0],
        "conditions": ["baseline", "positive_steering", "negative_steering", "orthogonal"],
        "directions": {"probe": {"name": "probe", "source": "logistic probe weights"}},
        "effects": {
            "positive_steering": {"notes": "", "by_strength": {"1": effect(0.12), "2": effect(0.24)}},
            "negative_steering": {"notes": "", "by_strength": {"1": effect(-0.10), "2": effect(-0.22)}},
            "orthogonal": {"notes": "", "by_strength": {"1": effect(0.01), "2": effect(0.02)}},
        },
        "quality_controls": {
            "baseline": {"0": {"n": 90, "mean_valid_choice_mass": 0.94,
                               "min_valid_choice_mass": 0.8, "chose_positive_rate": 0.4}},
            "positive_steering": {"2": {"n": 90, "mean_valid_choice_mass": 0.88,
                                        "min_valid_choice_mass": 0.6, "chose_positive_rate": 0.7}},
        },
    }


def _sae_results() -> dict:
    def diag(n, ev):
        return {"partition": "x", "n_samples": n, "n_features": 48, "input_dim": 24,
                "expansion_factor": 2.0, "reconstruction_mse": 0.4,
                "fraction_variance_explained": ev, "fraction_variance_unexplained": 1 - ev,
                "l0_mean": 3.0, "l0_std": 0.0, "dead_feature_fraction": 0.2,
                "n_dead_features": 10, "activation_frequency": {"mean": 0.06},
                "code_magnitude_mean": 1.1}
    return {
        "layer": 2, "unsupervised": True,
        "labels_used_for": "feature ranking on train+validation only, after the SAE was frozen",
        "sae_config": {"n_features": 48, "activation": "topk", "k": 3},
        "expansion_factor": 2.0,
        "training": {"labels_used_in_training": False, "selected_epoch": 20},
        "diagnostics": {"train": diag(300, 0.71), "validation": diag(60, 0.68),
                        "test": diag(90, 0.66)},
        "feature_ranking_top": [{"feature": 4, "auroc": 0.81}, {"feature": 9, "auroc": 0.74}],
        "selected_features": [4, 9],
        "held_out_feature_evaluation": {
            "selection_partition": "train+validation", "evaluation_partition": "test",
            "n_test": 90, "sae_unchanged_by_evaluation": True,
            "features": [{"feature": 4, "auroc": 0.77, "activation_frequency": 0.2, "status": "ok"},
                         {"feature": 9, "auroc": 0.55, "activation_frequency": 0.1, "status": "ok"}],
        },
        "feature_examples": [{"feature": 4, "interpretation": "not assigned; requires human inspection"}],
        "sae_frozen_before_evaluation": True, "sae_unchanged_after_evaluation": True,
    }


def _write_artifacts(root: Path, **which) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    payloads = {
        "probe": ("probe_results.json", _probe_results),
        "confound": ("confound_results.json", _confound_results),
        "transfer": ("cross_context_results.json", _transfer_results),
        "causal": ("causal_results.json", _causal_results),
        "sae": ("sae_results.json", _sae_results),
    }
    for key, (filename, builder) in payloads.items():
        if which.get(key, True):
            (root / filename).write_text(json.dumps(builder()))
    (root / "split_manifest.json").write_text(json.dumps(
        {"schema_version": 2, "dataset_sha256": "abc123", "group_column": "split_group_id",
         "train": [], "validation": [], "test": []}))
    (root / "run_metadata.json").write_text(json.dumps(
        {"git_commit": "deadbeef", "config": {"subject_model": "stub/model",
                                              "dataset_path": "data/ds.csv",
                                              "experiment_name": "fig_test"}}))
    return root


# --- no figures from absent data ------------------------------------------------

def test_empty_output_directory_is_refused(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(ResearchIntegrityError, match="no sample-data path"):
        build_all_figures(tmp_path / "out")


def test_missing_output_directory_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_all_figures(tmp_path / "nope")

def test_every_figure_reports_a_reason_when_its_artifact_is_absent(tmp_path):
    root = _write_artifacts(
        tmp_path / "out",
        probe=False,
        confound=False,
        transfer=False,
        causal=False,
        sae=False,
    )

    manifest = build_all_figures(root)
    assert manifest["n_generated"] == 0
    assert manifest["n_not_run"] == len(FIGURE_BUILDERS)
    for name, reason in manifest["not_run"].items():
        assert reason, name
    # No image files were written at all.
    assert not list(Path(manifest["figures_dir"]).glob("*.png"))


def test_cross_scenario_uses_real_transfer_artifact(tmp_path):
    root = _write_artifacts(tmp_path / "out")

    manifest = build_all_figures(
        root,
        only=["cross_scenario"],
    )

    entry = manifest["figures"]["cross_scenario"]
    assert entry["status"] == "ok"

    data = pd.read_csv(entry["source_data"])

    assert len(data) == 2
    assert set(data["source_context"]) == {"poker", "negotiation"}
    assert set(data["target_context"]) == {"poker", "negotiation"}
    assert sorted(data["target_test_auroc"].tolist()) == [0.68, 0.73]
    assert entry["generated_from"].endswith(
        "cross_context_results.json"
    )

def test_learning_curve_is_skipped_when_unconfigured(tmp_path):
    root = tmp_path / "out"
    _write_artifacts(root, probe=False, confound=False, causal=False, sae=False)
    (root / "probe_results.json").write_text(json.dumps(_probe_results(with_curve=False)))
    manifest = build_all_figures(root, only=["learning_curve"])
    assert manifest["figures"]["learning_curve"]["status"] == "not_run"
    assert "learning_curve_sizes" in manifest["figures"]["learning_curve"]["reason"]


# --- generation ----------------------------------------------------------------

def test_all_available_figures_are_generated_with_source_data(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    manifest = build_all_figures(root)
    assert manifest["n_generated"] == len(FIGURE_BUILDERS)
    for name, entry in manifest["figures"].items():
        if entry["status"] != "ok":
            continue
        assert Path(entry["figure"]).is_file(), name
        assert Path(entry["source_data"]).is_file(), name
        assert entry["caption"], name
        assert entry["uncertainty_method"], name


def test_manifest_traces_every_figure_to_a_real_artifact(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    manifest = build_all_figures(root)
    for name, entry in manifest["figures"].items():
        if entry["status"] != "ok":
            continue
        assert Path(entry["generated_from"]).is_file(), name
        assert entry["source_sha256"], name
        assert entry["source_sha256"] in manifest["source_checksums"].values()


def test_source_csv_matches_the_plotted_numbers(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    manifest = build_all_figures(
        root,
        only=["learning_curve", "dose_response"],
    )

    curve = pd.read_csv(
        manifest["figures"]["learning_curve"]["source_data"]
    )
    assert curve.n_train_groups.tolist() == [10, 30]
    assert curve.test_auroc_mean.tolist() == [0.66, 0.75]

    dose = pd.read_csv(
        manifest["figures"]["dose_response"]["source_data"]
    )
    assert set(dose.condition) == {
        "positive_steering",
        "negative_steering",
    }
    assert sorted(dose.strength.unique()) == [1.0, 2.0]


def test_layerwise_probe_uses_physical_transformer_layers(tmp_path):
    root = _write_artifacts(tmp_path / "out")

    manifest = build_all_figures(
        root,
        only=["layerwise_probe"],
    )

    entry = manifest["figures"]["layerwise_probe"]
    assert entry["status"] == "ok"

    data = pd.read_csv(entry["source_data"])

    assert data["activation_layer_index"].tolist() == [0, 1, 2, 3]
    assert data["physical_layer"].tolist() == [1, 3, 8, 12]

    assert "physical transformer layer" in entry["caption"]
    assert "8" in entry["caption"]

def test_figure_context_carries_provenance(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    context = build_figure_context(load_result_artifacts(root))
    assert context.model == "stub/model"
    assert context.dataset == "data/ds.csv"
    assert context.dataset_sha256 == "abc123"
    assert context.git_commit == "deadbeef"
    assert "n=90" in context.subtitle(split="test", n=90)
    assert "layer 2" in context.subtitle(split="test", n=90, layer=2)


def test_unknown_figure_name_is_refused(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    with pytest.raises(ResearchIntegrityError, match="Unknown figure names"):
        build_all_figures(root, only=["not_a_figure"])


def test_confound_figure_names_the_controls_that_did_not_run(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    manifest = build_all_figures(root, only=["confound_subsets"])

    entry = manifest["figures"]["confound_subsets"]
    caption = entry["caption"]

    assert "physical transformer layer" in caption
    assert "Controls that could not run" in caption
    assert "equity_matched" in caption


def test_sae_feature_figure_refuses_to_name_features(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    manifest = build_all_figures(root, only=["sae_features"])
    caption = manifest["figures"]["sae_features"]["caption"]
    assert "does not make a feature a deception feature" in caption


# --- figure audit ---------------------------------------------------------------

def test_audit_passes_on_a_freshly_built_figure_set(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    manifest = build_all_figures(root)
    assert audit_figures(manifest["figures_dir"]) == []


def test_audit_reports_a_missing_manifest(tmp_path):
    failures = audit_figures(tmp_path / "nothing")
    assert any("missing figures manifest" in f for f in failures)


def test_audit_detects_a_deleted_image(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    manifest = build_all_figures(root)
    Path(manifest["figures"]["layerwise_probe"]["figure"]).unlink()
    failures = audit_figures(manifest["figures_dir"])
    assert any("layerwise_probe" in f and "figure is missing" in f for f in failures)


def test_audit_detects_a_deleted_source_artifact(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    manifest = build_all_figures(root)
    (root / "probe_results.json").unlink()
    failures = audit_figures(manifest["figures_dir"])
    assert any("points at a missing artifact" in f for f in failures)


def test_audit_detects_a_stale_figure(tmp_path):
    """A figure whose source artifact changed after generation is stale."""
    root = _write_artifacts(tmp_path / "out")
    manifest = build_all_figures(root)
    changed = _probe_results()
    changed["test_auroc_mean"] = 0.99
    (root / "probe_results.json").write_text(json.dumps(changed))
    failures = audit_figures(manifest["figures_dir"])
    assert any("is stale" in f for f in failures)


def test_audit_detects_a_caption_less_figure(tmp_path):
    root = _write_artifacts(tmp_path / "out")
    manifest = build_all_figures(root)
    path = Path(manifest["figures_dir"]) / "figures_manifest.json"
    payload = json.loads(path.read_text())
    payload["figures"]["layerwise_probe"]["caption"] = ""
    path.write_text(json.dumps(payload))
    failures = audit_figures(manifest["figures_dir"])
    assert any("no caption metadata" in f for f in failures)
