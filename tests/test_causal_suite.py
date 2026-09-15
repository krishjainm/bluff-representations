"""Tests for the P3 behavioral causal intervention suite.

The central property under test is non-circularity: the primary endpoint must be
something the model emits, not the probe score whose direction defines the
intervention. Everything runs against the named stub runner -- no model is
loaded and no API is contacted.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _stubs import StubInterventionRunner

from deception_circuits.paper import PaperConfig, ResearchIntegrityError
from deception_circuits.paper_causal import (Condition, Direction, ForcedChoiceEndpoint,
                                             Intervention, assert_endpoint_independence,
                                             build_direction_set, default_conditions,
                                             nuisance_condition, orthogonal_direction,
                                             paired_bootstrap_effect, run_causal_suite,
                                             score_generation_quality, summarize_quality)

DIM = 16
N_LAYERS = 8
READOUT_LAYER = 5
ENDPOINT = ForcedChoiceEndpoint(options=("Yes", "No"), positive_option="Yes")


def _runner() -> StubInterventionRunner:
    return StubInterventionRunner(dim=DIM, n_layers=N_LAYERS, readout_layer=READOUT_LAYER, seed=0)


def _fixture(n_groups: int = 40):
    """Synthetic activations separated along the stub's readout direction.

    This makes the fitted probe direction approximately the stub's readout, which
    is the situation the controls are designed to discriminate.
    """
    runner = _runner()
    rng = np.random.default_rng(7)
    rows, activations = [], []
    for group in range(n_groups):
        for label in (0, 1):
            sid = f"s{group}-{label}"
            rows.append({"sample_id": sid, "base_item_id": group, "split_group_id": group,
                         "statement": f"hand {group} variant {label}", "response": "Yes" if label else "No",
                         "label": label, "scenario": "poker",
                         "street": ("flop", "turn", "river")[group % 3]})
            stack = rng.normal(scale=0.5, size=(N_LAYERS, DIM))
            stack[READOUT_LAYER] += (2.0 * label - 1.0) * runner.readout * 3.0
            activations.append(stack)
    df = pd.DataFrame(rows)
    return df, np.stack(activations), runner


def _manifest(df: pd.DataFrame) -> dict:
    ids = df.sample_id.tolist()
    cut = int(len(ids) * 0.7)
    return {"group_column": "split_group_id", "train": ids[:cut], "validation": [],
            "test": ids[cut:]}


def _config(**overrides) -> PaperConfig:
    base = dict(experiment_name="causal", dataset_path="x.csv", activation_dir="a",
                output_dir="o", seed=11)
    base.update(overrides)
    return PaperConfig(**base)


# --- non-circularity ------------------------------------------------------------

@pytest.mark.parametrize("name", ["probe_score", "probe_probability", "probe_logit"])
def test_probe_derived_endpoints_are_refused_as_primary(name):
    with pytest.raises(ResearchIntegrityError, match="circular"):
        assert_endpoint_independence(name)


def test_behavioral_endpoint_is_accepted():
    assert_endpoint_independence("forced_choice_positive_probability") is None


def test_suite_refuses_a_probe_derived_primary_endpoint():
    df, activations, runner = _fixture(n_groups=8)
    directions = build_direction_set(activations, df, _manifest(df), _config(), layer=READOUT_LAYER)
    bad = ForcedChoiceEndpoint(options=("Yes", "No"), positive_option="Yes", name="probe_score")
    with pytest.raises(ResearchIntegrityError, match="circular"):
        run_causal_suite(runner, df.head(4), directions, bad, layer=READOUT_LAYER, strengths=[1.0])


def test_probe_projection_is_labelled_as_diagnostic_only():
    df, activations, runner = _fixture(n_groups=10)
    directions = build_direction_set(activations, df, _manifest(df), _config(), layer=READOUT_LAYER)
    result = run_causal_suite(runner, df.head(6), directions, ENDPOINT,
                              layer=READOUT_LAYER, strengths=[1.0], n_resamples=30)
    assert result["primary_endpoint_is_probe_derived"] is False
    assert "diagnostic" in result["probe_score_role"]
    assert "diagnostic_probe_projection" in result["records"][0]
    # The endpoint itself must not be the probe projection.
    row = result["records"][0]
    assert row["endpoint_name"] == "forced_choice_positive_probability"
    assert row["endpoint_value"] != row["diagnostic_probe_projection"]


# --- steering vs patching terminology -------------------------------------------

def test_patching_with_a_strength_is_refused():
    with pytest.raises(ResearchIntegrityError, match="replaces the activation outright"):
        Intervention(kind="replace", vector=np.ones(4), layer=1, strength=0.5)


def test_steering_requires_a_vector_and_a_layer():
    with pytest.raises(ResearchIntegrityError, match="requires a vector"):
        Intervention(kind="add", layer=1, strength=1.0)
    with pytest.raises(ResearchIntegrityError, match="requires an explicit layer"):
        Intervention(kind="add", vector=np.ones(4), strength=1.0)


def test_baseline_carries_no_vector_or_strength():
    assert Intervention().kind == "none"
    with pytest.raises(ResearchIntegrityError, match="no vector or strength"):
        Intervention(kind="none", vector=np.ones(4))


def test_steering_and_patching_are_distinguishable():
    steer = Intervention(kind="add", vector=np.ones(4), layer=1, strength=2.0)
    patch = Intervention(kind="replace", vector=np.ones(4), layer=1)
    assert steer.is_steering and not steer.is_patching
    assert patch.is_patching and not patch.is_steering


def test_unknown_kind_and_timing_are_refused():
    with pytest.raises(ResearchIntegrityError, match="kind must be"):
        Intervention(kind="subtract", vector=np.ones(4), layer=1)
    with pytest.raises(ResearchIntegrityError, match="timing_policy must be"):
        Intervention(kind="add", vector=np.ones(4), layer=1, timing_policy="whenever")


# --- directions -----------------------------------------------------------------

def test_direction_set_contains_every_required_control():
    df, activations, _ = _fixture()
    directions = build_direction_set(activations, df, _manifest(df), _config(), layer=READOUT_LAYER)
    for name in ("probe", "random_matched_norm", "orthogonal", "shuffled_label"):
        assert name in directions
        assert np.isclose(np.linalg.norm(directions[name].vector), 1.0)
        assert directions[name].source


def test_probe_direction_recovers_the_readout():
    df, activations, runner = _fixture()
    directions = build_direction_set(activations, df, _manifest(df), _config(), layer=READOUT_LAYER)
    alignment = float(directions["probe"].vector @ runner.readout)
    assert alignment > 0.8, alignment


def test_orthogonal_direction_is_orthogonal_to_the_probe():
    df, activations, _ = _fixture()
    directions = build_direction_set(activations, df, _manifest(df), _config(), layer=READOUT_LAYER)
    assert abs(float(directions["orthogonal"].vector @ directions["probe"].vector)) < 1e-10


def test_orthogonal_direction_helper_is_deterministic():
    reference = np.array([1.0, 0.0, 0.0, 0.0])
    a = orthogonal_direction(reference, seed=3)
    assert np.allclose(a, orthogonal_direction(reference, seed=3))
    assert abs(float(a @ reference)) < 1e-12


def test_directions_are_fitted_on_train_only():
    """Mutating test-partition labels must not change any direction."""
    df, activations, _ = _fixture()
    manifest = _manifest(df)
    before = build_direction_set(activations, df, manifest, _config(), layer=READOUT_LAYER)
    flipped = df.copy()
    test_mask = flipped.sample_id.isin(manifest["test"])
    flipped.loc[test_mask, "label"] = 1 - flipped.loc[test_mask, "label"]
    after = build_direction_set(activations, flipped, manifest, _config(), layer=READOUT_LAYER)
    assert np.allclose(before["probe"].vector, after["probe"].vector)


def test_nuisance_direction_is_built_when_metadata_exists():
    df, activations, _ = _fixture()
    directions = build_direction_set(activations, df, _manifest(df), _config(),
                                     layer=READOUT_LAYER, nuisance_column="street")
    assert "nuisance_street" in directions
    assert "street" in directions["nuisance_street"].source


def test_nuisance_direction_is_absent_without_metadata():
    df, activations, _ = _fixture()
    directions = build_direction_set(activations, df, _manifest(df), _config(),
                                     layer=READOUT_LAYER, nuisance_column="hand_strength")
    assert not any(k.startswith("nuisance_") for k in directions)


def test_zero_norm_direction_is_refused():
    df, activations, _ = _fixture(n_groups=8)
    with pytest.raises(ResearchIntegrityError, match="zero-norm"):
        orthogonal_direction(np.zeros(4), seed=1)


# --- endpoint scoring -----------------------------------------------------------

def test_endpoint_scores_the_positive_option_probability():
    scored = ENDPOINT.score({"Yes": 1.0, "No": 0.0})
    assert scored["endpoint_value"] == pytest.approx(np.exp(1) / (np.exp(1) + 1))
    assert scored["chosen_option"] == "Yes"
    assert scored["chose_positive"] is True
    assert set(scored["option_probabilities"]) == {"Yes", "No"}


def test_endpoint_requires_logprobs_for_every_option():
    with pytest.raises(ResearchIntegrityError, match="did not return logprobs"):
        ENDPOINT.score({"Yes": 1.0})


def test_endpoint_construction_is_validated():
    with pytest.raises(ResearchIntegrityError, match="at least two distinct options"):
        ForcedChoiceEndpoint(options=("Yes", "Yes"), positive_option="Yes")
    with pytest.raises(ResearchIntegrityError, match="must be one of options"):
        ForcedChoiceEndpoint(options=("Yes", "No"), positive_option="Maybe")


# --- paired statistics ----------------------------------------------------------

def test_paired_bootstrap_recovers_a_known_shift():
    baseline = np.zeros(80)
    intervened = np.full(80, 0.25)
    effect = paired_bootstrap_effect(baseline, intervened, seed=1, n_resamples=200)
    assert effect["mean_difference"] == pytest.approx(0.25)
    assert effect["ci_lower"] == pytest.approx(0.25)
    assert effect["ci_upper"] == pytest.approx(0.25)
    assert effect["n_pairs"] == 80
    # Constant differences have no spread, so d_z is undefined rather than infinite.
    assert effect["effect_size_dz"] is None


def test_paired_bootstrap_reports_sign_consistency():
    rng = np.random.default_rng(0)
    baseline = rng.normal(size=100)
    effect = paired_bootstrap_effect(baseline, baseline + 1.0 + rng.normal(scale=0.1, size=100),
                                     seed=2, n_resamples=200)
    assert effect["mean_difference"] > 0.9
    assert effect["signed_consistency_rate"] > 0.95
    assert effect["effect_size_dz"] > 1.0


def test_paired_bootstrap_resamples_groups_not_rows():
    # Two groups with opposite effects: group resampling must widen the CI a lot.
    values = np.concatenate([np.full(50, 1.0), np.full(50, -1.0)])
    groups = ["a"] * 50 + ["b"] * 50
    grouped = paired_bootstrap_effect(np.zeros(100), values, groups, seed=3, n_resamples=400)
    ungrouped = paired_bootstrap_effect(np.zeros(100), values, seed=3, n_resamples=400)
    assert grouped["n_groups"] == 2
    assert (grouped["ci_upper"] - grouped["ci_lower"]) > (ungrouped["ci_upper"] - ungrouped["ci_lower"])


def test_paired_bootstrap_rejects_mismatched_inputs():
    with pytest.raises(ResearchIntegrityError, match="equal-length"):
        paired_bootstrap_effect([1.0, 2.0], [1.0], seed=1)
    with pytest.raises(ResearchIntegrityError, match="at least one pair"):
        paired_bootstrap_effect([], [], seed=1)


# --- full suite -----------------------------------------------------------------

def test_default_conditions_cover_the_required_control_set():
    names = {c.name for c in default_conditions()}
    assert names == {
        "baseline", "positive_steering", "negative_steering", "random_matched_norm",
        "orthogonal", "shuffled_label", "wrong_layer", "wrong_token_position",
        "activation_patch_mismatched_prompt",
    }


def test_suite_produces_the_expected_control_pattern():
    """Steering moves the endpoint; the controls that should not, do not."""
    df, activations, runner = _fixture()
    manifest = _manifest(df)
    directions = build_direction_set(activations, df, manifest, _config(), layer=READOUT_LAYER)
    held_out = df[df.sample_id.isin(manifest["test"])]
    result = run_causal_suite(runner, held_out, directions, ENDPOINT, layer=READOUT_LAYER,
                              strengths=[2.0], seed=5, n_resamples=200)
    # A patch has no dose, so it reports under a single key; steering conditions
    # report under the requested strength.
    effects = {name: next(iter(v["by_strength"].values()))["mean_difference"]
               for name, v in result["effects"].items()}
    assert effects["positive_steering"] > 0.05, effects
    assert effects["negative_steering"] < -0.05, effects
    # Sign reversal is the signature of a directional effect.
    assert np.sign(effects["positive_steering"]) != np.sign(effects["negative_steering"])
    # Applied where it cannot matter: exactly zero in this stub.
    assert effects["wrong_layer"] == pytest.approx(0.0)
    assert effects["wrong_token_position"] == pytest.approx(0.0)
    # The remaining controls are orthogonal or unrelated to the *probe*, which is
    # only ~0.99 aligned with the readout, so their residual readout component
    # leaves a small nonzero effect. The property that matters is that it is an
    # order of magnitude smaller than the steering effect.
    reference = abs(effects["positive_steering"])
    for control in ("orthogonal", "random_matched_norm", "shuffled_label"):
        assert abs(effects[control]) < 0.2 * reference, (control, effects)


def test_suite_runs_a_dose_response_grid():
    df, activations, runner = _fixture()
    manifest = _manifest(df)
    directions = build_direction_set(activations, df, manifest, _config(), layer=READOUT_LAYER)
    held_out = df[df.sample_id.isin(manifest["test"])]
    result = run_causal_suite(runner, held_out, directions, ENDPOINT, layer=READOUT_LAYER,
                              strengths=[0.5, 1.0, 4.0], seed=5, n_resamples=100)
    doses = result["effects"]["positive_steering"]["by_strength"]
    assert set(doses) == {"0.5", "1", "4"}
    magnitudes = [doses[k]["mean_difference"] for k in ("0.5", "1", "4")]
    assert magnitudes[0] < magnitudes[1] < magnitudes[2]
    # Relative magnitude is logged in units of the hidden-state norm.
    assert doses["4"]["mean_relative_magnitude"] > doses["0.5"]["mean_relative_magnitude"]


def test_baseline_is_evaluated_once_and_paired_by_sample_id():
    df, activations, runner = _fixture(n_groups=12)
    manifest = _manifest(df)
    directions = build_direction_set(activations, df, manifest, _config(), layer=READOUT_LAYER)
    held_out = df[df.sample_id.isin(manifest["test"])]
    result = run_causal_suite(runner, held_out, directions, ENDPOINT, layer=READOUT_LAYER,
                              strengths=[1.0, 2.0], seed=5, n_resamples=50)
    records = pd.DataFrame(result["records"])
    baseline = records[records.condition == "baseline"]
    assert len(baseline) == len(held_out)
    assert baseline.strength.unique().tolist() == [0.0]
    for name, entry in result["effects"].items():
        expected_doses = 1 if name == "activation_patch_mismatched_prompt" else 2
        assert len(entry["by_strength"]) == expected_doses, name
        for stats in entry["by_strength"].values():
            assert stats["n_pairs"] == len(held_out), name


def test_patching_uses_a_real_activation_from_a_different_prompt():
    df, activations, runner = _fixture(n_groups=12)
    manifest = _manifest(df)
    directions = build_direction_set(activations, df, manifest, _config(), layer=READOUT_LAYER)
    held_out = df[df.sample_id.isin(manifest["test"])]
    result = run_causal_suite(runner, held_out, directions, ENDPOINT, layer=READOUT_LAYER,
                              strengths=[1.0], seed=5, n_resamples=50)
    records = pd.DataFrame(result["records"])
    patch = records[records.condition == "activation_patch_mismatched_prompt"]
    assert (patch["kind"] == "replace").all()
    assert (patch["strength"] == 0.0).all()
    assert patch["direction_source"].iloc[0].startswith("hidden state from a different")
    # The patched vector is a real hidden state, so its norm is not 1.
    assert patch["direction_norm"].iloc[0] > 1.0


def test_every_record_logs_full_intervention_provenance():
    df, activations, runner = _fixture(n_groups=10)
    manifest = _manifest(df)
    directions = build_direction_set(activations, df, manifest, _config(), layer=READOUT_LAYER)
    held_out = df[df.sample_id.isin(manifest["test"])]
    result = run_causal_suite(runner, held_out, directions, ENDPOINT, layer=READOUT_LAYER,
                              strengths=[1.5], seed=5, n_resamples=50)
    for row in result["records"]:
        for key in ("sample_id", "group", "condition", "strength", "kind", "layer", "site",
                    "token_index", "timing_policy", "direction_name", "direction_source",
                    "hidden_norm", "endpoint_name", "endpoint_value", "chosen_option",
                    "valid_choice_mass"):
            assert key in row, key
    assert result["directions"]["probe"]["source"].endswith("train partition only")


def test_quality_controls_are_reported_beside_the_effects():
    df, activations, runner = _fixture(n_groups=10)
    manifest = _manifest(df)
    directions = build_direction_set(activations, df, manifest, _config(), layer=READOUT_LAYER)
    held_out = df[df.sample_id.isin(manifest["test"])]
    result = run_causal_suite(runner, held_out, directions, ENDPOINT, layer=READOUT_LAYER,
                              strengths=[1.0], seed=5, n_resamples=50)
    quality = result["quality_controls"]
    assert "baseline" in quality and "positive_steering" in quality
    assert quality["positive_steering"]["1"]["n"] == len(held_out)
    assert "mean_valid_choice_mass" in quality["positive_steering"]["1"]


def test_nuisance_condition_can_be_added_to_the_suite():
    df, activations, runner = _fixture()
    manifest = _manifest(df)
    directions = build_direction_set(activations, df, manifest, _config(),
                                     layer=READOUT_LAYER, nuisance_column="street")
    conditions = default_conditions() + (nuisance_condition("nuisance_street"),)
    held_out = df[df.sample_id.isin(manifest["test"])]
    result = run_causal_suite(runner, held_out, directions, ENDPOINT, layer=READOUT_LAYER,
                              strengths=[2.0], conditions=conditions, seed=5, n_resamples=50)
    assert "nuisance_steering_nuisance_street" in result["effects"]


def test_suite_requires_a_baseline_condition():
    df, activations, runner = _fixture(n_groups=8)
    directions = build_direction_set(activations, df, _manifest(df), _config(), layer=READOUT_LAYER)
    only_steer = (Condition("positive_steering", "add", "probe"),)
    with pytest.raises(ResearchIntegrityError, match="baseline condition is required"):
        run_causal_suite(runner, df.head(4), directions, ENDPOINT, layer=READOUT_LAYER,
                         strengths=[1.0], conditions=only_steer)


def test_suite_validates_its_inputs():
    df, activations, runner = _fixture(n_groups=8)
    directions = build_direction_set(activations, df, _manifest(df), _config(), layer=READOUT_LAYER)
    with pytest.raises(ResearchIntegrityError, match="at least one held-out prompt"):
        run_causal_suite(runner, df.head(0), directions, ENDPOINT, layer=READOUT_LAYER, strengths=[1.0])
    with pytest.raises(ResearchIntegrityError, match="at least one intervention strength"):
        run_causal_suite(runner, df.head(4), directions, ENDPOINT, layer=READOUT_LAYER, strengths=[])
    with pytest.raises(ResearchIntegrityError, match="sample_id and statement"):
        run_causal_suite(runner, df[["label"]].head(4), directions, ENDPOINT,
                         layer=READOUT_LAYER, strengths=[1.0])


def test_missing_direction_for_a_condition_is_a_hard_error():
    df, activations, runner = _fixture(n_groups=8)
    directions = build_direction_set(activations, df, _manifest(df), _config(), layer=READOUT_LAYER)
    conditions = (Condition("baseline", "none"), Condition("ghost", "add", "does_not_exist"))
    with pytest.raises(ResearchIntegrityError, match="which was not built"):
        run_causal_suite(runner, df.head(4), directions, ENDPOINT, layer=READOUT_LAYER,
                         strengths=[1.0], conditions=conditions)


def test_wrong_layer_offset_below_zero_is_refused():
    df, activations, runner = _fixture(n_groups=8)
    directions = build_direction_set(activations, df, _manifest(df), _config(), layer=1)
    conditions = (Condition("baseline", "none"), Condition("wrong_layer", "add", "probe", layer_offset=-9))
    with pytest.raises(ResearchIntegrityError, match="negative layer"):
        run_causal_suite(runner, df.head(4), directions, ENDPOINT, layer=1,
                         strengths=[1.0], conditions=conditions)


def test_suite_is_reproducible():
    df, activations, runner = _fixture(n_groups=12)
    manifest = _manifest(df)
    directions = build_direction_set(activations, df, manifest, _config(), layer=READOUT_LAYER)
    held_out = df[df.sample_id.isin(manifest["test"])]
    kwargs = dict(layer=READOUT_LAYER, strengths=[1.0], seed=5, n_resamples=50)
    first = run_causal_suite(_runner(), held_out, directions, ENDPOINT, **kwargs)
    second = run_causal_suite(_runner(), held_out, directions, ENDPOINT, **kwargs)
    assert first["effects"] == second["effects"]


# --- generation quality ---------------------------------------------------------

def test_generation_quality_flags_refusal_and_repetition():
    assert score_generation_quality("I cannot help with that")["refusal_like"] is True
    assert score_generation_quality("Yes")["refusal_like"] is False
    repeated = score_generation_quality("bet bet bet bet")
    assert repeated["repetition_diversity"] < 0.5
    assert score_generation_quality("   ")["empty"] is True
