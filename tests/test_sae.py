"""Tests for the defensible unsupervised SAE path.

The legacy module's defects (see docs/SAE_AUDIT.md) were mostly invisible: fake
weight tying that discarded the decoder's update every step, and parameter
mutation inside forward() that made a "frozen" autoencoder not frozen. These
tests are written to catch that class of bug by assertion rather than by reading.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deception_circuits.paper import PaperConfig, ResearchIntegrityError
from deception_circuits.paper_sae import (SAEConfig, SparseAutoencoder, apply_input_normalizer,
                                          evaluate_features_on_test, feature_activations,
                                          fit_input_normalizer, rank_features, run_sae_experiment,
                                          sae_diagnostics, seed_stability, top_activating_examples,
                                          train_sae)

DIM = 24
N_TRUE_ATOMS = 8


def _sparse_data(n: int = 900, seed: int = 0):
    """Data with genuine sparse structure: 8 unit atoms, 2 active per sample."""
    rng = np.random.default_rng(seed)
    atoms = rng.normal(size=(N_TRUE_ATOMS, DIM))
    atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
    x = np.zeros((n, DIM))
    active_atoms = []
    for i in range(n):
        picked = rng.choice(N_TRUE_ATOMS, size=2, replace=False)
        x[i] = (rng.uniform(1.0, 3.0, size=2)[:, None] * atoms[picked]).sum(0)
        active_atoms.append(sorted(picked))
    x += rng.normal(scale=0.05, size=x.shape)
    return x, np.array(active_atoms)


def _config(**overrides) -> SAEConfig:
    base = dict(n_features=32, activation="topk", k=2, epochs=20, batch_size=64, seed=0)
    base.update(overrides)
    return SAEConfig(**base)


# --- config validation ----------------------------------------------------------

def test_n_features_must_be_explicit_and_sane():
    with pytest.raises(ResearchIntegrityError, match="stated explicitly"):
        SAEConfig(n_features=1)


def test_topk_requires_an_explicit_k():
    with pytest.raises(ResearchIntegrityError, match="requires an explicit k"):
        SAEConfig(n_features=32, activation="topk")
    with pytest.raises(ResearchIntegrityError, match=r"k must be in \[1, n_features\]"):
        SAEConfig(n_features=32, activation="topk", k=64)


def test_unknown_activation_is_refused():
    with pytest.raises(ResearchIntegrityError, match="activation must be one of"):
        SAEConfig(n_features=32, activation="sigmoid")


def test_expansion_factor_is_recorded():
    assert _config(n_features=48).expansion_factor(24) == pytest.approx(2.0)


# --- decoder norm, tying, and code sign -----------------------------------------

def test_decoder_columns_are_unit_norm_after_training():
    x, _ = _sparse_data()
    model, _ = train_sae(x[:600], x[600:750], _config())
    norms = np.linalg.norm(model.decoder_matrix().detach().numpy(), axis=0)
    assert np.allclose(norms, 1.0, atol=1e-6), (norms.min(), norms.max())


def test_tied_weights_share_a_single_parameter():
    """The legacy module copied weights in-place; real tying has no second matrix."""
    model = SparseAutoencoder(DIM, _config(tied_weights=True))
    assert model.decoder_weight is None
    names = {n for n, _ in model.named_parameters()}
    assert names == {"encoder_weight", "encoder_bias", "decoder_bias"}
    # The decoder matrix is a view of the encoder, so they cannot diverge.
    assert torch.allclose(model.decoder_matrix(), model.encoder_weight.t())


def test_tied_weights_receive_gradient_through_the_decoder():
    """The legacy no_grad copy severed this path; assert it exists."""
    model = SparseAutoencoder(DIM, _config(tied_weights=True, activation="relu"))
    x = torch.randn(16, DIM)
    loss, _ = model.loss(x)
    loss.backward()
    assert model.encoder_weight.grad is not None
    assert float(model.encoder_weight.grad.abs().sum()) > 0


def test_untied_weights_have_an_independent_decoder():
    model = SparseAutoencoder(DIM, _config(tied_weights=False))
    assert model.decoder_weight is not None
    assert not torch.allclose(model.decoder_matrix(), model.encoder_weight.t())


def test_sparse_code_is_non_negative():
    """The legacy top-k selected on abs() and let negatives through."""
    x, _ = _sparse_data(n=200)
    for activation, extra in (("topk", {"k": 3}), ("relu", {})):
        model = SparseAutoencoder(DIM, _config(activation=activation, **extra))
        with torch.no_grad():
            _, code = model.encode(torch.from_numpy(x.astype(np.float32)))
        assert float(code.min()) >= 0.0, activation


def test_topk_activates_exactly_k_features():
    x, _ = _sparse_data(n=200)
    model = SparseAutoencoder(DIM, _config(k=3))
    _, code = model.encode(torch.from_numpy(x.astype(np.float32)))
    counts = (code > 0).sum(dim=-1).numpy()
    # At most k; fewer only if ReLU zeroed some of the top-k pre-activations.
    assert counts.max() <= 3
    assert counts.mean() > 0


def test_l1_penalty_does_not_scale_with_width():
    """Summed over features, meaned over batch: doubling width must not halve it."""
    x = torch.randn(32, DIM)
    narrow = SparseAutoencoder(DIM, _config(n_features=32, activation="relu"))
    wide = SparseAutoencoder(DIM, _config(n_features=64, activation="relu"))
    _, narrow_parts = narrow.loss(x)
    _, wide_parts = wide.loss(x)
    # A width-dependent mean would make the wide model's l1 roughly unchanged;
    # a correctly summed penalty grows with the number of active features.
    assert wide_parts["l1"] > 1.2 * narrow_parts["l1"]


# --- freezing -------------------------------------------------------------------

def test_frozen_sae_is_unchanged_by_evaluation():
    """The legacy forward() mutated parameters, so 'frozen' meant nothing."""
    x, _ = _sparse_data()
    model, training = train_sae(x[:600], x[600:750], _config())
    before = model.parameter_fingerprint()
    for _ in range(3):
        sae_diagnostics(model, x[750:], training["normalizer"], partition="test")
        feature_activations(model, x[750:], training["normalizer"])
    assert model.parameter_fingerprint() == before


def test_freeze_clears_requires_grad():
    model = SparseAutoencoder(DIM, _config()).freeze()
    assert all(not p.requires_grad for p in model.parameters())
    assert not model.training


def test_train_sae_returns_a_frozen_model():
    x, _ = _sparse_data()
    model, _ = train_sae(x[:600], x[600:750], _config())
    assert all(not p.requires_grad for p in model.parameters())


# --- normalization --------------------------------------------------------------

def test_normalizer_is_fitted_on_train_only_and_rescales_to_sqrt_d():
    x, _ = _sparse_data()
    stats = fit_input_normalizer(x[:600])
    assert stats["n_train"] == 600
    assert stats["fitted_on"] == "train partition only"
    normalized = apply_input_normalizer(x[:600], stats)
    assert float(np.linalg.norm(normalized, axis=1).mean()) == pytest.approx(np.sqrt(DIM), rel=1e-3)


def test_normalizer_refuses_constant_activations():
    with pytest.raises(ResearchIntegrityError, match="constant"):
        fit_input_normalizer(np.ones((10, 4)))


def test_normalizer_can_be_disabled():
    x, _ = _sparse_data(n=300)
    _, training = train_sae(x[:200], x[200:260], _config(normalize_inputs=False, epochs=3, batch_size=32))
    assert training["normalizer"] is None


# --- diagnostics ----------------------------------------------------------------

def test_diagnostics_report_every_required_quantity():
    x, _ = _sparse_data()
    model, training = train_sae(x[:600], x[600:750], _config())
    d = sae_diagnostics(model, x[750:], training["normalizer"], partition="test")
    for key in ("reconstruction_mse", "fraction_variance_explained",
                "fraction_variance_unexplained", "l0_mean", "l0_std",
                "dead_feature_fraction", "n_dead_features", "activation_frequency",
                "expansion_factor", "n_features", "input_dim", "n_samples"):
        assert key in d, key
    assert 0.0 <= d["fraction_variance_explained"] <= 1.0
    assert d["l0_mean"] == pytest.approx(2.0, abs=0.01)  # k = 2
    assert 0.0 <= d["dead_feature_fraction"] <= 1.0


def test_sae_reconstructs_sparse_structure_better_than_chance():
    x, _ = _sparse_data()
    model, training = train_sae(x[:600], x[600:750], _config(epochs=40))
    d = sae_diagnostics(model, x[750:], training["normalizer"])
    assert d["fraction_variance_explained"] > 0.5, d


def test_epoch_is_selected_on_validation_not_train():
    x, _ = _sparse_data()
    _, training = train_sae(x[:600], x[600:750], _config())
    assert training["selected_on"] == "validation reconstruction MSE"
    assert training["labels_used_in_training"] is False
    best = min(training["history"], key=lambda h: h["validation_mse"])
    assert training["selected_epoch"] == best["epoch"]


def test_training_refuses_a_batch_larger_than_the_training_set():
    x, _ = _sparse_data(n=100)
    with pytest.raises(ResearchIntegrityError, match="fewer than batch_size"):
        train_sae(x[:32], x[32:64], _config(batch_size=64))


# --- ranking and held-out evaluation -------------------------------------------

def _labelled_fixture(n_groups: int = 120):
    """Activations whose sparse structure correlates with a label."""
    rng = np.random.default_rng(1)
    atoms = rng.normal(size=(N_TRUE_ATOMS, DIM))
    atoms /= np.linalg.norm(atoms, axis=1, keepdims=True)
    rows, stack = [], []
    for group in range(n_groups):
        for label in (0, 1):
            # Label 1 always uses atom 0; label 0 never does.
            picked = [0, 1 + rng.integers(0, N_TRUE_ATOMS - 1)] if label else \
                     list(rng.choice(np.arange(1, N_TRUE_ATOMS), size=2, replace=False))
            vector = (rng.uniform(1.0, 3.0, size=2)[:, None] * atoms[list(picked)]).sum(0)
            vector += rng.normal(scale=0.05, size=DIM)
            stack.append(np.stack([vector, rng.normal(size=DIM)]))
            rows.append({"sample_id": f"s{group}-{label}", "base_item_id": group,
                         "split_group_id": group, "statement": f"hand {group} variant {label}",
                         "response": "raise", "label": label, "scenario": "poker",
                         "street": ("flop", "turn", "river")[group % 3]})
    df = pd.DataFrame(rows)
    ids = df.sample_id.tolist()
    manifest = {"group_column": "split_group_id", "train": ids[:160],
                "validation": ids[160:200], "test": ids[200:]}
    return df, np.stack(stack), manifest


def test_ranking_finds_the_label_correlated_feature():
    df, activations, manifest = _labelled_fixture()
    model, training = train_sae(activations[:160, 0], activations[160:200, 0],
                                _config(n_features=32, k=2, epochs=40))
    ranking = rank_features(model, activations[:200, 0], df.label.to_numpy()[:200],
                            training["normalizer"])
    top = ranking[ranking.status == "ranked"].iloc[0]
    assert top["abs_auroc_gap"] > 0.15, ranking.head().to_dict("records")


def test_ranking_records_rare_features_instead_of_dropping_them():
    df, activations, manifest = _labelled_fixture()
    model, training = train_sae(activations[:160, 0], activations[160:200, 0], _config())
    ranking = rank_features(model, activations[:200, 0], df.label.to_numpy()[:200],
                            training["normalizer"], min_activation_frequency=0.5)
    assert (ranking.status == "too_rare").any()
    assert len(ranking) == model.n_features
    assert ranking[ranking.status == "too_rare"].auroc.isna().all()


def test_ranking_requires_both_label_classes():
    df, activations, _ = _labelled_fixture()
    model, training = train_sae(activations[:160, 0], activations[160:200, 0], _config())
    with pytest.raises(ResearchIntegrityError, match="both label classes"):
        rank_features(model, activations[:160, 0], np.zeros(160, dtype=int), training["normalizer"])


def test_held_out_evaluation_does_not_modify_the_sae():
    df, activations, manifest = _labelled_fixture()
    model, training = train_sae(activations[:160, 0], activations[160:200, 0], _config())
    before = model.parameter_fingerprint()
    result = evaluate_features_on_test(model, activations[200:, 0], df.label.to_numpy()[200:],
                                       training["normalizer"], [0, 1, 2])
    assert result["sae_unchanged_by_evaluation"] is True
    assert model.parameter_fingerprint() == before
    assert result["selection_partition"] == "train+validation"
    assert result["evaluation_partition"] == "test"


def test_held_out_evaluation_requires_selected_features():
    df, activations, _ = _labelled_fixture()
    model, training = train_sae(activations[:160, 0], activations[160:200, 0], _config())
    with pytest.raises(ResearchIntegrityError, match="No features were selected"):
        evaluate_features_on_test(model, activations[200:, 0], df.label.to_numpy()[200:],
                                  training["normalizer"], [])


# --- interpretation exports -----------------------------------------------------

def test_feature_examples_carry_metadata_and_no_semantic_label():
    df, activations, manifest = _labelled_fixture()
    model, training = train_sae(activations[:200, 0], activations[160:200, 0], _config())
    report = top_activating_examples(model, activations[:200, 0], df.head(200).reset_index(drop=True),
                                     training["normalizer"], 0, k=5, metadata_columns=["street"])
    assert report["interpretation"] == "not assigned; requires human inspection"
    assert len(report["top_examples"]) <= 5
    assert report["contrast_examples"]
    for example in report["top_examples"]:
        assert "sample_id" in example and "label" in example and "street" in example
        assert "statement_preview" in example
        assert len(example["statement_preview"]) <= 240
    assert "label_means" in report
    # Must be JSON-serialisable: no numpy scalars leak into the artifact.
    json.dumps(report)


def test_feature_index_out_of_range_is_refused():
    df, activations, _ = _labelled_fixture()
    model, training = train_sae(activations[:160, 0], activations[160:200, 0], _config())
    with pytest.raises(ResearchIntegrityError, match="out of range"):
        top_activating_examples(model, activations[:160, 0], df.head(160).reset_index(drop=True),
                                training["normalizer"], 9999)


# --- seed stability -------------------------------------------------------------

def test_seed_stability_reports_pairwise_dictionary_similarity():
    x, _ = _sparse_data()
    result = seed_stability(x[:600], x[600:750], _config(epochs=8), seeds=[0, 1, 2])
    assert result["n_pairs"] == 3
    assert 0.0 <= result["mean_max_cosine_across_pairs"] <= 1.0
    for pair in result["pairs"]:
        assert set(pair) == {"seeds", "mean_max_cosine", "median_max_cosine", "fraction_above_0.9"}


def test_seed_stability_needs_at_least_two_seeds():
    x, _ = _sparse_data(n=300)
    with pytest.raises(ResearchIntegrityError, match="at least two seeds"):
        seed_stability(x[:200], x[200:260], _config(epochs=2, batch_size=32), seeds=[0])


# --- orchestration --------------------------------------------------------------

def _paper_config(**overrides) -> PaperConfig:
    base = dict(experiment_name="sae", dataset_path="x.csv", activation_dir="a",
                output_dir="o", seed=3)
    base.update(overrides)
    return PaperConfig(**base)


def test_experiment_separates_unsupervised_training_from_labelled_ranking(tmp_path):
    df, activations, manifest = _labelled_fixture()
    result = run_sae_experiment(activations, df, manifest, _paper_config(),
                                _config(epochs=25), layer=0, top_n_features=3,
                                metadata_columns=["street"], output_dir=tmp_path)
    assert result["unsupervised"] is True
    assert "after the SAE was frozen" in result["labels_used_for"]
    assert result["training"]["labels_used_in_training"] is False
    assert result["sae_frozen_before_evaluation"] is True
    assert result["sae_unchanged_after_evaluation"] is True
    assert set(result["diagnostics"]) == {"train", "validation", "test"}
    assert len(result["selected_features"]) == 3
    assert result["held_out_feature_evaluation"]["evaluation_partition"] == "test"
    assert len(result["feature_examples"]) == 3
    json.dumps(result)


def test_experiment_writes_real_artifacts(tmp_path):
    df, activations, manifest = _labelled_fixture()
    result = run_sae_experiment(activations, df, manifest, _paper_config(),
                                _config(epochs=8), layer=0, top_n_features=2,
                                output_dir=tmp_path)
    for key, path in result["artifacts"].items():
        assert Path(path).is_file(), key
    saved = torch.load(tmp_path / "sae_model.pt", weights_only=False)
    assert saved["unsupervised"] is True
    assert saved["sae_config"]["n_features"] == 32
    assert saved["layer"] == 0
    ranking = pd.read_csv(tmp_path / "sae_feature_ranking.csv")
    assert len(ranking) == 32


def test_experiment_requires_a_validation_partition():
    df, activations, manifest = _labelled_fixture()
    manifest = {**manifest, "validation": []}
    with pytest.raises(ResearchIntegrityError, match="non-empty validation partition"):
        run_sae_experiment(activations, df, manifest, _paper_config(), _config(epochs=2), layer=0)


def test_experiment_includes_seed_stability_when_requested(tmp_path):
    df, activations, manifest = _labelled_fixture()
    result = run_sae_experiment(activations, df, manifest, _paper_config(),
                                _config(epochs=5), layer=0, top_n_features=2,
                                stability_seeds=[0, 1], output_dir=tmp_path)
    assert result["seed_stability"]["n_pairs"] == 1
