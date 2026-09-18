"""Integrity tests for leakage-safe, provenance-tracked activation extraction.

Every test here uses the explicitly named stub provider below.  No model is
downloaded, no network call is made, and no API is contacted.
"""
from pathlib import Path
import json
import sys

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deception_circuits.paper import (PaperConfig, ResearchIntegrityError,
                                      load_activations, validate_dataset)
from deception_circuits.paper_extraction import (
    ExtractionSpec,
    TransformersHiddenStateProvider,
    audit_activation_provenance,
    extract_sample,
    load_activation_layer_indices,
    render_extraction_prompt,
    run_extraction,
    verify_extraction_manifest,
)

from _stubs import StubHiddenStateProvider as StubProvider

MODEL = "stub/tiny-test-model"


def _dataset(tmp_path: Path, n_groups: int = 20) -> Path:
    rows = []
    for group in range(n_groups):
        for label in (0, 1):
            sid = f"{group}-{label}"
            rows.append({
                "sample_id": sid, "base_item_id": group, "split_group_id": group,
                "statement": f"hand {group} street flop facing bet",
                # A distinctive token that must never appear in a pre-decision prompt.
                "response": f"UNIQUERESPONSETOKEN{label}",
                "label": label, "scenario": "poker",
            })
    csv = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _spec(**overrides) -> ExtractionSpec:
    overrides.setdefault("prompt_template_id", "poker_action_v1")
    return ExtractionSpec(subject_model=MODEL, **overrides)


# --- leakage: response tokens cannot enter the pre-decision path ----------------

@pytest.mark.parametrize("mode", ["prompt_end", "decision_token_prelogit", "mean_prompt"])
def test_pre_decision_prompt_never_contains_response_text(tmp_path, mode):
    df = pd.read_csv(_dataset(tmp_path, n_groups=3))
    template = "{statement}\nAction:" if mode != "mean_prompt" else "{statement}"
    spec = _spec(activation_mode=mode, prompt_template=template)
    provider = StubProvider()
    for _, row in df.iterrows():
        rendering = render_extraction_prompt(row, spec, provider)
        assert "UNIQUERESPONSETOKEN" not in rendering.text
        assert str(row["response"]) not in rendering.text
    # Nothing the provider ever saw contained response text either.
    assert all("UNIQUERESPONSETOKEN" not in seen for seen in provider.encoded)


@pytest.mark.parametrize("mode", ["prompt_end", "decision_token_prelogit", "mean_prompt"])
def test_pre_decision_spec_rejects_a_response_referencing_template(mode):
    with pytest.raises(ResearchIntegrityError, match="pre-decision"):
        _spec(activation_mode=mode, prompt_template="{statement}\nAction:{response}")


def test_pre_decision_spec_rejects_leakage_optin():
    with pytest.raises(ResearchIntegrityError, match="allow_response_leakage"):
        _spec(activation_mode="prompt_end", allow_response_leakage=True)


def test_prompt_end_rejects_a_template_suffix_that_misses_the_boundary():
    with pytest.raises(ResearchIntegrityError, match="must end at the explicit decision boundary"):
        _spec(prompt_template="{statement}\nWhat now?")


def test_prompt_end_enforces_the_boundary_on_the_rendered_prompt(tmp_path):
    """A suffix-less template is allowed, but the rendered row must still end
    at the boundary -- otherwise the read position is not the decision point."""
    df = pd.read_csv(_dataset(tmp_path, n_groups=1))
    spec = _spec(prompt_template="Question: {statement}")
    with pytest.raises(ResearchIntegrityError, match="does not end at the declared decision boundary"):
        render_extraction_prompt(df.iloc[0], spec, StubProvider())


def test_boundary_can_come_from_the_data_rather_than_the_template(tmp_path):
    """A prompt whose own final line is the decision needs no appended marker."""
    df = pd.DataFrame([{"sample_id": "a", "statement": "state\nThe player decided to: Raise.",
                        "response": "x", "label": 1}])
    spec = _spec(prompt_template="{statement}", decision_boundary_marker="Raise.")
    rendering = render_extraction_prompt(df.iloc[0], spec, StubProvider())
    assert rendering.text.endswith("Raise.")
    assert rendering.token_index == rendering.n_tokens - 1


def test_response_token_mode_is_opt_in_only():
    with pytest.raises(ResearchIntegrityError, match="diagnostic"):
        _spec(activation_mode="response_token", prompt_template="{statement}\nAction:{response}")
    # Opted in, it works but is flagged as not pre-decision.
    spec = _spec(activation_mode="response_token", prompt_template="{statement}\nAction:{response}",
                 allow_response_leakage=True)
    assert spec.is_pre_decision is False


def test_response_token_reads_a_position_after_the_prompt_boundary(tmp_path):
    df = pd.read_csv(_dataset(tmp_path, n_groups=1))
    spec = _spec(activation_mode="response_token", prompt_template="{statement}\nAction: {response}",
                 allow_response_leakage=True)
    rendering = render_extraction_prompt(df.iloc[0], spec, StubProvider())
    assert "UNIQUERESPONSETOKEN" in rendering.text
    assert rendering.token_index >= rendering.prompt_span[1]


# --- token index provenance -----------------------------------------------------

def test_prompt_end_token_index_is_the_final_prompt_token(tmp_path):
    df = pd.read_csv(_dataset(tmp_path, n_groups=1))
    row = df.iloc[0]
    spec = _spec()
    provider = StubProvider()
    rendering = render_extraction_prompt(row, spec, provider)
    expected_tokens = provider.encode(f"{row['statement']}\nAction:")
    assert rendering.n_tokens == len(expected_tokens)
    assert rendering.token_index == len(expected_tokens) - 1
    # The stub encodes position into channel 1, so the tensor proves which token was read.
    tensor, rendering, layers = extract_sample(row, spec, provider)
    assert tensor.shape == (4, 3)
    assert tensor[0, 1].item() == pytest.approx(rendering.token_index)
    assert tensor[0, 0].item() == pytest.approx(expected_tokens[-1])
    assert layers == [0, 1, 2, 3]


def test_mean_prompt_pools_over_the_prompt_span(tmp_path):
    df = pd.read_csv(_dataset(tmp_path, n_groups=1))
    spec = _spec(activation_mode="mean_prompt", prompt_template="{statement}")
    tensor, rendering, _ = extract_sample(df.iloc[0], spec, StubProvider())
    assert rendering.token_index is None
    n = rendering.n_tokens
    # Mean of positions 0..n-1 in channel 1.
    assert tensor[0, 1].item() == pytest.approx((n - 1) / 2)


def test_truncation_that_would_move_the_decision_boundary_is_refused(tmp_path):
    df = pd.read_csv(_dataset(tmp_path, n_groups=1))
    spec = _spec(max_sequence_length=2)
    with pytest.raises(ResearchIntegrityError, match="truncation would move the decision boundary"):
        render_extraction_prompt(df.iloc[0], spec, StubProvider())


def test_configured_layer_subset_is_honored(tmp_path):
    df = pd.read_csv(_dataset(tmp_path, n_groups=1))
    spec = _spec(layer_indices=(1, 3))
    tensor, _, layers = extract_sample(df.iloc[0], spec, StubProvider())
    assert layers == [1, 3]
    assert tensor.shape == (2, 3)
    with pytest.raises(ResearchIntegrityError, match="exceed the model's"):
        extract_sample(df.iloc[0], _spec(layer_indices=(0, 99)), StubProvider())

def test_activation_layer_mapping_preserves_physical_layer_numbers(tmp_path):
    """Stored tensor axis positions must map back to real transformer layers."""
    df = pd.read_csv(_dataset(tmp_path, n_groups=2))
    out = tmp_path / "acts"
    spec = _spec(layer_indices=(1, 3))

    manifest = run_extraction(df, spec, out, StubProvider())

    assert manifest["activation_shape"] == [2, 3]
    assert load_activation_layer_indices(out) == [1, 3]

# --- resumability and mismatch refusal ------------------------------------------

def test_extraction_resumes_sample_wise(tmp_path):
    df = pd.read_csv(_dataset(tmp_path, n_groups=4))
    out = tmp_path / "acts"
    spec = _spec()
    first = run_extraction(df.head(3), spec, out, StubProvider())
    assert first["n_newly_extracted"] == 3 and first["n_resumed"] == 0
    second = run_extraction(df, spec, out, StubProvider())
    assert second["n_resumed"] == 3
    assert second["n_newly_extracted"] == len(df) - 3
    assert second["n_samples"] == len(df)
    # A deleted artifact is re-extracted rather than trusted from the journal.
    (out / "sample_0-0.pt").unlink()
    third = run_extraction(df, spec, out, StubProvider())
    assert third["n_newly_extracted"] == 1 and third["n_resumed"] == len(df) - 1

def test_resume_reextracts_when_prompt_content_changes(tmp_path):
    """A sample_id alone must never authorize reuse after its prompt changed."""
    df = pd.read_csv(_dataset(tmp_path, n_groups=3))
    out = tmp_path / "acts"
    spec = _spec()

    first = run_extraction(df, spec, out, StubProvider())
    assert first["n_newly_extracted"] == len(df)

    changed = df.copy()
    changed.loc[0, "statement"] = (
        str(changed.loc[0, "statement"]) + " CHANGED_PROMPT_CONTENT"
    )

    second = run_extraction(changed, spec, out, StubProvider())

    assert second["n_newly_extracted"] == 1
    assert second["n_resumed"] == len(df) - 1

    changed_sid = str(changed.loc[0, "sample_id"])

    original_record = next(
        record
        for record in first["samples"]
        if str(record["sample_id"]) == changed_sid
    )
    changed_record = next(
        record
        for record in second["samples"]
        if str(record["sample_id"]) == changed_sid
    )

    assert changed_record["prompt_sha256"] != original_record["prompt_sha256"]


def test_resume_reextracts_when_activation_artifact_hash_changes(tmp_path):
    """A modified tensor file must not be silently trusted during resume."""
    df = pd.read_csv(_dataset(tmp_path, n_groups=3))
    out = tmp_path / "acts"
    spec = _spec()

    first = run_extraction(df, spec, out, StubProvider())
    assert first["n_newly_extracted"] == len(df)

    sid = str(df.iloc[0]["sample_id"])
    artifact = out / f"sample_{sid}.pt"

    original_tensor = torch.load(
        artifact,
        map_location="cpu",
        weights_only=True,
    ).clone()

    torch.save(torch.full((4, 3), 999.0), artifact)

    second = run_extraction(df, spec, out, StubProvider())

    assert second["n_newly_extracted"] == 1
    assert second["n_resumed"] == len(df) - 1

    repaired_tensor = torch.load(
        artifact,
        map_location="cpu",
        weights_only=True,
    )

    repaired_record = next(
        record
        for record in second["samples"]
        if str(record["sample_id"]) == sid
    )

    assert torch.equal(repaired_tensor, original_tensor)
    assert "artifact_sha256" in repaired_record
    assert len(repaired_record["artifact_sha256"]) == 64

def test_extraction_refuses_to_overwrite_a_different_spec(tmp_path):
    df = pd.read_csv(_dataset(tmp_path, n_groups=3))
    out = tmp_path / "acts"
    run_extraction(df, _spec(), out, StubProvider())
    conflicting = _spec(activation_site="mlp")
    with pytest.raises(ResearchIntegrityError, match="different extraction spec"):
        run_extraction(df, conflicting, out, StubProvider())


def test_manifest_records_full_provenance(tmp_path):
    df = pd.read_csv(_dataset(tmp_path, n_groups=3))
    out = tmp_path / "acts"
    manifest = run_extraction(df, _spec(model_revision="abc123"), out, StubProvider())
    assert manifest["is_pre_decision"] is True
    assert manifest["activation_shape"] == [4, 3]
    record = manifest["samples"][0]
    for key in ("sample_id", "token_index", "n_prompt_tokens", "layer_indices", "shape",
                "subject_model", "model_revision", "tokenizer_revision", "activation_site",
                "prompt_template_id", "max_sequence_length", "mode", "prompt_sha256",
                "spec_fingerprint", "truncated"):
        assert key in record, key
    assert record["subject_model"] == MODEL
    assert record["model_revision"] == "abc123"
    # The journal is append-only and one line per extracted sample.
    lines = (out / "extraction_manifest.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(df)


# --- config/manifest agreement in the strict loader ------------------------------

def _config(csv: Path, acts: Path, tmp_path: Path, **overrides) -> PaperConfig:
    base = dict(experiment_name="t", dataset_path=str(csv), activation_dir=str(acts),
                output_dir=str(tmp_path / "out"), subject_model=MODEL,
                prompt_template_id="poker_action_v1", activation_site="residual_stream",
                activation_mode="prompt_end", n_seeds=2, bootstrap_resamples=30)
    base.update(overrides)
    return PaperConfig(**base)


def test_load_activations_requires_a_provenance_manifest(tmp_path):
    csv = _dataset(tmp_path, n_groups=3)
    df = validate_dataset(csv)
    acts = tmp_path / "acts"
    acts.mkdir()
    for sid in df["sample_id"]:
        torch.save(torch.zeros(4, 3), acts / f"sample_{sid}.pt")
    with pytest.raises(FileNotFoundError, match="provenance manifest"):
        load_activations(df, acts)
    # The tensor-level checks remain reachable for unit testing.
    assert load_activations(df, acts, require_manifest=False).shape == (len(df), 4, 3)


def test_load_activations_rejects_mode_mismatch_between_config_and_artifacts(tmp_path):
    csv = _dataset(tmp_path, n_groups=3)
    df = validate_dataset(csv)
    acts = tmp_path / "acts"
    diagnostic = _spec(activation_mode="response_token",
                       prompt_template="{statement}\nAction: {response}",
                       allow_response_leakage=True)
    run_extraction(df, diagnostic, acts, StubProvider())
    config = _config(csv, acts, tmp_path, activation_mode="prompt_end")
    with pytest.raises(ResearchIntegrityError, match="do not match this config"):
        load_activations(df, acts, config)


def test_load_activations_rejects_an_incomplete_manifest(tmp_path):
    csv = _dataset(tmp_path, n_groups=4)
    df = validate_dataset(csv)
    acts = tmp_path / "acts"
    run_extraction(df.head(4), _spec(), acts, StubProvider())
    with pytest.raises(ResearchIntegrityError, match="missing .* dataset samples"):
        load_activations(df, acts, _config(csv, acts, tmp_path))


def test_audit_flags_a_response_token_diagnostic_as_not_primary(tmp_path):
    csv = _dataset(tmp_path, n_groups=3)
    df = validate_dataset(csv)
    acts = tmp_path / "acts"
    diagnostic = _spec(activation_mode="response_token",
                       prompt_template="{statement}\nAction: {response}",
                       allow_response_leakage=True)
    run_extraction(df, diagnostic, acts, StubProvider())
    config = _config(csv, acts, tmp_path, activation_mode="response_token")
    failures = audit_activation_provenance(df, acts, config)
    assert any("diagnostic" in f for f in failures)


def test_audit_passes_on_a_complete_pre_decision_extraction(tmp_path):
    csv = _dataset(tmp_path, n_groups=3)
    df = validate_dataset(csv)
    acts = tmp_path / "acts"
    run_extraction(df, _spec(), acts, StubProvider())
    assert audit_activation_provenance(df, acts, _config(csv, acts, tmp_path)) == []


def test_audit_detects_a_deleted_artifact(tmp_path):
    csv = _dataset(tmp_path, n_groups=3)
    df = validate_dataset(csv)
    acts = tmp_path / "acts"
    run_extraction(df, _spec(), acts, StubProvider())
    (acts / "sample_0-0.pt").unlink()
    failures = audit_activation_provenance(df, acts, _config(csv, acts, tmp_path))
    assert any("missing" in f.lower() for f in failures)

def test_load_activations_rejects_tampered_artifact_hash(tmp_path):
    """Normal analysis must reject an artifact modified after extraction."""
    csv = _dataset(tmp_path, n_groups=3)
    df = validate_dataset(csv)
    acts = tmp_path / "acts"

    run_extraction(df, _spec(), acts, StubProvider())

    sid = str(df.iloc[0]["sample_id"])
    artifact = acts / f"sample_{sid}.pt"
    torch.save(torch.full((4, 3), 999.0), artifact)

    with pytest.raises(ResearchIntegrityError, match="artifact hash mismatch"):
        load_activations(df, acts, _config(csv, acts, tmp_path))

def test_spec_from_paper_config_maps_residual_stream_and_layer_selection(tmp_path):
    csv = _dataset(tmp_path, n_groups=3)
    acts = tmp_path / "acts"
    spec = ExtractionSpec.from_paper_config(
        _config(csv, acts, tmp_path, activation_layers="all"))
    assert spec.activation_site == "block"
    assert spec.layer_indices is None
    spec = ExtractionSpec.from_paper_config(
        _config(csv, acts, tmp_path, activation_layers="3, 1, 1"))
    assert spec.layer_indices == (1, 3)


def test_placeholder_subject_model_is_refused():
    with pytest.raises(ResearchIntegrityError, match="real model identifier"):
        ExtractionSpec(subject_model="REPLACE_WITH_EXACT_MODEL_ID")


def test_shipped_real_configs_build_a_valid_extraction_spec():
    """The committed run configs must survive exactly what the CLI does to them.

    prompt_template was previously unreachable from a config, so every real config
    with a non-default decision boundary hard-failed at spec construction.
    """
    from deception_circuits.paper import PaperConfig

    root = Path(__file__).resolve().parents[1] / "configs"
    shipped = sorted(root.glob("real_*.yaml"))
    assert shipped, "expected committed real-run configs"
    for path in shipped:
        config = PaperConfig.from_yaml(path)
        spec = ExtractionSpec.from_paper_config(config)
        # Either the template appends a literal suffix ending at the boundary, or it
        # ends at the placeholder and the boundary comes from the row's own text --
        # which is why the binding check is per-row, in render_extraction_prompt.
        suffix = spec.prompt_template.rsplit("}", 1)[-1]
        assert (not suffix.strip()
                or suffix.rstrip().endswith(spec.decision_boundary_marker)), path.name
        assert spec.torch_dtype in ("float16", "bfloat16", "float32"), path.name
        # And a row that ends at the boundary must render with the read position last.
        row = pd.Series({"statement": f"game state\n{spec.decision_boundary_marker}"
                         if not suffix.strip() else "game state", "response": "x"})
        rendering = render_extraction_prompt(row, spec, StubProvider())
        assert rendering.text.rstrip().endswith(spec.decision_boundary_marker), path.name
        assert rendering.token_index == rendering.n_tokens - 1, path.name


def test_dtype_is_validated_rather_than_silently_defaulted():
    """A typo must fail, not quietly load the model in float32 at double memory."""
    with pytest.raises(ResearchIntegrityError, match="torch_dtype must be one of"):
        _spec(torch_dtype="bfloat_16")
    assert _spec(torch_dtype="bfloat16").torch_dtype == "bfloat16"


def test_dtype_is_reachable_from_a_paper_config(tmp_path):
    csv = _dataset(tmp_path, n_groups=3)
    config = _config(csv, tmp_path / "acts", tmp_path, torch_dtype="bfloat16")
    assert ExtractionSpec.from_paper_config(config).torch_dtype == "bfloat16"


def test_spec_fingerprint_changes_with_every_scientific_choice():
    base = _spec()
    variants = [
        _spec(activation_site="mlp"),
        _spec(model_revision="r1"),
        _spec(tokenizer_revision="r1"),
        _spec(layer_indices=(0, 1)),
        _spec(max_sequence_length=256),
        _spec(prompt_template="Hand: {statement}\nAction:"),
        _spec(prompt_template_id="other"),
        _spec(torch_dtype="bfloat16"),
    ]
    fingerprints = {base.fingerprint()} | {v.fingerprint() for v in variants}
    assert len(fingerprints) == len(variants) + 1

def test_block_extraction_uses_decoder_block_hooks_not_model_hidden_states(monkeypatch):
    """Block/residual extraction must capture the real block output.

    For Llama, output_hidden_states replaces the final recorded block output
    with the post-final-RMSNorm last_hidden_state. The paper intervention site
    is the decoder block output before that norm, so extraction must use the
    same module hook.
    """
    provider = TransformersHiddenStateProvider.__new__(
        TransformersHiddenStateProvider
    )
    provider.device = "cpu"

    expected = torch.arange(
        24,
        dtype=torch.float32,
    ).reshape(2, 3, 4)

    class ForbiddenModel:
        def __call__(self, *args, **kwargs):
            raise AssertionError(
                "block extraction must not use model output_hidden_states"
            )

    provider.model = ForbiddenModel()

    def fake_hooked_hidden_states(ids, site):
        assert site == "block"
        assert ids.shape == (1, 3)
        return expected

    monkeypatch.setattr(
        provider,
        "_hooked_hidden_states",
        fake_hooked_hidden_states,
    )

    actual = provider.hidden_states(
        [10, 20, 30],
        "residual_stream",
    )

    assert torch.equal(actual, expected)