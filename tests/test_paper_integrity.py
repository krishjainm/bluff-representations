from pathlib import Path
import json
import sys

import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _stubs import StubHiddenStateProvider

from deception_circuits.paper import (
    PaperConfig,
    ResearchIntegrityError,
    audit_run,
    load_activations,
    make_split_manifest,
    render_predecision_prompt,
    run_probe_experiment,
    validate_dataset,
)
from deception_circuits.paper_extraction import ExtractionSpec, run_extraction
from deception_circuits.causal_generation import CausalLMInterventionRunner

STUB_MODEL = "stub/tiny-test-model"


def _config(csv: Path, activations: Path, tmp_path: Path, **overrides) -> PaperConfig:
    base = dict(experiment_name="test", dataset_path=str(csv), activation_dir=str(activations),
                output_dir=str(tmp_path / "out"), subject_model=STUB_MODEL,
                prompt_template_id="poker_action_v1", n_seeds=2, bootstrap_resamples=30)
    base.update(overrides)
    return PaperConfig(**base)


def _fixture(tmp_path: Path):
    """Build a dataset plus real (stub-derived) pre-decision activations.

    The statements differ between the two rows of a group, so a prompt-end probe
    legitimately has signal here; nothing in this fixture asserts a performance
    level, it only exercises split/selection/CI logic.
    """
    rows = []
    for group in range(20):
        for label in (0, 1):
            rows.append({"sample_id": f"{group}-{label}", "base_item_id": group,
                         "split_group_id": group, "statement": f"hand {group} variant {label}",
                         "response": "response", "label": label, "scenario": "poker"})
    csv = tmp_path / "dataset.csv"
    df = pd.DataFrame(rows)
    df.to_csv(csv, index=False)
    activations = tmp_path / "activations"
    spec = ExtractionSpec(subject_model=STUB_MODEL, prompt_template_id="poker_action_v1")
    run_extraction(df, spec, activations, StubHiddenStateProvider())
    return csv, activations


def test_paper_probe_is_group_safe_and_validation_selected(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    config = _config(csv, activation_dir, tmp_path)
    df = validate_dataset(csv)
    manifest = make_split_manifest(df, config)
    split_by_id = {sid: name for name in ("train", "validation", "test") for sid in manifest[name]}
    assert df.assign(split=df.sample_id.map(split_by_id)).groupby("split_group_id").split.nunique().max() == 1
    result = run_probe_experiment(df, load_activations(df, activation_dir, config), manifest, config)
    assert result["selection_partition"] == "validation"
    assert result["test_partition_used_for_selection"] is False
    assert len(result["runs"]) == 2
    assert result["runs"][0]["test_auroc_grouped_ci"]["valid_resamples"] > 0
    assert "prompt_text" in result["baselines"]


def test_missing_activation_file_is_a_hard_error(tmp_path):
    csv, activation_dir = _fixture(tmp_path)
    (activation_dir / "sample_0-0.pt").unlink()
    # Matches the per-sample tensor error, not the manifest error, so this test
    # cannot pass merely because provenance is absent.
    with pytest.raises(FileNotFoundError, match=r"Missing activation for sample_id=0-0"):
        load_activations(validate_dataset(csv), activation_dir, require_manifest=False)


def test_duplicate_sample_id_is_rejected(tmp_path):
    csv, _ = _fixture(tmp_path)
    df = pd.read_csv(csv)
    df.loc[1, "sample_id"] = df.loc[0, "sample_id"]
    df.to_csv(csv, index=False)
    with pytest.raises(ResearchIntegrityError, match="unique"):
        validate_dataset(csv)


def test_predecision_prompt_has_explicit_boundary_and_no_response_argument():
    prompt, boundary = render_predecision_prompt("Raise if appropriate")
    assert prompt == "Raise if appropriate\nAction:"
    assert boundary == len(prompt)
    with pytest.raises(ResearchIntegrityError, match="Action"):
        render_predecision_prompt("x", "Question: {statement}")


def test_patching_replaces_and_prefill_policy_runs_once():
    # Avoid model loading: the hook itself is pure tensor logic.
    runner = CausalLMInterventionRunner.__new__(CausalLMInterventionRunner)
    runner.device = "cpu"
    patch = torch.full((3,), 7.0)
    hook = runner._make_hook("replace", patch, strength=0.1, timing_policy="prefill_only")
    first = hook(None, None, torch.zeros(1, 2, 3))
    second = hook(None, None, torch.zeros(1, 1, 3))
    assert torch.equal(first[0, -1], patch)
    assert torch.equal(second, torch.zeros(1, 1, 3))
def test_global_audit_requires_transfer_for_multicontext_dataset(tmp_path):
    rows = []

    for context in ("poker", "negotiation"):
        for group in range(20):
            for label in (0, 1):
                rows.append({
                    "sample_id": f"{context}-{group}-{label}",
                    "base_item_id": f"{context}-{group}",
                    "split_group_id": f"{context}-{group}",
                    "statement": (
                        f"{context} example {group} variant {label}"
                    ),
                    "response": "response",
                    "label": label,
                    "scenario": context,
                })

    csv = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    df = validate_dataset(csv)

    activation_dir = tmp_path / "activations"

    run_extraction(
        df,
        ExtractionSpec(
            subject_model=STUB_MODEL,
            prompt_template_id="poker_action_v1",
        ),
        activation_dir,
        StubHiddenStateProvider(),
    )

    config = _config(
        csv,
        activation_dir,
        tmp_path,
        nuisance_columns=(),
        learning_curve_sizes=(),
        intervention_strengths=(),
        sae_config={},
    )

    manifest = make_split_manifest(df, config)

    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    (out / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (out / "resolved_config.yaml").write_text(
        "{}\n",
        encoding="utf-8",
    )
    (out / "run_metadata.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    probe_result = run_probe_experiment(
        df,
        load_activations(
            df,
            activation_dir,
            config,
        ),
        manifest,
        config,
    )

    (out / "probe_results.json").write_text(
        json.dumps(probe_result, indent=2) + "\n",
        encoding="utf-8",
    )

    failures = audit_run(
        out,
        df,
        config,
    )

    assert any(
        "cross_context_results.json is missing" in failure
        for failure in failures
    )