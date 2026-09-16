"""Command line entry point for the strict paper pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from .paper import (PaperConfig, audit_notes, audit_run, load_activations, make_split_manifest,
                    run_probe_experiment, save_manifest, validate_dataset, write_run_metadata)
from .paper_extraction import ExtractionSpec, run_extraction

COMMANDS = ("validate-data", "make-splits", "collect-activations", "train-probes",
            "run-baselines", "analyze-confounds", "train-sae", "run-interventions",
            "make-figures", "audit")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deception-paper", description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu",
                        help="torch device for collect-activations, train-sae, "
                             "and run-interventions (e.g. cuda)")
    parser.add_argument(
        "--confirm-model-load", action="store_true",
        help="collect-activations / run-interventions: required acknowledgement that this loads "
             "subject-model weights (a potentially large download) and runs real forward passes",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="collect-activations: re-extract every sample instead of resuming",
    )
    return parser


def _manifest_path(config: PaperConfig, out: Path) -> Path:
    return Path(config.split_manifest_path) if config.split_manifest_path else out / "split_manifest.json"


def main() -> None:
    args = _build_parser().parse_args()
    config = PaperConfig.from_yaml(args.config)
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.command == "audit":
        errors = audit_run(out, validate_dataset(config.dataset_path), config)
        figures_dir = out / "figures"
        if (figures_dir / "figures_manifest.json").is_file():
            from .paper_figures import audit_figures
            errors = errors + audit_figures(figures_dir)
        print("PASS" if not errors else "FAIL\n" + "\n".join(f"  FAIL: {e}" for e in errors))
        for note in audit_notes(out, config):
            print(f"  note: {note}")
        raise SystemExit(bool(errors))

    df = validate_dataset(config.dataset_path)

    if args.command == "validate-data":
        print(f"PASS: {len(df)} rows")
        return

    if args.command == "make-splits":
        path = _manifest_path(config, out)
        save_manifest(make_split_manifest(df, config), path)
        print(path)
        return

    if args.command == "collect-activations":
        spec = ExtractionSpec.from_paper_config(config)
        if not args.confirm_model_load:
            raise SystemExit(
                "collect-activations loads subject-model weights and runs real forward passes.\n"
                f"  model:  {spec.subject_model} (revision={spec.model_revision})\n"
                f"  site:   {spec.activation_site}\n"
                f"  mode:   {spec.activation_mode}\n"
                f"  rows:   {len(df)}\n"
                f"  output: {config.activation_dir}\n"
                "Re-run with --confirm-model-load to authorize it."
            )
        # Imported lazily so that the other commands never touch model weights.
        from .paper_extraction import TransformersHiddenStateProvider

        provider = TransformersHiddenStateProvider(spec, device=args.device)
        manifest = run_extraction(
            df, spec, config.activation_dir, provider,
            resume=not args.no_resume, progress_every=50,
        )
        print(f"extracted={manifest['n_newly_extracted']} resumed={manifest['n_resumed']} "
              f"total={manifest['n_samples']} shape={manifest['activation_shape']}")
        return

    if args.command == "analyze-confounds":
        from .paper_confounds import run_confound_suite

        path = _manifest_path(config, out)
        if not path.is_file():
            raise FileNotFoundError("Create a split manifest before analyzing confounds")
        split_manifest = json.loads(path.read_text())
        probe_results = out / "probe_results.json"
        if not probe_results.is_file():
            raise FileNotFoundError(
                "Run train-probes first: the confound suite reuses the validation-selected "
                "layer rather than choosing its own, so it cannot leak test information."
            )
        runs = json.loads(probe_results.read_text())["runs"]
        # Modal selected layer across seeds; selection happened on validation upstream.
        layers = [int(r["selected_layer"]) for r in runs]
        layer = max(set(layers), key=layers.count)
        activations = load_activations(df, config.activation_dir, config)
        result = run_confound_suite(activations, df, split_manifest, config, layer=layer)
        result["selected_layer_source"] = {
            "from": "probe_results.json", "per_seed_layers": layers, "rule": "modal layer across seeds"}
        (out / "confound_results.json").write_text(json.dumps(result, indent=2) + "\n")
        unavailable = result["metadata_availability"]["unavailable_analyses"]
        print(out / "confound_results.json")
        if unavailable:
            print("unavailable without more metadata: " + ", ".join(unavailable))
        return

    if args.command == "run-baselines":
        # Deliberately model-free: this is the bar the hidden-state probe has to
        # clear, and it costs nothing, so it should be run before any GPU time.
        from .paper import run_baselines
        from .paper_confounds import describe_metadata_availability

        path = _manifest_path(config, out)
        if not path.is_file():
            raise FileNotFoundError("Create a split manifest before running baselines")
        split_manifest = json.loads(path.read_text())
        result = {
            "note": "no activations and no model are involved in any number here",
            "n_train": len(split_manifest["train"]), "n_test": len(split_manifest["test"]),
            "partition_summary": split_manifest.get("partition_summary"),
            "metadata_availability": describe_metadata_availability(df, config.nuisance_columns),
            "baselines": run_baselines(df, split_manifest, config),
        }
        (out / "baseline_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(out / "baseline_results.json")
        for name, entry in result["baselines"].items():
            if isinstance(entry, dict) and "auroc" in entry:
                print(f"  {name:28s} AUROC {entry['auroc']:.4f}  PR-AUC {entry['pr_auc']:.4f}")
            elif isinstance(entry, dict):
                print(f"  {name:28s} {entry.get('status', entry)}")
            else:
                print(f"  {name:28s} {entry:.4f}")
        return

    if args.command == "make-figures":
        from .paper_figures import build_all_figures

        manifest = build_all_figures(out)
        print(f"{manifest['figures_dir']}/figures_manifest.json")
        print(f"  generated {manifest['n_generated']}, not run {manifest['n_not_run']}")
        for name, reason in manifest["not_run"].items():
            print(f"  not run: {name}: {reason}")
        return

    if args.command == "train-sae":
        from .paper_sae import SAEConfig, run_sae_experiment

        path = _manifest_path(config, out)
        probe_results = out / "probe_results.json"
        if not path.is_file():
            raise FileNotFoundError("Create a split manifest before training an SAE")
        if not config.sae_config:
            raise SystemExit(
                "Set sae_config in the config. n_features must be explicit: there is no "
                "'same as input' default, because that is not an overcomplete dictionary.")
        split_manifest = json.loads(path.read_text())
        if probe_results.is_file():
            layers = [int(r["selected_layer"]) for r in json.loads(probe_results.read_text())["runs"]]
            layer = max(set(layers), key=layers.count)
        else:
            raise FileNotFoundError(
                "train-sae reuses the validation-selected layer from probe_results.json so the "
                "SAE and the probe are not analysed at inconsistent layers. Run train-probes first.")
        sae_config = SAEConfig(**config.sae_config)
        activations = load_activations(df, config.activation_dir, config)
        result = run_sae_experiment(
            activations, df, split_manifest, config, sae_config, layer=layer,
            top_n_features=config.sae_top_n_features,
            stability_seeds=[int(s) for s in config.sae_stability_seeds],
            metadata_columns=config.nuisance_columns, output_dir=out,
            device=args.device)
        (out / "sae_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(out / "sae_results.json")
        test = result["diagnostics"]["test"]
        print(f"  device={result['training'].get('device')} "
              f"layer={result['layer']} expansion={result['expansion_factor']:.2f} "
              f"EV={test['fraction_variance_explained']:.3f} L0={test['l0_mean']:.2f} "
              f"dead={test['dead_feature_fraction']:.3f}")
        print(f"  selected features (ranked on train+validation): {result['selected_features']}")
        return

    if args.command == "run-interventions":
        from .paper_causal import (ForcedChoiceEndpoint, build_direction_set,
                                   choose_wrong_layer_offset, default_conditions,
                                   nuisance_condition, run_causal_suite)

        path = _manifest_path(config, out)
        probe_results = out / "probe_results.json"
        if not path.is_file() or not probe_results.is_file():
            raise FileNotFoundError(
                "run-interventions needs a split manifest and probe_results.json: the steering "
                "direction is fitted at the validation-selected layer, not chosen here.")
        if not config.endpoint_options or not config.endpoint_positive_option:
            raise SystemExit(
                "Set endpoint_options and endpoint_positive_option in the config. The primary "
                "endpoint must be the model's own output, never the probe score.")
        if not config.intervention_strengths:
            raise SystemExit("Set intervention_strengths in the config (a dose-response grid).")

        split_manifest = json.loads(path.read_text())
        runs = json.loads(probe_results.read_text())["runs"]
        layers = [int(r["selected_layer"]) for r in runs]
        layer = int(config.intervention_layer) if config.intervention_layer is not None \
            else max(set(layers), key=layers.count)
        held_out = df[df.sample_id.isin(set(split_manifest["test"]))]
        if config.causal_eval_size and len(held_out) > config.causal_eval_size:
            # Deterministic head of the held-out set; size is configurable, not hardcoded.
            held_out = held_out.head(config.causal_eval_size)

        if not args.confirm_model_load:
            raise SystemExit(
                "run-interventions loads subject-model weights and runs real forward passes.\n"
                f"  model:      {config.subject_model}\n"
                f"  layer:      {layer} (site={config.intervention_site or 'block'})\n"
                f"  prompts:    {len(held_out)} held-out\n"
                f"  strengths:  {list(config.intervention_strengths)}\n"
                f"  endpoint:   P({config.endpoint_positive_option}) over "
                f"{list(config.endpoint_options)}\n"
                "Re-run with --confirm-model-load to authorize it.")

        from .paper_causal import TransformersInterventionRunner

        activations = load_activations(df, config.activation_dir, config)
        directions = build_direction_set(activations, df, split_manifest, config, layer=layer,
                                         nuisance_column=config.causal_nuisance_column)
        conditions = default_conditions(
            wrong_layer_offset=choose_wrong_layer_offset(layer, activations.shape[1]))
        for name in directions:
            if name.startswith("nuisance_"):
                conditions = conditions + (nuisance_condition(name),)
        runner = TransformersInterventionRunner(
            config.subject_model, device=args.device,
            site=config.intervention_site or "block", revision=config.model_revision)
        # Same template the activations were extracted with, so the intervention
        # lands at the position the direction was fitted for.
        spec = ExtractionSpec.from_paper_config(config)
        result = run_causal_suite(
            runner, held_out, directions,
            ForcedChoiceEndpoint(tuple(config.endpoint_options), config.endpoint_positive_option),
            layer=layer, strengths=list(config.intervention_strengths),
            conditions=conditions, seed=config.seed,
            n_resamples=config.bootstrap_resamples,
            group_column=split_manifest.get("group_column", "base_item_id"),
            prompt_template=spec.prompt_template)
        # Per-row records go to CSV; the JSON keeps summaries only.
        records = result.pop("records")
        pd.DataFrame(records).to_csv(out / "intervention_records.csv", index=False)
        (out / "causal_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(out / "causal_results.json")
        for name, entry in result["effects"].items():
            for strength, stats in entry["by_strength"].items():
                print(f"  {name:38s} a={strength:>5s}  d={stats['mean_difference']:+.4f} "
                      f"[{stats['ci_lower']:+.4f},{stats['ci_upper']:+.4f}]")
        return

    if args.command == "train-probes":
        path = _manifest_path(config, out)
        if not path.is_file():
            raise FileNotFoundError("Create a split manifest before training probes")
        split_manifest = json.loads(path.read_text())
        activations = load_activations(df, config.activation_dir, config)
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(config.__dict__, sort_keys=True))
        write_run_metadata(config, out)
        result = run_probe_experiment(df, activations, split_manifest, config)
        (out / "probe_results.json").write_text(json.dumps(result, indent=2) + "\n")
        print(out / "probe_results.json")
        return


if __name__ == "__main__":
    main()
