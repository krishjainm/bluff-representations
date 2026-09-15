"""Command line entry point for the strict paper pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .paper import (PaperConfig, audit_run, load_activations, make_split_manifest,
                    run_probe_experiment, save_manifest, validate_dataset, write_run_metadata)
from .paper_extraction import ExtractionSpec, run_extraction

COMMANDS = ("validate-data", "make-splits", "collect-activations", "train-probes", "audit")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="deception-paper", description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu", help="collect-activations: torch device")
    parser.add_argument(
        "--confirm-model-load", action="store_true",
        help="collect-activations: required acknowledgement that this loads subject-model "
             "weights (a potentially large download) and runs real forward passes",
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
        print("PASS" if not errors else "FAIL\n" + "\n".join(errors))
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
