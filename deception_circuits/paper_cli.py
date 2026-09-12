"""Command line entry point for the strict paper pipeline."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import yaml
from .paper import (PaperConfig, audit_run, load_activations, make_split_manifest,
                    run_probe_experiment, save_manifest, validate_dataset, write_run_metadata)

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("command", choices=["validate-data", "make-splits", "train-probes", "audit"]); p.add_argument("--config", required=True)
    a = p.parse_args(); c = PaperConfig.from_yaml(a.config); out = Path(c.output_dir); out.mkdir(parents=True, exist_ok=True)
    if a.command == "audit":
        errors = audit_run(out); print("PASS" if not errors else "FAIL\n" + "\n".join(errors)); raise SystemExit(bool(errors))
    df = validate_dataset(c.dataset_path)
    if a.command == "validate-data": print(f"PASS: {len(df)} rows"); return
    manifest_path = Path(c.split_manifest_path or out / "split_manifest.json")
    if a.command == "make-splits": save_manifest(make_split_manifest(df, c), manifest_path); print(manifest_path); return
    if not manifest_path.is_file(): raise FileNotFoundError("Create a split manifest before training probes")
    manifest = json.loads(manifest_path.read_text()); acts = load_activations(df, c.activation_dir)
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(c.__dict__, sort_keys=True)); write_run_metadata(c, out)
    result = run_probe_experiment(df, acts, manifest, c); (out / "probe_results.json").write_text(json.dumps(result, indent=2) + "\n"); print(out / "probe_results.json")
if __name__ == "__main__": main()
