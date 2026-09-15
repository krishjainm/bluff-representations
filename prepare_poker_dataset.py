#!/usr/bin/env python3
"""Materialize the canonical CSV from the raw normalized_poker_gpt4o dataset.

The raw file lives on the `dataset-integration` branch, not in this working
tree. Restore it first:

    git checkout origin/dataset-integration -- sample_data/normalized_poker_gpt4o.fixed.csv

then run this to produce a schema-valid CSV plus a dataset card:

    python prepare_poker_dataset.py --prompt-variant judge_question
    python prepare_poker_dataset.py --prompt-variant neutral_state

Read docs/POKER_DATASET_NOTES.md before interpreting anything produced from it.
Outputs go under data/derived/ and are git-ignored: they are large and derived.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from deception_circuits.paper import validate_dataset
from deception_circuits.poker_adapter import adapt_normalized_poker_csv

DEFAULT_SOURCE = "sample_data/normalized_poker_gpt4o.fixed.csv"
FALLBACK_SOURCE = "sample_data/normalized_poker_gpt4o.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default=None, help=f"default: {DEFAULT_SOURCE}")
    parser.add_argument("--prompt-variant", default="judge_question",
                        choices=["judge_question", "neutral_state"])
    parser.add_argument("--out-dir", default="data/derived")
    args = parser.parse_args()

    source = Path(args.source) if args.source else Path(DEFAULT_SOURCE)
    if args.source is None and not source.is_file() and Path(FALLBACK_SOURCE).is_file():
        source = Path(FALLBACK_SOURCE)
    if not source.is_file():
        raise SystemExit(
            f"Source CSV not found: {source}\n"
            "Restore it with:\n"
            "  git checkout origin/dataset-integration -- "
            "sample_data/normalized_poker_gpt4o.fixed.csv"
        )

    adapted, report = adapt_normalized_poker_csv(str(source), prompt_variant=args.prompt_variant)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"poker_bluff_{args.prompt_variant}_v1"
    csv_path = out_dir / f"{stem}.csv"
    adapted.to_csv(csv_path, index=False)

    report["source_file"] = str(source)
    report["output_file"] = str(csv_path)
    card_path = out_dir / f"{stem}_dataset_card.json"
    card_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Fail here rather than three stages later if the adapter regressed.
    validate_dataset(csv_path)
    print(f"wrote {csv_path} ({report['n_rows']} rows, {report['n_groups']} groups)")
    print(f"wrote {card_path}")
    print("integrity flags:")
    for key, value in report["integrity_flags"].items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
