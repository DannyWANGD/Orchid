"""Prepare two v2 review packets for the frozen 92M decoder comparison."""

import argparse
from collections import Counter
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.blind import export_reviewer_zips, make_packets
from orchid.io import ROOT, read_json, read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=Path, default=ROOT / "report/evidence/prosody-study/test/generations.jsonl")
    parser.add_argument("--protocol", type=Path, default=ROOT / "artifacts/eval_prompts.json")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    blind = protocol["blind"]
    modes = ("raw", "rhymed")
    rows = [
        {**row, "model": row["mode"], "suite": "main"}
        for row in read_jsonl(args.generations)
        if row["suite"] == "test" and row["prompt_id"] in blind["prompt_ids"]
        and row["sampling_seed"] == blind["sample_seed"] and row["mode"] in modes
    ]
    if Counter(len(row["keywords"]) for row in rows) != {1: 20, 2: 20, 3: 20}:
        raise ValueError("Expected ten frozen prompt pairs per keyword count")
    args.output.mkdir(parents=True)
    adapted = args.output / "review_generations.jsonl"
    write_jsonl(adapted, rows)

    organizer = args.output / "organizer"
    names = make_packets(adapted, args.protocol, organizer, modes, seed=20261121, rubric_version="v2")
    templates = tuple(organizer / name for name in names)
    export_reviewer_zips(adapted, args.protocol, organizer / "private_key.json", templates, None,
                         args.output / "packets")
    print(f"Two independent 60-row packets at {args.output / 'packets'}")


if __name__ == "__main__":
    main()
