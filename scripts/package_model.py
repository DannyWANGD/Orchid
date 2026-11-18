"""Export inference weights, tokenizer, version and integrity manifest."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.inference import verify_bundle
from orchid.io import sha256_file, write_bytes, write_json
from orchid.training import load_checkpoint, save_checkpoint


def export(checkpoint, tokenizer, output):
    full = load_checkpoint(checkpoint)
    meta = dict(full["meta"])
    if sha256_file(tokenizer) != meta["lineage"]["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from training")
    output.mkdir(parents=True)
    meta["tokenizer_path"] = "tokenizer.json"
    write_bytes(output / "tokenizer.json", tokenizer.read_bytes())
    write_json(output / "config.json", meta)
    save_checkpoint(output / "weights.pt", full["model"])
    write_json(output / "bundle.json", {
        "version": 1, "source_run": meta["run_id"], "source_checkpoint_sha256": sha256_file(checkpoint),
        "source_stage": meta["stage"], "source_step": full["cursor"]["step"],
        "files": {name: sha256_file(output / name) for name in ("weights.pt", "tokenizer.json", "config.json")},
    })
    write_bytes(output / "README.txt", (
        "Orchid inference bundle v1. Traditional characters; four lines of five characters.\n"
        "From the Orchid repository root with dependencies installed:\n"
        f"python -E scripts/generate.py --checkpoint {output} --keywords 春 雨 --device cpu\n"
        "Use --device cuda for CUDA. The course source includes the offline Pingshui rhyme table.\n"
        "Use --mode raw for original model sampling.\n"
    ).encode())
    return verify_bundle(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--tokenizer", default=Path("artifacts/tokenizer.json"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    print(export(args.checkpoint, args.tokenizer, args.output))


if __name__ == "__main__":
    main()
