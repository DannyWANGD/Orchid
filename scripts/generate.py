"""Generate a five-by-four poem with a trained Orchid checkpoint or bundle."""

import argparse
from pathlib import Path
import sys

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.generation import Sampling
from orchid.inference import load_poet
from orchid.io import json_text
from orchid.metrics import judge_format_keywords


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt or bundle directory")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--keywords", nargs="+", default=["春", "雨"])
    parser.add_argument("--form", choices=("wujue",), default="wujue")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--temperature", type=float, default=.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--mode", choices=("rhymed", "rhyme-only", "raw"),
                        help="Default rhymed mode enforces four lines, keywords and distinct Pingshui rhymes")
    parser.add_argument("--candidates", type=int, default=4,
                        help="Number of keyword placements to generate and rank in rhymed mode (1–16)")
    parser.add_argument("--constrain-form", action="store_true",
                        help="Use the form-only decoder from the original experiments")
    parser.add_argument("--constrain-keywords", action="store_true",
                        help="Use the original fixed-position keyword decoder")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.mode and (args.constrain_form or args.constrain_keywords):
        parser.error("Choose --mode or the original --constrain-* options")
    mode = args.mode or ("fixed-keywords" if args.constrain_keywords else
                         "form" if args.constrain_form else "rhymed")
    rhymed = mode in ("rhymed", "rhyme-only")
    keywords_required = mode in ("rhymed", "fixed-keywords")
    torch.set_num_threads(2)
    try:
        spec = Sampling(args.temperature, args.top_k, args.top_p, args.max_new_tokens)
        poem = load_poet(args.checkpoint, args.device, tokenizer_path=args.tokenizer)
        if args.form != poem.task.form:
            raise ValueError("The requested form does not match the checkpoint")
        words = poem.task.validate_keywords(args.keywords)
        result = poem.tokens(words, spec, args.seed, constrain_form=mode == "form",
                             constrain_keywords=keywords_required, constrain_rhyme=rhymed,
                             candidates=args.candidates)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json_text({**result, **judge_format_keywords(result["generated"], words),
                         "keywords": words, "device": str(poem.device), "load_seconds": poem.load_seconds,
                         "mode": mode, "constrained_form": mode != "raw", "constrained_rhyme": rhymed,
                         "constrained_keywords": keywords_required, "model_hash": poem.checkpoint_sha256}))
    else:
        text = result["generated"]
        print("\n".join(text[i:i + 6] for i in range(0, len(text), 6)) if rhymed else text)
        rhyme = f" · 韻部 {'、'.join(result['ping_matches'])}" if rhymed else ""
        print(f"{poem.device} · load {poem.load_seconds:.3f}s · generate {result['seconds']:.3f}s · "
              f"{mode} · {result['stop']}{rhyme}", file=sys.stderr)


if __name__ == "__main__":
    main()
