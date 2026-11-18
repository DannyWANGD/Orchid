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
    parser.add_argument("--constrain-form", action="store_true",
                        help="Force four five-character lines; does not ensure keyword use or rhyme")
    parser.add_argument("--constrain-keywords", action="store_true",
                        help="Force the form and place each keyword in a line; may sound less natural")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        spec = Sampling(args.temperature, args.top_k, args.top_p, args.max_new_tokens)
        poem = load_poet(args.checkpoint, args.device, tokenizer_path=args.tokenizer)
        if args.form != poem.task.form:
            raise ValueError("The requested form does not match the checkpoint")
        words = poem.task.validate_keywords(args.keywords)
        result = poem.tokens(words, spec, args.seed, constrain_form=args.constrain_form,
                             constrain_keywords=args.constrain_keywords)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    if args.json:
        print(json_text({**result, **judge_format_keywords(result["generated"], words),
                         "keywords": words, "device": str(poem.device), "load_seconds": poem.load_seconds,
                         "constrained_form": args.constrain_form or args.constrain_keywords,
                         "constrained_keywords": args.constrain_keywords, "model_hash": poem.checkpoint_sha256}))
    else:
        print(result["generated"])
        print(f"{poem.device} · load {poem.load_seconds:.3f}s · generate {result['seconds']:.3f}s · "
              f"{result['stop']}" + (" · keywords and form constrained" if args.constrain_keywords else
                                    " · form constrained" if args.constrain_form else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
