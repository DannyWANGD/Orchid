"""Build a train-only character vocabulary and separate pretraining token streams."""

import argparse
from collections import Counter
from itertools import chain
from pathlib import Path
import sys

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.corpus import SPLITS
from orchid.io import read_json, read_jsonl, read_yaml, sha256_file, write_json
from orchid.tokenizer import BOS, EOS, UNK, CharacterTokenizer


def build(config):
    proc, art = (Path(config["paths"][name]) for name in ("processed", "artifacts"))
    manifest = art / "split_manifest.json"
    splits = read_json(manifest)["splits"]
    paths = {s: proc / f"{s}.jsonl" for s in SPLITS}

    def poems(split):
        # Boundaries are fixed in the manifest, regardless of processed filename.
        seen = []
        for r in read_jsonl(paths[split]):
            if r["split"] != split:
                raise ValueError(f"Unexpected split in {paths[split]}: {r['poem_id']}")
            seen.append(r["poem_id"])
            yield r
        if seen != splits[split]:
            raise ValueError(f"Unexpected poem order or membership in {paths[split]}")

    tokenizer = CharacterTokenizer.fit(r["text"] for r in poems("train"))
    sha = tokenizer.save(art / "tokenizer.json")
    (proc / "tokens").mkdir(parents=True, exist_ok=True)
    stats = {
        "vocab_size": len(tokenizer), "tokenizer_sha256": sha,
        "split_manifest_sha256": sha256_file(manifest),
        "data_manifest_sha256": sha256_file(art / "data_manifest.json"),
        "splits": {},
    }
    for s in SPLITS:
        unk_chars: Counter = Counter()

        def sequence():
            for row in poems(s):
                text = row["text"]
                ids = tokenizer.encode(text)
                unk_chars.update(c for c, idx in zip(text, ids) if idx == UNK)
                if s == "train" and tokenizer.decode(ids) != text:
                    raise AssertionError(f"Train round trip failed: {row['poem_id']}")
                yield chain((BOS,), ids, (EOS,))

        array = np.fromiter(chain.from_iterable(sequence()), dtype=np.int32)
        output = proc / "tokens" / f"{s}.npy"
        with output.with_suffix(".tmp").open("wb") as dst:
            np.save(dst, array, allow_pickle=False)
        output.with_suffix(".tmp").replace(output)
        unknown = sum(unk_chars.values())
        text_count = len(array) - 2 * len(splits[s])
        stats["splits"][s] = {
            "poems": len(splits[s]), "stream_tokens": len(array), "body_and_punctuation_tokens": text_count,
            "unk_count": unknown, "unk_types": len(unk_chars), "unk_fraction": unknown / text_count if text_count else 0,
            "top_unk": unk_chars.most_common(12),
            "token_file_sha256": sha256_file(output), "poem_file_sha256": sha256_file(paths[s]),
        }
    write_json(art / "token_stats.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    args = parser.parse_args()
    stats = build(read_yaml(args.config))
    print(f"Vocabulary: {stats['vocab_size']:,}")
    for s, values in stats["splits"].items():
        print(f"{s}: {values['stream_tokens']:,} tokens, OOV {values['unk_fraction']:.3%}")


if __name__ == "__main__":
    main()
