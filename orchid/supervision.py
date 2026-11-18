"""Deterministic weak keyword labels bound to the original poem's split."""

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import random

from .corpus import IS_HAN_LINE, SPLITS, canonical_poem, is_wujue
from .data import prompt_ids, response_item
from .io import read_json, read_jsonl, sha256_file, write_json, write_jsonl
from .runner import verified
from .tokenizer import EOS, CharacterTokenizer


RULE = "orchid-imagery-contiguous-one-per-poem-v1"


@dataclass(frozen=True)
class Task:
    form: str
    lines: int
    chars_per_line: int
    keywords: dict
    script: str

    def __post_init__(self):
        if (self.form, self.lines, self.chars_per_line) != ("wujue", 4, 5):
            raise ValueError("This tokenizer's form control is defined for a four-by-five poem")
        if self.script != "source_traditional":
            raise ValueError("Unsupported keyword character policy")

    def validate_keywords(self, words):
        from unicodedata import normalize

        bounds = self.keywords
        normalized = [normalize("NFC", w) for w in words]
        if (not bounds["min"] <= len(normalized) <= bounds["max"]
                or len(set(normalized)) != len(normalized)):
            raise ValueError("Provide 1–3 distinct keywords")
        for w in normalized:
            if not bounds["min_chars"] <= len(w) <= bounds["max_chars"] or not IS_HAN_LINE.fullmatch(w):
                raise ValueError(f"Expected 1–2 Han characters per keyword: {w!r}")
        return normalized


def candidates(lines, allowed):
    # Find full words strictly within a line, then eliminate redundant subwords.
    hits = {line[start:start + size] for line in lines
            for start in range(len(line)) for size in (1, 2)
            if line[start:start + size] in allowed}
    return sorted(w for w in hits if len(w) == 2 or all(w not in z for z in hits if len(z) == 2))


def read_words(path):
    return {w for line in path.read_text(encoding="utf-8").splitlines()
            for w in line.partition("#")[0].split()}


def source_rows(proc, token_stats, split):
    path = proc / f"{split}.jsonl"
    verified(path, token_stats["splits"][split]["poem_file_sha256"])
    return [r for r in read_jsonl(path) if r["split"] == split and r["form"] == "wujue"]


def build_supervision(data, settings, task: Task, context):
    proc, art = (Path(data["paths"][key]) for key in ("processed", "artifacts"))
    token_stats = read_json(art / "token_stats.json")
    tok = CharacterTokenizer.load(art / "tokenizer.json")
    paths = {"tokenizer": "tokenizer.json", "split_manifest": "split_manifest.json", "data_manifest": "data_manifest.json"}
    lineage = {f"{key}_sha256": verified(art / filename, token_stats[f"{key}_sha256"])
               for key, filename in paths.items()}
    original = settings.get("frozen_holdout_data")
    if original:
        from .io import read_yaml

        origin = read_yaml(Path(original))["paths"]
        original_proc, original_art = Path(origin["processed"]), Path(origin["artifacts"])
        original_stats = read_json(original_art / "sft_stats.json")
        original_tokens = read_json(original_art / "token_stats.json")
        if original_stats["tokenizer_sha256"] != lineage["tokenizer_sha256"]:
            raise ValueError("Frozen supervision requires identical tokenizers")
    whitelist_file, stop_file = (Path(settings[k]) for k in ("whitelist", "stopwords"))
    stop = read_words(stop_file)
    whitelist = read_words(whitelist_file) - stop
    for w in sorted(whitelist):
        task.validate_keywords([w])
    train = source_rows(proc, token_stats, "train")
    counts = Counter(w for row in train for w in candidates(row["lines"], whitelist))
    retained = {w for w, n in counts.items() if n >= settings["min_train_poems"]}
    lexicon = {
        "rule": RULE, "seed": settings["seed"], "whitelist": str(whitelist_file),
        "whitelist_sha256": sha256_file(whitelist_file), "stopwords": str(stop_file),
        "stopwords_sha256": sha256_file(stop_file), "min_train_poems": settings["min_train_poems"],
        "frequencies": {w: counts[w] for w in sorted(retained)}, **lineage,
    }
    lex_file = art / "imagery_lexicon.json"
    write_json(lex_file, lexicon)
    stats = {"rule": RULE, "lexicon_sha256": sha256_file(lex_file), **lineage, "splits": {}}
    if original:
        stats["frozen_holdout"] = {
            "data_config": original, "sft_stats_sha256": sha256_file(original_art / "sft_stats.json"),
            "splits": {s: original_stats["splits"][s]["file_sha256"] for s in ("validation", "test")},
        }
    for s in SPLITS:
        file = proc / "sft" / f"{s}.jsonl"
        if original and s != "train":
            from .io import write_bytes

            if token_stats["splits"][s] != original_tokens["splits"][s]:
                raise ValueError(f"Frozen holdout poems changed for {s}")
            frozen = original_proc / "sft" / f"{s}.jsonl"
            verified(frozen, original_stats["splits"][s]["file_sha256"])
            write_bytes(file, frozen.read_bytes())
            stats["splits"][s] = original_stats["splits"][s]
            continue
        rows, dropped, pairs, oov = (train if s == "train" else source_rows(proc, token_stats, s)), Counter(), [], 0
        for row in rows:
            assert is_wujue(row["lines"])
            words = candidates(row["lines"], retained)
            if not words:
                dropped["no_match"] += 1
                continue
            rng = random.Random(f"{settings['seed']}:{row['poem_id']}")
            k = min(len(words), rng.randint(task.keywords["min"], task.keywords["max"]))
            selected = task.validate_keywords(rng.sample(words, k))
            text = canonical_poem(row["lines"])
            ids = prompt_ids(tok, selected, strict=True)
            target = tok.encode(text)
            if s == "train" and tok.ids["<unk>"] in target:
                dropped["response_oov"] += 1
                continue
            if len(ids) + len(target) > context:
                dropped["overlong"] += 1
                continue
            if tok.ids["<unk>"] in target:
                oov += 1
            pairs.append({"poem_id": row["poem_id"], "group_id": row["group_id"], "split": s,
                          "form": task.form, "keywords": selected, "label_source": RULE,
                          "source": row["canonical_source"], "prompt": tok.decode(ids),
                          "response": text})
        write_jsonl(file, pairs)
        stats["splits"][s] = {"candidates": len(rows), "pairs": len(pairs), "excluded": dict(dropped),
                              "response_has_oov": oov, "keyword_oov": 0,
                              "keyword_count": Counter(len(r["keywords"]) for r in pairs),
                              "file_sha256": sha256_file(file)}
        if s == "train":
            train_pairs = pairs
    write_json(art / "sft_stats.json", stats)
    if not train_pairs:
        raise ValueError("The train-only lexicon yielded no SFT pairs")
    rng = random.Random(settings["seed"])
    audit = []
    for n in (1, 2, 3):
        subset = sorted((r for r in train_pairs if len(r["keywords"]) == n),
                        key=lambda r: (min(counts[w] for w in r["keywords"]), r["poem_id"]))
        bands = (subset[:len(subset)//2], subset[len(subset)//2:])
        for band, samples in zip(("lower", "higher"), bands):
            for row in rng.sample(samples, min(len(samples), 9)):
                audit.append({**row, "frequency_band": band, "train_poem_counts": [counts[w] for w in row["keywords"]]})
    write_jsonl(proc / "sft" / "audit.jsonl", audit)

    row = next((r for r in train_pairs if len(r["keywords"]) == 3), train_pairs[0])
    x, y = response_item(tok, row["keywords"], row["response"], strict=True)
    ids = prompt_ids(tok, row["keywords"])
    targets = y[y != -100].tolist()
    if targets != [*tok.encode(row["response"], strict=True), EOS]:
        raise AssertionError("Incorrect response-only labels")
    write_json(art / "sft_mask_example.json", {
        "poem_id": row["poem_id"], "keywords": row["keywords"], "prompt_len": len(ids),
        "response": row["response"], "task": asdict(task),
        "positions": [{"position": i, "input_id": a, "input": tok.vocab[a], "label_id": b,
                       "target": tok.vocab[b] if b >= 0 else None}
                      for i, (a, b) in enumerate(zip(x.tolist(), y.tolist()))],
    })
    return stats
