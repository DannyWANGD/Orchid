"""Freeze test prompts, reference membership and the evaluation sampling plan."""

from collections import Counter
from pathlib import Path
import random

from .io import read_json, read_jsonl, sha256_file, write_json
from .runner import verified
from .tokenizer import CharacterTokenizer
from .supervision import Task


def freeze(cfg, paths, task: Task):
    art, proc = paths["artifacts"], paths["processed"]
    tok_file = art / "tokenizer.json"
    token = CharacterTokenizer.load(tok_file)
    stat = read_json(art / "sft_stats.json")
    tr, te = (proc / "sft" / f"{s}.jsonl" for s in ("train", "test"))
    for s, file in (("train", tr), ("test", te)):
        verified(file, stat["splits"][s]["file_sha256"])
    tok_sha = verified(tok_file, stat["tokenizer_sha256"])
    train = list(read_jsonl(tr))
    individual = Counter(w for s in train for w in s["keywords"])
    sets = {tuple(sorted(row["keywords"])) for row in train}
    cfg_input = cfg["prompt_config"]
    seed = cfg_input["selection_seed"]
    main = Path(cfg_input["main"])
    edge = Path(cfg_input["edge"])

    def describe(kw):
        return {"keywords": kw,
                "train_combo": tuple(sorted(kw)) in sets,
                "train_keyword_counts": [individual[k] for k in kw],
                "oov": sorted({char for word in kw for char in word if char not in token.ids})}

    raw = [task.validate_keywords(line.partition("#")[0].split()) for line in main.read_text().splitlines()
           if line.partition("#")[0].strip()]
    if len({tuple(sorted(w)) for w in raw}) != len(raw):
        raise ValueError("Duplicate prompt combinations")
    primary = [{"id": f"M{i:03d}", **describe(words)} for i, words in enumerate(raw, 1)]
    if any(r["oov"] for r in primary):
        raise ValueError("Primary test contains OOV; put these requests in the edge suite.")
    holdout_groups = set()
    refs = []
    test = list(read_jsonl(te))
    random.Random(seed).shuffle(test)
    for row in test:
        if len(refs) >= cfg_input["reference_count"]:
            break
        if row["group_id"] in holdout_groups:
            continue
        holdout_groups.add(row["group_id"])
        refs.append({"id": f"T{len(refs)+1:03d}", **describe(row["keywords"]),
                     "poem_id": row["poem_id"], "group_id": row["group_id"]})
    edge_rows = [{"id": f"X{i:03d}", "case": row["case"], **describe(row["keywords"])}
                 for i, row in enumerate(read_json(edge), 1)]
    swaps = {}
    for row in primary:
        kw = row["keywords"]
        options = [other for other in primary
                   if len(other["keywords"]) == len(kw)
                   and not (set(kw) & set(other["keywords"]))
                   and not any(x in y or y in x for x in kw for y in other["keywords"])]
        if options:
            options.sort(key=lambda r: (sorted(map(len, kw)) != sorted(map(len, r["keywords"])), r["id"]))
            min_cost = sorted(map(len, kw)) != sorted(map(len, options[0]["keywords"]))
            good = [r for r in options if (sorted(map(len, kw)) != sorted(map(len, r["keywords"]))) == min_cost]
            swaps[row["id"]] = random.Random(f"{seed}:{row['id']}:swap").choice(good)["id"]
    rng = random.Random(f"{seed}:blind")
    blind_ids = []
    total = cfg_input["blind_count"]
    for n in (1, 2, 3):
        available = [r["id"] for r in primary if len(r["keywords"]) == n]
        count = total // 3 + (n <= total % 3)
        blind_ids.extend(rng.sample(available, count))
    plan = {
        "version": cfg_input["version"], "selection_seed": seed, "tokenizer_sha256": tok_sha,
        "sft_stats_sha256": sha256_file(art / "sft_stats.json"),
        "main_sha256": sha256_file(main), "edge_sha256": sha256_file(edge),
        "sampling": cfg["sampling"], "sampling_seeds": cfg["seeds"],
        "suites": {"main": primary, "reference": refs, "edge": edge_rows},
        "swaps": swaps, "blind": {"prompt_ids": blind_ids, "sample_seed": cfg["seeds"][0]},
    }
    frozen = Path(cfg_input["materialized"])
    if frozen.is_file() and read_json(frozen) != plan:
        raise ValueError(f"{frozen} is frozen. Use a new protocol version/path to change evaluation prompts.")
    write_json(frozen, plan)
    return plan
