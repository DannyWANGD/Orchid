"""Paired train-only subsets for the Pingshui continuation study."""

from collections import Counter, defaultdict
import random

from .io import json_text, read_jsonl, sha256_text
from .metrics import PINGSHUI_SHA256, bundled_rhyme_table, parse_poem, rhyme_status


def training_subset(path, policy, pairs, seed):
    if policy not in ("random", "rhyme") or not isinstance(pairs, int) or pairs < 1:
        raise ValueError("Expected training_subset policy random or rhyme and a positive pair count")
    rows = list(read_jsonl(path))
    if len(rows) < pairs or any(row["split"] != "train" for row in rows):
        raise ValueError("The verified training split must contain at least the requested pair count")
    table = bundled_rhyme_table()
    rhyme_indices = defaultdict(list)
    source_counts = Counter()
    for index, row in enumerate(rows):
        score = rhyme_status(parse_poem(row["response"]), table)
        status = "repeated" if score.get("repeated_rhyme_char") else score["rhyme"]
        if not status:
            raise ValueError(f"Non-five-by-four training target: {row['poem_id']}")
        source_counts[status] += 1
        if status == "pass":
            rhyme_indices[len(row["keywords"])].append(index)

    rng = random.Random(seed)
    control = rng.sample(range(len(rows)), pairs)
    quotas = Counter(len(rows[i]["keywords"]) for i in control)
    selected = control
    if policy == "rhyme":
        if any(len(rhyme_indices[k]) < n for k, n in quotas.items()):
            raise ValueError("The rhyme pool cannot match the control's keyword counts")
        selected = [i for k, n in sorted(quotas.items()) for i in rng.sample(rhyme_indices[k], n)]
    ids = sorted(selected)
    passing = {i for indices in rhyme_indices.values() for i in indices}
    return set(ids), {
        "policy": policy, "pairs": pairs, "seed": seed, "keyword_count": dict(quotas),
        "source_rhyme_counts": dict(source_counts), "selected_distinct_rhyme": len(passing.intersection(ids)),
        "selected_poem_ids_sha256": sha256_text(json_text([rows[i]["poem_id"] for i in ids], None)),
        "rhyme_sha256": PINGSHUI_SHA256,
    }
