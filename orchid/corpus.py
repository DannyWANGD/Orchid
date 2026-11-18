"""Pinned Tang corpus, sentence normalization and split before any learning."""

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random
import time
from unicodedata import normalize
from urllib.error import URLError
from urllib.parse import quote
from urllib.request import urlopen

import regex

from .io import read_json, sha256_file, sha256_text, write_bytes, write_json, write_jsonl


HAN = r"[\p{Unified_Ideograph}〇]"
IS_HAN_LINE = regex.compile(HAN + "+")
SOURCE_SEPARATORS = regex.compile(r"[，。！？；,.!?;\r\n]+")
SPLITS = ("train", "validation", "test")
CORPUS_RULE = "tang-nfc-sentence-v1"


def source_lines(paragraphs) -> tuple[tuple[str, ...], str | None]:
    if (not isinstance(paragraphs, list) or not paragraphs
            or any(not isinstance(p, str) or not p.strip() for p in paragraphs)):
        return (), "invalid_paragraphs"
    parts = SOURCE_SEPARATORS.split(normalize("NFC", "\n".join(paragraphs)))
    lines = tuple(p.strip() for p in parts if p.strip())
    if not lines:
        return (), "no_sentences"
    if any(not IS_HAN_LINE.fullmatch(p) for p in lines):
        return (), "missing_glyph" if any("□" in p for p in lines) else "non_han_content"
    return lines, None


def is_wujue(lines):
    return len(lines) == 4 and all(len(line) == 5 for line in lines)


def canonical_poem(lines):
    return "".join(line + ("，" if i % 2 == 0 and i < len(lines) - 1 else "。")
                   for i, line in enumerate(lines))


def fetch(url, retries=3):
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=30) as response:
                return response.read()
        except (OSError, URLError):
            if attempt + 1 == retries:
                raise
            time.sleep(attempt + 1)
    raise AssertionError("unreachable")


def snapshot(config, *, offline=False):
    source = config["source"]
    path = Path(config["paths"]["raw"])
    manifest_path = Path(config["paths"]["artifacts"]) / "data_manifest.json"
    old = read_json(manifest_path) if manifest_path.exists() else None
    if old and old["source"] != source:
        raise ValueError("Raw source changed. Use a different artifacts directory for a new dataset.")
    expected = {r["name"]: r["sha256"] for r in old["files"]} if old else {}
    base = f"https://raw.githubusercontent.com/{source['repo']}/{source['revision']}/"
    base += quote(source["directory"]) + "/"
    names = [f"poet.tang.{1000 * i}.json" for i in range(source["tang_shards"])]

    def materialize(name):
        target = path / name
        url = base + quote(name)
        # Unverified cached data never become the reference copy for this revision.
        if target.is_file() and name in expected:
            digest = sha256_file(target)
            if digest == expected[name]:
                return {"name": name, "url": url, "bytes": target.stat().st_size, "sha256": digest}
            if offline:
                raise ValueError(f"Checksum mismatch: {target}")
        if offline:
            raise FileNotFoundError(f"No verified copy of {target}")
        content = fetch(url)
        digest = hashlib.sha256(content).hexdigest()
        if name in expected and digest != expected[name]:
            raise ValueError(f"Pinned upstream checksum mismatch: {name}")
        if not isinstance(json.loads(content), list):
            raise ValueError(f"Expected an array in {url}")
        write_bytes(target, content)
        return {"name": name, "url": url, "bytes": len(content), "sha256": digest}

    with ThreadPoolExecutor(max_workers=8) as pool:
        files = list(pool.map(materialize, names))
    manifest = {
        "source": source, "files": files, "retrieved_on": old["retrieved_on"] if old else date.today().isoformat(),
        "use": "Tang dynasty poem text; edition and attribution retained as source metadata",
        "license": f"https://github.com/{source['repo']}/blob/{source['revision']}/LICENSE",
    }
    write_json(manifest_path, manifest)
    return manifest


def edit_within_two(a, b):
    """Exact Levenshtein <= 2 for equal-length strings, including displaced chars."""
    if len(a) != len(b):
        raise ValueError("The near-duplicate protocol compares equal-length bodies only")
    if sum(x != y for x, y in zip(a, b)) <= 2:
        return True
    # The only other way to spend two edits with equal length is delete + insert.
    deleted = {a[:i] + a[i + 1:] for i in range(len(a))}
    return any(b[:i] + b[i + 1:] in deleted for i in range(len(b)))


def two_deletions(body):
    for i, j in combinations(range(len(body)), 2):
        yield body[:i] + body[i + 1:j] + body[j + 1:]


def duplicate_edges(rows):
    """Enumerate aligned-lines, <=2 edits (without losing candidates), and embeddings."""
    five = [i for i, row in enumerate(rows) if row["form"] == "wujue"]
    by_three, by_deleted, by_lines = defaultdict(list), defaultdict(list), {}
    for i in five:
        r = rows[i]
        ls = tuple(r["lines"])
        by_lines[ls] = i
        hits = {}
        for skip in range(4):
            key = (skip, ls[:skip] + ls[skip + 1:])
            for j in by_three[key]:
                hits[j] = "three_aligned_lines"
            by_three[key].append(i)
        checked = set(hits)
        for key in set(two_deletions(r["body"])):
            for j in by_deleted[key]:
                if j not in checked and edit_within_two(r["body"], rows[j]["body"]):
                    hits[j] = "edit_distance_leq_2"
                checked.add(j)
            by_deleted[key].append(i)
        for j, reason in sorted(hits.items()):
            yield j, i, reason
    # Release the large deletion index before walking the entire cleaned corpus.
    del by_deleted
    for i, row in enumerate(rows):
        ls = row["lines"]
        if len(ls) > 4:
            for k in range(len(ls) - 3):
                if (j := by_lines.get(tuple(ls[k:k + 4]))) is not None:
                    yield j, i, "embedded_four_lines"


def assign_splits(rows, edges, ratios, seed):
    parents = list(range(len(rows)))

    def root(i):
        while i != parents[i]:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, j, _ in edges:
        a, b = sorted((root(i), root(j)))
        parents[b] = a
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[root(i)].append(i)
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1) > 1e-9 or min(ratios.values()) <= 0:
        raise ValueError("Expected positive train/validation/test fractions adding to 1")
    assigned = {s: [] for s in SPLITS}
    multi = {}
    rng = random.Random(seed)
    for has_four in (False, True):
        subset = [(r, ids) for r, ids in groups.items()
                  if has_four == any(rows[i]["form"] == "wujue" for i in ids)]
        rng.shuffle(subset)
        n = len(subset)
        val_end = round(n * ratios["validation"])
        test_end = val_end + round(n * ratios["test"])
        for pos, (r, ids) in enumerate(subset):
            split = "validation" if pos < val_end else "test" if pos < test_end else "train"
            member_ids = sorted(rows[i]["poem_id"] for i in ids)
            group_id = rows[r]["poem_id"]
            if len(ids) > 1:
                multi[group_id] = member_ids
            for i in ids:
                rows[i].update(group_id=group_id, split=split)
                assigned[split].append(rows[i])
    for rows_in_split in assigned.values():
        rows_in_split.sort(key=lambda r: r["poem_id"])
    for i, j, _ in edges:
        if rows[i]["split"] != rows[j]["split"]:
            raise AssertionError("Near-duplicate crosses a split")
    for a, b in combinations(SPLITS, 2):
        for key in ("body_sha256", "group_id"):
            if {r[key] for r in assigned[a]} & {r[key] for r in assigned[b]}:
                raise AssertionError(f"{key} overlaps {a} and {b}")
    return assigned, multi


def prepare(config, manifest):
    raw, proc, art = (Path(config["paths"][p]) for p in ("raw", "processed", "artifacts"))
    counts = Counter()
    rejected, exact = [], {}
    for file in manifest["files"]:
        for index, row in enumerate(read_json(raw / file["name"])):
            lines, reason = source_lines(row.get("paragraphs"))
            counts.update(raw=1, with_tags=bool(row.get("tags")), with_keywords="keywords" in row)
            src = {"file": file["name"], "index": index, "id": row.get("id"),
                   "title": row.get("title"), "author": row.get("author")}
            if reason:
                rejected.append(dict(src, reason=reason, paragraphs=row.get("paragraphs")))
                continue
            wujue = is_wujue(lines)
            counts.update(clean=1, wujue_raw=wujue, wujue_with_tags=wujue and bool(row.get("tags")))
            key = "".join(lines)
            if key in exact:
                exact[key]["sources"].append(src)
                if tuple(exact[key]["lines"]) != lines:
                    exact[key]["alternatives"].append({"source": src, "lines": lines})
                    # Prefer the version with clear 5x4 punctuation, then keep original shard order.
                    if wujue and exact[key]["form"] != "wujue":
                        exact[key].update(lines=lines, text=canonical_poem(lines), form="wujue",
                                          canonical_source=src, paragraphs=row["paragraphs"])
                continue
            sha = sha256_text(key)
            exact[key] = {
                "poem_id": sha[:24], "body_sha256": sha, "body": key, "lines": lines,
                "text": canonical_poem(lines), "form": "wujue" if wujue else "other",
                "sources": [src], "canonical_source": src, "paragraphs": row["paragraphs"],
                "alternatives": [],
            }
    rows = sorted(exact.values(), key=lambda r: r["poem_id"])
    if len({r["poem_id"] for r in rows}) != len(rows):
        raise AssertionError("poem_id truncated hash collision")
    edges = list(duplicate_edges(rows))
    assigned, multi = assign_splits(rows, edges, config["split"], config["seed"])
    stats = {
        "rule": CORPUS_RULE, "sources": len(manifest["files"]), **counts,
        "unique": len(rows), "wujue_unique": sum(r["form"] == "wujue" for r in rows),
        "unique_han_chars": sum(len(r["body"]) for r in rows),
        "alternate_sentence_boundaries": sum(bool(r["alternatives"]) for r in rows),
        "reject_reasons": Counter(r["reason"] for r in rejected),
        "near_duplicate_edges": Counter(reason for _, _, reason in edges),
        "groups_with_multiple_bodies": len(multi),
        "splits": {s: {
            "poems": len(items), "wujue_candidates": sum(r["form"] == "wujue" for r in items),
            "han_chars": sum(len(r["body"]) for r in items),
            "length_histogram_han": dict(sorted(Counter(len(r["body"]) for r in items).items())),
        } for s, items in assigned.items()},
    }
    split_manifest = {
        "rule": CORPUS_RULE, "seed": config["seed"], "split_ratios": config["split"],
        "data_manifest_sha256": sha256_file(art / "data_manifest.json"),
        "splits": {s: [r["poem_id"] for r in items] for s, items in assigned.items()},
        "non_singleton_groups": multi,
    }
    split_path = art / "split_manifest.json"
    if split_path.exists() and read_json(split_path) != split_manifest:
        raise ValueError("Recomputed split differs from the frozen manifest. Use a new artifacts directory to change it.")
    for s, items in assigned.items():
        write_jsonl(proc / f"{s}.jsonl", items)
    write_jsonl(proc / "excluded.jsonl", rejected)
    write_jsonl(proc / "near_duplicates.jsonl",
                ({"left": rows[i]["poem_id"], "right": rows[j]["poem_id"],
                  "rule": reason} for i, j, reason in edges))
    write_json(split_path, split_manifest, indent=None)
    write_json(art / "data_stats.json", stats)
    rng = random.Random(config["seed"])
    review = {
        "seed": config["seed"],
        "candidates": rng.sample([r for r in rows if r["form"] == "wujue"], k=min(50, stats["wujue_unique"])),
        "excluded": rng.sample(rejected, k=min(50, len(rejected))),
        "edges": [{"rule": reason, "left": rows[i], "right": rows[j]}
                  for i, j, reason in rng.sample(edges, k=min(16, len(edges)))],
        "alternate_boundaries": [r for r in rows if r["alternatives"]],
    }
    write_json(proc / "format_audit.json", review)
    return stats
