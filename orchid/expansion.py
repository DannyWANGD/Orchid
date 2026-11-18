"""Add Song poems to train while protecting the frozen Tang evaluation splits."""

from collections import Counter, defaultdict
from itertools import chain
from pathlib import Path

import numpy as np

from .corpus import SPLITS, canonical_poem, edit_within_two, is_wujue, source_lines
from .io import read_json, read_jsonl, read_yaml, sha256_file, sha256_text, write_bytes, write_json, write_jsonl
from .runner import verified
from .tokenizer import BOS, EOS, UNK, CharacterTokenizer


RULE = "tang-song-train-only-v1"


class HoldoutFilter:
    """Exact fragments, aligned triples and <=2 full-body edits for same-length poems."""

    def __init__(self, heldout):
        self.full_prefix = defaultdict(set)
        self.heldout_fragments = set()
        self.triples = set()
        self.edit_anchors = defaultdict(lambda: defaultdict(list))
        for row in heldout:
            lines, body = tuple(row["lines"]), row["body"]
            n = len(body)
            if n >= 20:
                self.full_prefix[body[:8]].add(body)
                self.heldout_fragments.update(
                    "".join(lines[start:end])
                    for start in range(len(lines)) for end in range(start+1,min(start+6,len(lines)+1))
                    if len("".join(lines[start:end])) >= 20
                )
            self.triples.update(
                tuple(lines[i:i+3]) for i in range(len(lines)-2)
                if min(map(len,lines[i:i+3])) >= 5
            )
            if n >= 15:
                for i,j in zip((0,n//3,2*n//3), (n//3,2*n//3,n)):
                    self.edit_anchors[n][(i,j)].append((body[i:j],body))
        self.edit_anchors = {
            n: {segment: self._index(pairs) for segment,pairs in bucket.items()}
            for n,bucket in self.edit_anchors.items()
        }

    @staticmethod
    def _index(pairs):
        results = defaultdict(list)
        for key, body in pairs:
            results[key].append(body)
        return results

    def reason(self, lines, body):
        n = len(body)
        if n >= 20:
            if body in self.heldout_fragments:
                return "body_in_tang_holdout"
            for start in range(n-19):
                if any(body.startswith(ref,start) for ref in self.full_prefix.get(body[start:start+8], ())):
                    return "tang_holdout_body_in_song"
        if any(tuple(lines[i:i+3]) in self.triples for i in range(len(lines)-2)):
            return "shared_three_consecutive_lines"
        checked = set()
        for (i,j), bucket in self.edit_anchors.get(n, {}).items():
            for shift in (-1,0,1):
                if i+shift < 0 or j+shift > n:
                    continue
                for ref in bucket.get(body[i+shift:j+shift], ()):
                    if ref not in checked:
                        if edit_within_two(body, ref):
                            return "heldout_edit_distance_leq_2"
                        checked.add(ref)
        return None


def write_tokens(path, arr):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
    temp.replace(path)


def prepare(config):
    tang = read_yaml(Path(config["tang"]))
    song = read_yaml(Path(config["song"]))
    old_art, old_proc = (Path(tang["paths"][k]) for k in ("artifacts","processed"))
    art, proc = (Path(config["paths"][k]) for k in ("artifacts","processed"))
    raw = Path(song["paths"]["raw"])
    source = read_json(art/"song_source.json")
    if source["source"] != song["source"]:
        raise ValueError("Song source does not match configuration")
    tok_path = old_art/"tokenizer.json"
    ts = read_json(old_art/"token_stats.json")
    tok_sha = verified(tok_path, ts["tokenizer_sha256"])
    old_m = verified(old_art/"data_manifest.json", ts["data_manifest_sha256"])
    old_split = verified(old_art/"split_manifest.json", ts["split_manifest_sha256"])
    write_bytes(art/"tokenizer.json", tok_path.read_bytes())
    tok = CharacterTokenizer.load(tok_path)

    def tang_rows(s):
        path = old_proc/f"{s}.jsonl"
        verified(path, ts["splits"][s]["poem_file_sha256"])
        return list(read_jsonl(path))

    train = tang_rows("train")
    held = {s:tang_rows(s) for s in ("validation","test")}
    tang_bodies = {r["body"] for r in chain(train,*held.values())}
    exclusion = HoldoutFilter(list(chain.from_iterable(held.values())))
    unique = {}
    counts = Counter()
    exclusion_examples = defaultdict(list)

    def add_reason(reason, src, text=None):
        counts["rejected_"+reason] += 1
        if len(exclusion_examples[reason]) < 3:
            exclusion_examples[reason].append(dict(src, text=text))

    for file in source["files"]:
        path = raw/file["name"]
        verified(path, file["sha256"])
        for i,row in enumerate(read_json(path)):
            counts["raw"] += 1
            lines, reason = source_lines(row.get("paragraphs"))
            src = {"file": file["name"], "index": i, "id":row.get("id"),
                   "title": row.get("title"), "author": row.get("author")}
            if reason:
                counts["rejected_"+reason] += 1
                continue
            body = "".join(lines)
            if body in tang_bodies:
                add_reason("exact_tang_body",src,body)
                continue
            if body in unique:
                counts["exact_song_duplicates"] += 1
                unique[body]["sources"].append(src)
                continue
            if reason := exclusion.reason(lines,body):
                add_reason(reason,src,body)
                continue
            sha = sha256_text(body)
            unique[body] = {
                "poem_id":sha[:24], "body_sha256":sha, "body":body, "lines":lines, "text":canonical_poem(lines),
                "form":"wujue" if is_wujue(lines) else "other", "sources":[src], "canonical_source":src,
                "paragraphs":row["paragraphs"], "alternatives":[], "group_id":sha[:24], "split":"train",
            }

    additions = sorted(unique.values(),key=lambda r:r["poem_id"])
    del unique
    train = sorted([*train,*additions],key=lambda r:r["poem_id"])
    if len(train)!=len({r["poem_id"] for r in train}) or not additions:
        raise AssertionError("Song training set empty or poem hash collision")
    id_manifest = {
        "rule":RULE, "tang_split_sha256":old_split,
        "splits":{s:[r["poem_id"] for r in rows] for s,rows in {"train":train, **held}.items()},
        "song_train_poems":len(additions),
    }
    manifest = {
        "rule":RULE, "tang_data_manifest_sha256":old_m,
        "song_source_sha256":sha256_file(art/"song_source.json"),
        "tang_holdout_policy":"Tang validation and test bytes unchanged; Song train additions only.",
        "source":{"tang":tang["source"], "song":song["source"]},
    }
    write_json(art/"data_manifest.json",manifest)
    split_manifest = art/"split_manifest.json"
    if split_manifest.exists() and read_json(split_manifest) != id_manifest:
        raise ValueError("The derived Tang–Song train set changed")
    write_json(split_manifest,id_manifest,indent=None)
    ts_new = {
        "vocab_size":len(tok), "tokenizer_sha256":tok_sha, "tokenizer_from_tang_only":True,
        "split_manifest_sha256":sha256_file(split_manifest),
        "data_manifest_sha256":sha256_file(art/"data_manifest.json"), "splits":{},
    }
    proc.mkdir(parents=True,exist_ok=True)
    write_jsonl(proc/"train.jsonl",train)
    unk_chars = Counter()

    def token_rows():
        for row in train:
            text = row["text"]
            ids = tok.encode(text)
            unk_chars.update(c for c,i in zip(text,ids) if i==UNK)
            yield chain((BOS,),ids,(EOS,))

    arr = np.fromiter(chain.from_iterable(token_rows()),dtype=np.int32)
    token_file = proc/"tokens/train.npy"
    write_tokens(token_file,arr)
    text_count = len(arr)-2*len(train)
    ts_new["splits"]["train"] = {
        "poems":len(train), "stream_tokens":len(arr), "body_and_punctuation_tokens":text_count,
        "unk_count":sum(unk_chars.values()), "unk_types":len(unk_chars),
        "unk_fraction":sum(unk_chars.values())/text_count,
        "top_unk":unk_chars.most_common(12), "token_file_sha256":sha256_file(token_file),
        "poem_file_sha256":sha256_file(proc/"train.jsonl"),
    }
    for s in SPLITS[1:]:
        for filename,key in ((f"{s}.jsonl","poem_file_sha256"),(f"tokens/{s}.npy","token_file_sha256")):
            content = old_proc/filename
            verified(content, ts["splits"][s][key])
            write_bytes(proc/filename, content.read_bytes())
        ts_new["splits"][s] = ts["splits"][s]
    write_json(art/"token_stats.json",ts_new)
    stats = {
        "rule":RULE, **counts, "song_kept":len(additions),
        "song_four_five":sum(r["form"]=="wujue" for r in additions),
        "tang_train":len(train)-len(additions), "train_total":len(train), "train_tokens":len(arr),
        "exclusion_examples":dict(exclusion_examples), "validation_test_unchanged":True,
    }
    write_json(art/"expansion_stats.json",stats)
    return stats
