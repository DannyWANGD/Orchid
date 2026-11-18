"""Parse unmodified poem generations without silently fixing malformed text."""

from unicodedata import normalize

import regex

from .corpus import HAN, is_wujue, canonical_poem


PUNCT = r"[，。！？；,.!?;]"
SEPARATOR = rf"(?:{PUNCT}\n?|\n)"
PARSED = regex.compile(rf"{HAN}+(?:{SEPARATOR}{HAN}+)*{PUNCT}?")
SPLIT_LINES = regex.compile(SEPARATOR)


def parse_poem(text):
    normalized = normalize("NFC", text.replace("\r\n", "\n")).strip()
    if not PARSED.fullmatch(normalized):
        return ()
    return tuple(s for s in SPLIT_LINES.split(normalized) if s)


def judge_format_keywords(text, keywords):
    lines = parse_poem(text)
    hits = [w for w in keywords if any(w in line for line in lines)]
    form = is_wujue(lines)
    kws = bool(lines) and len(hits) == len(keywords)
    return {"lines": lines, "parsed": bool(lines), "format_ok": form, "keywords_ok": kws,
            "joint_ok": form and kws, "keyword_hits": hits,
            "canonical_punctuation": form and normalize("NFC", text).strip() == canonical_poem(lines)}


def rhyme_table(path):
    from collections import defaultdict

    from .io import read_json

    map_ = defaultdict(set)
    for tone, divisions in read_json(path).items():
        for division in divisions.values():
            for name, strings in division.items():
                for char in regex.findall(HAN, "".join(strings)):
                    map_[char].add((tone, name))
    return map_


def rhyme_status(lines, table):
    if not is_wujue(lines):
        return {"rhyme": None}
    x, y = lines[1][-1], lines[3][-1]
    a, b = table.get(x, set()), table.get(y, set())
    matches = sorted(group for tone, group in a & b if tone == "ping")
    if not a or not b:
        status, cause = "unknown", "missing_form"
    elif len(a) == len(b) == len(matches) == 1:
        status, cause = "pass", None
    elif matches:
        status, cause = "unknown", "ambiguous_reading"
    else:
        status, cause = "fail", "no_common_level_rhyme"
    return {"rhyme": status, "rhyme_cause": cause, "rhyme_characters": [x, y],
            "ping_matches": matches, "repeated_rhyme_char": x == y}


class TrainMatches:
    def __init__(self, rows):
        from collections import defaultdict

        from .corpus import two_deletions

        self.exact, self.fragments, self.three, self.edits = {}, defaultdict(list), defaultdict(list), defaultdict(list)
        self.five = {}
        for row in rows:
            lines, body, idx = tuple(row["lines"]), row["body"], row["poem_id"]
            self.exact[body] = idx
            if is_wujue(lines):
                self.five[idx] = (lines, body)
                for k in range(4):
                    self.three[(k, lines[:k] + lines[k+1:])].append(idx)
                for piece in set(two_deletions(body)):
                    self.edits[piece].append(idx)
            elif len(lines) > 4:
                for k in range(len(lines)-3):
                    if is_wujue(ls := lines[k:k+4]):
                        self.fragments[ls].append({"poem_id": idx, "start_line": k+1})

    def find(self, lines):
        from .corpus import edit_within_two, two_deletions

        if not is_wujue(lines):
            return {}
        lines = tuple(lines)
        body = "".join(lines)
        exact = self.exact.get(body)
        others = {idx for k in range(4) for idx in self.three.get((k, lines[:k]+lines[k+1:]), [])}
        near = {idx: "three_aligned_lines" for idx in others if idx != exact}
        checked = {*others, exact}
        for key in set(two_deletions(body)):
            for idx in self.edits.get(key, []):
                if idx not in checked and edit_within_two(body, self.five[idx][1]):
                    near[idx] = "edit_distance_leq_2"
                checked.add(idx)
        return {"exact": exact, "longer_poem_fragments": self.fragments.get(lines, []),
                "near": [{"poem_id": i, "reason": near[i]} for i in sorted(near)]}
