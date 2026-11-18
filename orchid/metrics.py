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
