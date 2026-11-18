"""An NFC character vocabulary learned from training text only."""

from collections.abc import Iterable, Sequence
from pathlib import Path
from unicodedata import normalize

from .io import read_json, sha256_file, write_json


SPECIAL = ("<pad>", "<unk>", "<bos>", "<eos>", "<wujue>", "<kw>", "<kwsep>", "<sep>")
PAD, UNK, BOS, EOS, FORM, KW, KWSEP, SEP = range(len(SPECIAL))
TOKENIZER_VERSION = "orchid-char-nfc-v1"


class CharacterTokenizer:
    def __init__(self, vocab: Sequence[str]):
        self.vocab = tuple(vocab)
        if (self.vocab[:len(SPECIAL)] != SPECIAL
                or any(len(c) != 1 or normalize("NFC", c) != c for c in self.vocab[len(SPECIAL):])
                or len(set(self.vocab)) != len(self.vocab)):
            raise ValueError("Invalid Orchid character vocabulary")
        self.ids = {c: i for i, c in enumerate(self.vocab)}

    @classmethod
    def fit(cls, training_texts: Iterable[str]):
        chars = set()
        for text in training_texts:
            if any(s in text for s in SPECIAL):
                raise ValueError("Reserved marker in corpus text")
            chars.update(normalize("NFC", text))
        if not chars:
            raise ValueError("Cannot fit a vocabulary without training text")
        return cls((*SPECIAL, *sorted(chars)))

    @classmethod
    def load(cls, path: Path):
        record = read_json(path)
        if record["version"] != TOKENIZER_VERSION or record["normalization"] != "NFC":
            raise ValueError(f"Unsupported tokenizer: {path}")
        token = cls(record["id_to_token"])
        if record["special_ids"] != {s: token.ids[s] for s in SPECIAL}:
            raise ValueError("Token IDs do not match the fixed special IDs")
        return token

    def save(self, path: Path):
        write_json(path, {
            "version": TOKENIZER_VERSION, "normalization": "NFC",
            "id_to_token": self.vocab, "special_ids": {s: self.ids[s] for s in SPECIAL},
        })
        return sha256_file(path)

    def __len__(self):
        return len(self.vocab)

    def encode(self, text: str, *, strict=False) -> list[int]:
        if any(s in text for s in SPECIAL):
            raise ValueError("Reserved markers must be explicitly encoded as special IDs")
        text = normalize("NFC", text)
        if strict:
            missing = sorted({c for c in text if c not in self.ids})
            if missing:
                raise ValueError("Characters outside the training vocabulary: " + "、".join(missing))
        return [self.ids.get(c, UNK) for c in text]

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for i in ids:
            if not 0 <= i < len(self.vocab):
                raise ValueError(f"Out-of-range token ID: {i}")
            out.append(self.vocab[i])
        return "".join(out)
