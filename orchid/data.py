"""Pretraining windows and response-only supervision for one causal GPT."""

from math import ceil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .io import read_jsonl
from .tokenizer import BOS, EOS, FORM, KW, KWSEP, PAD, SEP


IGNORE = -100


class TokenWindows(Dataset):
    def __init__(self, tokens: np.ndarray, context: int):
        if context < 1 or tokens.ndim != 1 or len(tokens) < 2:
            raise ValueError("A window dataset needs a 1D stream of at least two tokens")
        self.tokens, self.context = tokens, context

    @classmethod
    def load(cls, path: Path, context: int):
        return cls(np.load(path, mmap_mode="r", allow_pickle=False), context)

    def __len__(self):
        return ceil((len(self.tokens) - 1) / self.context)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        start = index * self.context
        chunk = torch.tensor(self.tokens[start:start + self.context + 1], dtype=torch.long)
        count = len(chunk) - 1
        x = torch.full((self.context,), PAD, dtype=torch.long)
        y = torch.full((self.context,), IGNORE, dtype=torch.long)
        x[:count], y[:count] = chunk[:-1], chunk[1:]
        return x, y


def prompt_ids(tokenizer, keywords, *, strict=True):
    ids = [BOS, FORM, KW]
    for k, word in enumerate(keywords):
        if k:
            ids.append(KWSEP)
        ids.extend(tokenizer.encode(word, strict=strict))
    return ids + [SEP]


def response_item(tokenizer, keywords, response, *, strict=False):
    prompt = prompt_ids(tokenizer, keywords, strict=strict)
    seq = torch.tensor([*prompt, *tokenizer.encode(response, strict=strict), EOS], dtype=torch.long)
    x, y = seq[:-1], seq[1:].clone()
    # y[i] predicts seq[i+1]. The first answer character is at seq[len(prompt)].
    y[:len(prompt) - 1] = IGNORE
    return x, y


class Responses(Dataset):
    def __init__(self, path, tokenizer, context, split, *, indices=None):
        self.items, self.excluded_overlong = [], 0
        for index, row in enumerate(read_jsonl(Path(path))):
            if row["split"] != split:
                raise ValueError(f"Wrong split in {path}: {row['poem_id']}")
            if indices is not None and index not in indices:
                continue
            if not row["response"]:
                raise ValueError(f"Empty response in {path}: {row['poem_id']}")
            item = response_item(tokenizer, row["keywords"], row["response"], strict=split == "train")
            if len(item[0]) > context:
                self.excluded_overlong += 1
            else:
                self.items.append(item)
        if not self.items:
            raise ValueError(f"No usable supervision in {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_responses(batch):
    xs, ys = zip(*batch)
    return (torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=PAD),
            torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=IGNORE))


def batch_loader(dataset, indices, *, batch_size, pin_memory=False):
    # A local generator prevents DataLoader's worker-seed setup consuming model RNG.
    return DataLoader(dataset, batch_size=batch_size, sampler=indices, num_workers=0,
                      pin_memory=pin_memory, generator=torch.Generator().manual_seed(0),
                      collate_fn=collate_responses if isinstance(dataset, Responses) else None)
