"""Load a checked Orchid checkpoint or portable inference bundle."""

from dataclasses import dataclass
from pathlib import Path
import random
from time import perf_counter

import torch

from .corpus import IS_HAN_LINE
from .data import prompt_ids
from .generation import Sampling, sample_tokens
from .io import read_json, sha256_file
from .metrics import judge_format_keywords
from .model import GPT, GPTConfig
from .supervision import Task
from .tokenizer import EOS, CharacterTokenizer
from .training import load_checkpoint, synchronize


@dataclass
class Poet:
    model: GPT
    tokenizer: CharacterTokenizer
    task: Task
    meta: dict
    device: torch.device
    checkpoint_sha256: str
    load_seconds: float

    def tokens(self, keywords, sampling: Sampling, seed, *, constrain_form=False, constrain_keywords=False):
        words = self.task.validate_keywords(keywords)
        prefix = prompt_ids(self.tokenizer, words)
        if constrain_form or constrain_keywords:
            allowed_han = tuple(i for i, c in enumerate(self.tokenizer.vocab) if IS_HAN_LINE.fullmatch(c))
            fixed = {}
            if constrain_keywords:
                # Put each word in a separate sentence so all keywords fit even
                # when the caller requests three two-character words.
                positions = random.Random(seed).sample(range(self.task.lines), len(words))
                for line, keyword in zip(positions, words):
                    ids = self.tokenizer.encode(keyword, strict=True)
                    offset = (self.task.chars_per_line - len(ids)) // 2
                    for i, token_id in enumerate(ids):
                        fixed[6 * line + offset + i] = token_id

            def allowed_at(step, _):
                if step in fixed:
                    return (fixed[step],)
                if step == 24:
                    return (EOS,)
                if step % 6 == 5:
                    return (self.tokenizer.ids["，" if step % 12 == 5 else "。"],)
                return allowed_han

            if sampling.max_new_tokens < 25:
                raise ValueError("Format constraint requires a budget of at least 25 tokens")
        else:
            allowed_at = None
        synchronize(self.device)
        t0 = perf_counter()
        ids, stop = sample_tokens(self.model, prefix, sampling, seed, allowed_at=allowed_at)
        synchronize(self.device)
        elapsed = perf_counter() - t0
        text = self.tokenizer.decode(ids)
        if constrain_keywords and (stop != "eos" or not judge_format_keywords(text, words)["joint_ok"]):
            raise RuntimeError("The keyword-constrained decoder failed its output invariant")
        return {"generated": text, "generated_ids": ids,
                "stop": stop, "seconds": elapsed, "generated_steps": len(ids) + (stop == "eos")}


def verify_bundle(directory):
    index = read_json(directory / "bundle.json")
    if index["version"] != 1:
        raise ValueError("Unsupported inference bundle")
    for name in ("weights.pt", "tokenizer.json", "config.json"):
        if sha256_file(directory / name) != index["files"][name]:
            raise ValueError(f"Corrupt or mismatched bundle file: {name}")
    return index


def load_poet(checkpoint, device="auto", *, tokenizer_path=None):
    started = perf_counter()
    checkpoint = Path(checkpoint)
    device = torch.device("cuda" if (device == "auto" and torch.cuda.is_available())
                          else "cpu" if device == "auto" else device)
    if device.type not in ("cpu", "cuda") or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Unavailable or unsupported inference device: {device}")
    if checkpoint.is_dir():
        index = verify_bundle(checkpoint)
        source = checkpoint / "weights.pt"
        tokenizer_path = checkpoint / "tokenizer.json"
        meta = read_json(checkpoint / "config.json")
        state = torch.load(source, map_location="cpu", weights_only=True)
        checkpoint_sha = index["files"]["weights.pt"]
    else:
        source = checkpoint
        checkpoint_sha = sha256_file(source)
        full = load_checkpoint(source)
        meta, state = full["meta"], full["model"]
        del full
        tokenizer_path = Path(tokenizer_path or meta["tokenizer_path"])
    if sha256_file(tokenizer_path) != meta["lineage"]["tokenizer_sha256"]:
        raise ValueError("Tokenizer hash does not match the checkpoint")
    token = CharacterTokenizer.load(tokenizer_path)
    config = GPTConfig(**meta["model"])
    if config.vocab_size != len(token):
        raise ValueError("Checkpoint vocabulary size mismatch")
    model = GPT(config)
    model.load_state_dict(state)
    model.to(device).eval()
    synchronize(device)
    return Poet(model, token, Task(**meta["resolved"]["task_config"]), meta, device,
                checkpoint_sha, perf_counter() - started)
