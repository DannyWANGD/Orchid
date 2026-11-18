"""Constrain four-by-five form and distinct even-line Pingshui rhymes."""

from collections import defaultdict
from dataclasses import replace
import random

import torch

from .corpus import IS_HAN_LINE
from .generation import sample_tokens
from .metrics import bundled_rhyme_table
from .tokenizer import EOS


class RhymeDecoder:
    def __init__(self, tokenizer):
        self.table = bundled_rhyme_table()
        self.han = tuple(i for i, c in enumerate(tokenizer.vocab)
                         if len(c) == 1 and IS_HAN_LINE.fullmatch(c))
        self.punctuation = (tokenizer.ids["，"], tokenizer.ids["。"])
        groups = defaultdict(list)
        for i in self.han:
            readings = self.table.get(tokenizer.vocab[i])
            if readings and len(readings) == 1:
                tone, group = next(iter(readings))
                if tone == "ping":
                    groups[group].append(i)
        self.partners = {i: tuple(other for other in ids if other != i)
                         for ids in groups.values() if len(ids) > 1 for i in ids}
        self.first_rhyme = tuple(sorted(self.partners))
        if not self.first_rhyme:
            raise ValueError("No unambiguous Pingshui rhyme pairs in the model vocabulary")

    def sample(self, model, prompt, sampling, seed, placements):
        def allowed_at(step, generated):
            if step == 24:
                return (EOS,)
            line, offset = divmod(step, 6)
            if offset == 5:
                return (self.punctuation[line % 2],)
            if step == 10:
                return self.first_rhyme
            if step == 22:
                return self.partners[generated[10]]
            if line in placements:
                start, word = placements[line]
                if start <= offset < start + len(word):
                    already_used = any(generated[i:i + len(word)] == word
                                       for i in range(len(generated) - len(word) + 1))
                    if not already_used:
                        return (word[offset - start],)
            return self.han

        return sample_tokens(model, prompt, sampling, seed, allowed_at=allowed_at)

    @torch.inference_mode()
    def generate(self, model, prompt, sampling, seed, word_ids, candidate_count):
        if sampling.max_new_tokens < 25:
            raise ValueError("Rhyme decoding requires a budget of at least 25 tokens")
        if not 1 <= candidate_count <= 16:
            raise ValueError("Rhyme decoding requires 1–16 candidates")
        sampling = replace(sampling, max_new_tokens=25)
        rng = random.Random(seed)
        candidates = []
        for index in range(candidate_count):
            placements = {}
            for word, line in zip(word_ids, rng.sample(range(4), len(word_ids))):
                # Even lines reserve the last character for the rhyme.
                length = 5 - (line % 2)
                placements[line] = rng.randrange(length - len(word) + 1), word
            candidate = self.sample(model, prompt, sampling, (seed + index) % 2**64, placements)
            if candidate[1] != "eos" or len(candidate[0]) != 24:
                raise RuntimeError("Rhyme decoding failed to produce four complete lines")
            candidates.append(candidate)
        if len(candidates) == 1:
            return candidates[0]

        # Score complete candidates under the original logits, including forced tokens.
        batch = torch.tensor([prompt + ids + [EOS] for ids, _ in candidates],
                             device=model.token_embedding.weight.device)
        targets = batch[:, len(prompt):]
        logits = model(batch[:, :-1])[:, len(prompt) - 1:]
        log_probs = logits.float().log_softmax(dim=-1)
        likelihood = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean(-1).tolist()
        best = max(range(len(candidates)),
                   key=lambda i: likelihood[i] - repetition_penalty(candidates[i][0]))
        return candidates[best]


def repetition_penalty(ids):
    lines = [tuple(ids[i:i + 5]) for i in range(0, 24, 6)]
    bigrams = [pair for line in lines for pair in zip(line, line[1:])]
    return 1.5 * (len(lines) - len(set(lines))) + 0.25 * (len(bigrams) - len(set(bigrams)))
