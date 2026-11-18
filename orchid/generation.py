"""Sampling only the last-position logits; preserve every generated token."""

from dataclasses import dataclass
import math

import torch

from .tokenizer import EOS


@dataclass(frozen=True)
class Sampling:
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 1.0
    max_new_tokens: int = 64

    def __post_init__(self):
        if (not math.isfinite(self.temperature) or self.temperature < 0
                or not 0 < self.top_p <= 1 or self.top_k < 0 or self.max_new_tokens < 1):
            raise ValueError("Invalid temperature, top-k, top-p or token budget")


def draw(logits, params, rng):
    if params.temperature == 0:
        return int(logits.argmax().item())
    logits = logits.float() / params.temperature
    if 0 < params.top_k < logits.numel():
        logits, vocab_indices = logits.topk(params.top_k)
    else:
        vocab_indices = torch.arange(logits.numel(), device=logits.device)
    if params.top_p < 1:
        logits, order = logits.sort(descending=True)
        vocab_indices = vocab_indices[order]
        probs = logits.softmax(dim=-1)
        logits = logits.masked_fill(probs.cumsum(dim=-1) - probs >= params.top_p, -torch.inf)
    idx = torch.multinomial(logits.softmax(dim=-1), 1, generator=rng)
    return int(vocab_indices[idx].item())


@torch.inference_mode()
def sample_tokens(model, prompt, params, seed, *, allowed_at=None):
    if not prompt or len(prompt) + params.max_new_tokens > model.config.context_length:
        raise ValueError("Prompt and generation budget exceed the model context or prompt is empty")
    if model.training:
        raise ValueError("Sampling requires model.eval()")
    device = model.token_embedding.weight.device
    rng = torch.Generator(device=device).manual_seed(seed)
    buffer = torch.empty((1, len(prompt) + params.max_new_tokens), device=device, dtype=torch.long)
    buffer[0, :len(prompt)] = torch.tensor(prompt, device=device, dtype=torch.long)
    generated = []
    for step in range(params.max_new_tokens):
        end = len(prompt) + step
        logits = model(buffer[:, :end], last_only=True)[0, -1]
        if allowed_at is not None:
            allowed = allowed_at(step, generated)
            if not allowed:
                raise ValueError(f"No allowed tokens at generation step {step}")
            masked = torch.full_like(logits, -torch.inf)
            masked[list(allowed)] = logits[list(allowed)]
            logits = masked
        selected = draw(logits, params, rng)
        if selected == EOS:
            return generated, "eos"
        generated.append(selected)
        buffer[0, end] = selected
    return generated, "limit"
