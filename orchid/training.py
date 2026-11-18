"""Token-weighted single-device training and complete checkpoints."""

from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from .data import IGNORE
from .model import cross_entropy_sum


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 10
    micro_batch: int = 32
    accumulation: int = 4
    lr: float = 3e-4
    min_lr_fraction: float = .1
    warmup_fraction: float = .03
    weight_decay: float = .1
    betas: tuple[float, float] = (.9, .95)
    eps: float = 1e-8
    grad_clip: float = 1.
    eval_every: int = 0
    patience: int = 3
    min_improvement: float = .0001

    def __post_init__(self):
        if min(self.epochs, self.micro_batch, self.accumulation) < 1:
            raise ValueError("epochs, micro_batch and accumulation must be positive")
        if (not 0 < self.lr or not 0 <= self.warmup_fraction < 1
                or not 0 < self.min_lr_fraction <= 1 or not 0 < self.grad_clip
                or min(self.weight_decay, self.eval_every, self.patience, self.min_improvement) < 0):
            raise ValueError("Invalid optimizer, logging or early stopping configuration")
        object.__setattr__(self, "betas", tuple(self.betas))


def runtime_settings(config):
    requested = config.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU and CUDA are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else torch.cuda.current_device())
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    precision = config.get("precision", "auto")
    if precision == "auto":
        precision = "bf16" if bf16 else "fp32"
    if precision not in ("fp32", "bf16") or (precision == "bf16" and not bf16):
        raise ValueError("BF16 requires a CUDA device with native support; CPU uses FP32")
    return device, precision


def seed_everything(seed, device, config):
    if not 0 <= seed < 2**32:
        raise ValueError("Seed must be in [0, 2**32)")
    deterministic = bool(config.get("deterministic", False))
    threads = int(config.get("cpu_threads", 2))
    if threads < 1:
        raise ValueError("cpu_threads must be positive")
    if deterministic and device.type == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision(config.get("fp32_matmul_precision", "highest"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def autocast(precision):
    return torch.amp.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_adamw(model, cfg, device):
    decay, no_decay = [], []
    for p in model.parameters():  # parameters() only yields the tied parameter once.
        if p.requires_grad:
            (decay if p.ndim > 1 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, fused=device.type == "cuda",
    )


def schedule(optimizer, cfg, steps):
    warmup = min(max(0, steps - 1), round(steps * cfg.warmup_fraction))

    def factor(position):
        if position < warmup:
            return (position + 1) / warmup
        progress = min(1., (position - warmup) / max(1, steps - 1 - warmup))
        return cfg.min_lr_fraction + (1 - cfg.min_lr_fraction) * (1 + math.cos(progress * math.pi)) / 2

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def batched_steps(loader, accumulation):
    iterator = iter(loader)
    while batches := list(islice(iterator, accumulation)):
        yield batches


def move_batch(batch, device):
    return tuple(t.to(device, non_blocking=device.type == "cuda") for t in batch)


def optimize(model, batches, optimizer, scheduler, cfg, device, precision):
    target_count = sum(int((y != IGNORE).sum().item()) for _, y in batches)
    if target_count <= 0 or any(not (y != IGNORE).any() for _, y in batches):
        raise ValueError("No valid target tokens in at least one microbatch")
    used_lr = optimizer.param_groups[0]["lr"]
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device, dtype=torch.float64)
    for batch in batches:
        x, y = move_batch(batch, device)
        with autocast(precision):
            loss = cross_entropy_sum(model(x), y)
        (loss / target_count).backward()
        total += loss.detach()
    nll = float(total.item() / target_count)
    if not math.isfinite(nll):
        raise FloatingPointError("Nonfinite training loss")
    norm = float(torch.nn.utils.clip_grad_norm_(
        model.parameters(), cfg.grad_clip, error_if_nonfinite=True).item())
    optimizer.step()
    scheduler.step()
    return nll, target_count, norm, used_lr


@torch.inference_mode()
def evaluate_nll(model, loader, device, precision="fp32"):
    was_training = model.training
    model.eval()
    total, n = torch.zeros((), device=device, dtype=torch.float64), 0
    try:
        for cpu_x, cpu_y in loader:
            count = int((cpu_y != IGNORE).sum().item())
            if count == 0:
                raise ValueError("Validation microbatch has no valid targets")
            x, y = move_batch((cpu_x, cpu_y), device)
            with autocast(precision):
                total += cross_entropy_sum(model(x), y)
            n += count
        if n <= 0:
            raise ValueError("Empty validation set")
        mean = float(total.item() / n)
        if not math.isfinite(mean):
            raise FloatingPointError("Nonfinite validation loss")
        return {"nll": mean, "ppl": math.exp(mean), "target_tokens": n}
    finally:
        model.train(was_training)


def rng_state(device):
    n = np.random.get_state()
    return {
        "python": random.getstate(), "numpy": (n[0], n[1].tolist(), n[2], n[3], n[4]),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(saved, device):
    random.setstate(saved["python"])
    n = saved["numpy"]
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(saved["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(saved["cuda"], device=device)


def save_checkpoint(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as out:
        tmp = Path(out.name)
        try:
            torch.save(data, out)
            out.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("version") != 1:
        raise ValueError(f"Not an Orchid v1 checkpoint: {path}")
    return checkpoint
