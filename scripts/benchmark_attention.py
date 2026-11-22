"""Measure Orchid attention implementations and output projection on identical weights."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from math import ceil
from pathlib import Path
import platform
from statistics import median
import sys
from time import perf_counter_ns

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.inference import load_poet
from orchid.io import write_json
from orchid.tokenizer import SPECIAL
from orchid.training import synchronize


def set_backend(model, backend):
    for block in model.blocks:
        block.attention.backend = backend


def measure_pair(model, ids, variants, warmups, repeats):
    device = ids.device
    samples = defaultdict(list)
    order = tuple(variants.items())
    for i in range(warmups + repeats):
        for name, (backend, last_only) in (order if i % 2 == 0 else reversed(order)):
            set_backend(model, backend)
            started = perf_counter_ns()
            model(ids, last_only=last_only)
            synchronize(device)
            if i >= warmups:
                samples[name].append((perf_counter_ns() - started) / 1e6)

    records = {}
    for name, (backend, last_only) in order:
        set_backend(model, backend)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        logits = model(ids, last_only=last_only)
        synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) - baseline if device.type == "cuda" else None
        records[name] = {
            "backend": backend, "last_only": last_only, "logits_shape": list(logits.shape),
            "logits_fp32_bytes": logits.numel() * logits.element_size(),
            "p50_ms": median(samples[name]),
            "p95_ms": sorted(samples[name])[ceil(repeats * .95) - 1],
            "samples_ms": samples[name], "peak_extra_allocated_bytes": peak,
        }
        del logits
    return records


def compare_full_logits(model, ids):
    set_backend(model, "sdpa")
    sdpa = model(ids)
    set_backend(model, "manual")
    manual = model(ids)
    return {
        "max_abs_difference": (sdpa - manual).abs().max().item(),
        "argmax_agreement": (sdpa.argmax(-1) == manual.argmax(-1)).sum().item(),
        "positions": ids.numel(),
    }


def causal_check(model, ids):
    prefix = ids.size(1) // 2
    mutated = ids.clone()
    low, high = len(SPECIAL), model.config.vocab_size
    mutated[:, prefix:] = low + (ids[:, prefix:] - low + 1) % (high - low)
    errors = {}
    for backend in ("sdpa", "manual"):
        set_backend(model, backend)
        before = model(ids)[:, :prefix]
        after = model(mutated)[:, :prefix]
        errors[backend] = (before - after).abs().max().item()
    return {"prefix_positions": prefix, "edited_suffix_tokens": ids.size(0) * (ids.size(1) - prefix),
            "max_abs_prefix_logit_change": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--lengths", nargs="+", type=int, default=(32, 128, 256))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--warmups", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=24)
    args = parser.parse_args()
    if min(args.batch_size, args.warmups, args.repeats) < 1 or min(args.lengths) < 2:
        parser.error("Use positive batch/warm-up/repeat counts and sequence lengths of at least two")

    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    poet = load_poet(args.checkpoint, args.device)
    model = poet.model.float()
    if max(args.lengths) > model.config.context_length:
        parser.error("A sequence length exceeds the model context")
    generator = torch.Generator().manual_seed(args.seed)
    inputs = {
        length: torch.randint(len(SPECIAL), model.config.vocab_size, (args.batch_size, length),
                              generator=generator).to(poet.device)
        for length in sorted(set(args.lengths))
    }
    results = {}
    with torch.inference_mode():
        for length, ids in inputs.items():
            results[str(length)] = {
                "timing": measure_pair(model, ids, {"sdpa": ("sdpa", True), "manual": ("manual", True)},
                                       args.warmups, args.repeats),
                "full_logits_comparison": compare_full_logits(model, ids),
            }
        ids = inputs[max(inputs)]
        causality = causal_check(model, ids)
        projection = measure_pair(model, ids, {"last_only": ("sdpa", True), "full": ("sdpa", False)},
                                  args.warmups, args.repeats)
        full_last = model(ids)[:, -1:]
        projection_error = (model(ids, last_only=True) - full_last).abs().max().item()

    write_json(args.output, {
        "task": "Orchid attention and vocabulary-projection microbenchmarks",
        "dataset": "Controlled synthetic sequences of non-special tokenizer IDs sampled uniformly on CPU.",
        "metrics": ["p50_ms", "p95_ms", "peak_extra_allocated_bytes", "max_abs_logit_difference"],
        "seeds": [args.seed],
        "provenance": {"mode": "measured", "source": "scripts/benchmark_attention.py",
                       "timestamp": datetime.now(timezone.utc).isoformat(),
                       "hardware": torch.cuda.get_device_name(poet.device) if poet.device.type == "cuda" else platform.machine(),
                       "operating_system": platform.system(), "torch": str(torch.__version__),
                       "cuda_build": torch.version.cuda, "checkpoint": str(args.checkpoint),
                       "checkpoint_sha256": poet.checkpoint_sha256},
        "protocol": {"batch_size": args.batch_size, "warmups_per_variant": args.warmups,
                     "timed_forwards_per_variant": args.repeats, "inference_mode": True, "dropout": False,
                     "dtype": str(next(model.parameters()).dtype), "device": str(poet.device), "padding": False,
                     "float32_matmul_precision": torch.get_float32_matmul_precision(),
                     "timing": "Synchronized wall time per warmed forward with alternating variant order.",
                     "memory": "Separate warmed forward, peak allocated bytes minus live allocation before forward."},
        "summary": results,
        "causality": {"sequence_length": ids.size(1), **causality},
        "output_projection": {"sequence_length": ids.size(1), "timing": projection,
                              "max_abs_last_logit_difference": projection_error},
    })


if __name__ == "__main__":
    main()
