"""Measure a full model FP32 step, a 24-poem debug fit, and BF16 throughput."""

import argparse
from dataclasses import replace
from itertools import islice
from math import ceil
from pathlib import Path
import random
import sys
import time

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.data import batch_loader
from orchid.generation import Sampling, sample_tokens
from orchid.io import read_jsonl, read_yaml, write_json
from orchid.model import GPT, cross_entropy_sum
from orchid.runner import code_provenance, environment, load_inputs
from orchid.tokenizer import BOS, EOS
from orchid.training import (
    TrainConfig, batched_steps, build_adamw, evaluate_nll, optimize, runtime_settings,
    schedule, seed_everything, synchronize,
)


def run(config, warmup, timed):
    device, precision = runtime_settings(config["runtime"])
    seed = config["seed"]
    seed_everything(seed, device, config["runtime"])
    tokenizer, _, lineage, spec, datasets = load_inputs("pretrain", config)
    model = GPT(spec).to(device)
    parameters = model.parameter_count()
    if parameters != spec.expected_parameters():
        raise AssertionError("Parameter formula and model disagree")
    x, y = (t.to(device) for t in next(iter(batch_loader(datasets["train"], [0, 1], batch_size=2))))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss = cross_entropy_sum(model(x), y) / y.numel()
    loss.backward()
    synchronize(device)
    if not loss.isfinite().item() or any(p.grad is None or not p.grad.isfinite().all() for p in model.parameters()):
        raise FloatingPointError("FP32 gradient check failed")
    fp = {"dtype": str(next(model.parameters()).dtype), "nll": float(loss.detach().item()),
          "peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
          "batch_shape": list(x.shape)}
    print(f"FP32 forward/backward OK, {parameters:,} parameters", flush=True)
    del model, x, y, loss

    data = read_yaml(Path(config["data"]))
    raw = Path(data["paths"]["processed"]) / "train.jsonl"
    selected = random.Random(seed).sample([r for r in read_jsonl(raw) if r["form"] == "wujue"], 24)
    tokens = torch.tensor([[BOS, *tokenizer.encode(r["text"], strict=True), EOS] for r in selected])
    batch = (tokens[:, :-1], tokens[:, 1:])
    debug = GPT(replace(spec, dropout=0.0)).to(device)
    debug_steps = 100
    debug_cfg = TrainConfig(lr=0.001, weight_decay=0.0, accumulation=1)
    opt = build_adamw(debug, debug_cfg, device)
    lr = schedule(opt, debug_cfg, debug_steps)
    before = evaluate_nll(debug, [batch], device, precision)["nll"]
    for _ in range(debug_steps):
        optimize(debug, [batch], opt, lr, debug_cfg, device, precision)
    after = evaluate_nll(debug, [batch], device, precision)["nll"]
    debug.eval()
    examples = []
    for row, seq in zip(selected, tokens.tolist()):
        prefix = seq[:6]
        completion, stop = sample_tokens(debug, prefix, Sampling(temperature=0, max_new_tokens=20), seed)
        actual = tokenizer.decode(prefix[1:] + completion)
        examples.append({"poem_id": row["poem_id"], "expected": row["text"], "generated": actual, "stop": stop,
                         "exact": row["text"] == actual and stop == "eos"})
    exact = sum(r["exact"] for r in examples)
    if after >= before or exact == 0:
        raise AssertionError("Small training subset did not learn")
    print(f"24-poem debug fit {before:.4f} -> {after:.4f}; exact {exact}/24", flush=True)
    del debug, opt, lr

    model = GPT(spec).to(device)
    cfg = TrainConfig(**config["training"])
    opt = build_adamw(model, cfg, device)
    lr = schedule(opt, cfg, warmup + timed)

    def windows():
        ds = datasets["train"]
        epoch = 0
        while True:
            order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
            yield from batched_steps(batch_loader(ds, order, batch_size=cfg.micro_batch,
                                                   pin_memory=device.type == "cuda"), cfg.accumulation)
            epoch += 1

    it = windows()
    for window in islice(it, warmup):
        optimize(model, window, opt, lr, cfg, device, precision)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"Completed {warmup} warmup steps, measuring {timed} steps", flush=True)
    start, tokens_seen = time.perf_counter(), 0
    for i, window in enumerate(islice(it, timed), 1):
        _, count, _, _ = optimize(model, window, opt, lr, cfg, device, precision)
        tokens_seen += count
        if i % 50 == 0:
            print(f"Measured {i}/{timed} steps", flush=True)
    synchronize(device)
    seconds = time.perf_counter() - start
    per_epoch = ceil(len(datasets["train"]) / (cfg.micro_batch * cfg.accumulation))
    return {
        "parameters": parameters, "model": spec.__dict__, "fp32_check": fp,
        "debug_fit": {"purpose": "debug training subset; not a generalization result",
                      "steps": debug_steps, "precision": precision, "before_nll": before, "after_nll": after,
                      "exact": exact, "total": len(examples), "examples": examples},
        "throughput": {"warmup_steps": warmup, "timed_steps": timed, "device": str(device),
                       "precision": precision, "tokens": tokens_seen, "tokens_per_second": tokens_seen / seconds,
                       "seconds": seconds,
                       "peak_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                       "steps_per_epoch": per_epoch,
                       "projected_training_seconds": seconds / timed * per_epoch * cfg.epochs,
                       "projected_checkpoints_bytes": parameters * 4 * 7},
        "lineage": lineage, "environment": environment(device), "code": code_provenance("pretrain"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pretrain.yaml"))
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--timed", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("artifacts/training_checks.json"))
    args = parser.parse_args()
    if args.warmup < 0 or args.timed <= 0:
        parser.error("Warmup must be nonnegative and timed steps must be positive")
    output = run(read_yaml(args.config), args.warmup, args.timed)
    write_json(args.output, output)
    print(output["throughput"])


if __name__ == "__main__":
    main()
