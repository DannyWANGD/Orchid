"""Profile short real BF16 train steps for the specified model and batch sizes."""

import argparse
from dataclasses import replace
import gc
from pathlib import Path
import sys
from time import perf_counter

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.data import batch_loader
from orchid.io import read_yaml, write_json
from orchid.model import GPT
from orchid.runner import environment, load_inputs
from orchid.training import TrainConfig, build_adamw, optimize, runtime_settings, schedule, seed_everything, synchronize


def trial(spec, dataset, cfg, batch, device, prec, warmup, repeats):
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    training = replace(cfg, micro_batch=batch, accumulation=1)
    loader = batch_loader(dataset, list(range(batch)), batch_size=batch, pin_memory=True)
    micro = next(iter(loader))
    torch.cuda.reset_peak_memory_stats(device)
    model = GPT(spec).to(device)
    opt = build_adamw(model, training, device)
    scheduler = schedule(opt, training, warmup + repeats)
    for _ in range(warmup):
        optimize(model, [micro], opt, scheduler, training, device, prec)
    synchronize(device)
    start, count = perf_counter(), 0
    for _ in range(repeats):
        count += optimize(model, [micro], opt, scheduler, training, device, prec)[1]
    synchronize(device)
    dt = perf_counter() - start
    return {"micro_batch": batch, "warmup_steps": warmup, "timed_steps": repeats,
            "target_tokens": count, "seconds": dt, "tokens_per_second": count / dt,
            "mean_step_seconds": dt / repeats, "parameters": model.parameter_count(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "free_before_bytes": free, "total_bytes": total}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--configs", nargs="+", type=Path, required=True)
    p.add_argument("--batches", nargs="+", type=int, default=(32, 64))
    p.add_argument("--warmup", type=int, default=6)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if min(a.repeats, *a.batches) < 1 or a.warmup < 1:
        p.error("Positive batch sizes and step counts required.")
    results = []
    for config_path in a.configs:
        config = read_yaml(config_path)
        device, precision = runtime_settings(config["runtime"])
        if device.type != "cuda":
            p.error("This sweep is for CUDA.")
        _, _, lineage, model, ds = load_inputs("pretrain", config)
        seed_everything(config["seed"], device, config["runtime"])
        for b in a.batches:
            try:
                r = trial(model, ds["train"], TrainConfig(**config["training"]), b, device, precision, a.warmup, a.repeats)
            except torch.cuda.OutOfMemoryError:
                r = {"micro_batch": b, "parameters": model.expected_parameters(), "status": "cuda_oom"}
            r.update(config=str(config_path), precision=precision, lineage=lineage)
            results.append(r)
            gc.collect()
            torch.cuda.empty_cache()
            print(config_path.name, b, round(r.get("tokens_per_second", 0)), r.get("peak_reserved_bytes"), flush=True)
            write_json(a.output, {"environment": environment(device), "runs": results})


if __name__ == "__main__":
    main()
