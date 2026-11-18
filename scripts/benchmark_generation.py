"""Measure the same warmup and generation requests on CPU and CUDA."""

import argparse
from pathlib import Path
import sys

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.evaluation import benchmark
from orchid.generation import Sampling
from orchid.inference import load_poet
from orchid.io import read_json, sha256_file, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--protocol", default=Path("artifacts/eval_prompts.json"), type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--devices", choices=("cpu", "cuda"), nargs="+", default=("cuda", "cpu"))
    p.add_argument("--cpu-threads", type=int, default=2)
    a = p.parse_args()
    spec = read_json(a.protocol)
    settings = Sampling(**spec["sampling"])
    all_results = {}
    torch.set_num_threads(a.cpu_threads)
    for d in a.devices:
        poet = load_poet(a.checkpoint, d)
        all_results[d] = benchmark(poet, spec["suites"]["main"], settings, spec["sampling_seeds"][0])
        print(d, all_results[d], flush=True)
        digest = poet.checkpoint_sha256
        del poet
    write_json(a.output, {"checkpoint": str(a.checkpoint), "checkpoint_sha256": digest,
                         "prompt_sha256": sha256_file(a.protocol), "sampling": spec["sampling"],
                         "sample_seed": spec["sampling_seeds"][0], "cpu_threads": a.cpu_threads,
                         "benchmarks": all_results})


if __name__ == "__main__":
    main()
