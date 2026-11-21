"""Measure candidate ranking and candidate count on fixed, paired prompts."""

import argparse
from collections import Counter
from math import ceil
from pathlib import Path
from statistics import median
import sys

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.evaluation import ratio, sample_mean_delta
from orchid.generation import Sampling
from orchid.inference import load_poet
from orchid.io import ROOT, read_json, read_jsonl, sha256_file, write_json, write_jsonl
from orchid.metrics import PINGSHUI_SHA256, bundled_rhyme_table, judge_format_keywords, rhyme_status
from orchid.prosody_decode import SEPARATED_REPEAT_WEIGHT, separated_repeats


def validation_prompts(path):
    prompts = {}
    for row in read_jsonl(path):
        if row["model"] == "rhyme-sft" and row["suite"] == "validation":
            prompts[row["prompt_id"]] = {"id": row["prompt_id"], "keywords": row["keywords"]}
    if len(prompts) != 90:
        raise ValueError("Expected the 90 validation prompts used in the continuation study")
    return list(prompts.values())


def result_row(poet, prompt, spec, seed, options, rhyme_table):
    keywords = prompt["keywords"]
    result = poet.tokens(keywords, spec, seed, **options)
    quality = judge_format_keywords(result["generated"], keywords)
    rhyme = rhyme_status(quality["lines"], rhyme_table)
    repeats = separated_repeats(quality["lines"])
    distinct_rhyme = rhyme["rhyme"] == "pass" and not rhyme["repeated_rhyme_char"]
    return {
        "prompt_id": prompt["id"], "sampling_seed": seed, "keywords": keywords,
        "generated": result["generated"], "seconds": result["seconds"],
        "form_ok": quality["format_ok"], "keywords_ok": quality["keywords_ok"],
        "distinct_rhyme": distinct_rhyme, "all_rules": quality["joint_ok"] and distinct_rhyme,
        "separated_repeats": repeats, "has_separated_repeat": bool(repeats),
    }


def summarize(rows):
    times = sorted(row["seconds"] for row in rows)
    metrics = ("form_ok", "keywords_ok", "distinct_rhyme", "all_rules", "has_separated_repeat")
    return {
        "requests": len(rows), "counts": {key: ratio(sum(row[key] for row in rows), len(rows)) for key in metrics},
        "repeat_instances": sum(row["separated_repeats"] for row in rows),
        "median_seconds": median(times), "p95_seconds": times[ceil(.95 * len(times)) - 1],
        "by_keyword_count": {
            str(n): {key: ratio(sum(row[key] for row in subset), len(subset)) for key in metrics}
            for n in (1, 2, 3) if (subset := [row for row in rows if len(row["keywords"]) == n])
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "challenge"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output}")
    protocol_path = ROOT / "artifacts/eval_prompts.json"
    protocol = read_json(protocol_path)
    constrained = {"constrain_rhyme": True, "constrain_keywords": True}
    modes = {"original": constrained, "repeat-aware": {**constrained, "ranking": "repeat-aware"}}
    if args.split == "validation":
        source = ROOT / "report/evidence/prosody-study/validation/generations.jsonl"
        prompts = validation_prompts(source)
        seeds = protocol["sampling_seeds"][:1]
    else:
        source = ROOT / "data/prompts/ranking_challenge.json"
        prompts, seeds = read_json(source)["prompts"], protocol["sampling_seeds"]
        prompts_words = {tuple(sorted(p["keywords"])) for p in prompts}
        old_prompts = {tuple(sorted(p["keywords"])) for p in protocol["suites"]["main"]}
        if (len(prompts) != len(prompts_words) or len({p["id"] for p in prompts}) != len(prompts)
                or prompts_words & old_prompts or Counter(len(p["keywords"]) for p in prompts) != {1: 12, 2: 12, 3: 12}):
            raise ValueError("Challenge prompts must be disjoint from the main test and balanced by keyword count")
        modes = {"raw": {}, "one-candidate": {**constrained, "candidates": 1}, **modes}
    torch.set_num_threads(2)
    poet = load_poet(args.checkpoint, args.device)
    spec = Sampling(**protocol["sampling"])
    table = bundled_rhyme_table()
    for prompt in prompts:
        poet.tokenizer.encode("".join(poet.task.validate_keywords(prompt["keywords"])), strict=True)
    results = {mode: [] for mode in modes}
    for options in modes.values():
        poet.tokens(prompts[0]["keywords"], spec, seeds[0], **options)
    for prompt in prompts:
        for seed in seeds:
            for mode, options in modes.items():
                results[mode].append(result_row(poet, prompt, spec, seed, options, table))
    if any(not row["all_rules"] for mode, rows in results.items() if mode != "raw" for row in rows):
        raise RuntimeError("A constrained mode failed a generation rule")
    args.output.mkdir(parents=True)
    records = args.output / "generations.jsonl"
    write_jsonl(records, ({**row, "mode": mode} for mode, rows in results.items() for row in rows))
    changes = sum(a["generated"] != b["generated"]
                  for a, b in zip(results["original"], results["repeat-aware"], strict=True))
    write_json(args.output / "metrics.json", {
        "provenance": "Measured with matched prompts, seeds, checkpoint, and candidate pools for the two rankings.",
        "split": args.split, "prompt_count": len(prompts), "seeds": seeds, "device": args.device,
        "torch": str(torch.__version__), "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "weights_sha256": poet.checkpoint_sha256, "prompt_source_sha256": sha256_file(source),
        "sampling": protocol["sampling"], "rhyme_table_sha256": PINGSHUI_SHA256,
        "separated_repeat_weight": SEPARATED_REPEAT_WEIGHT, "generations_sha256": sha256_file(records),
        "baseline_to_repeat_aware": {
            "changed_outputs": changes,
            "repeat_rate_paired_delta": sample_mean_delta(
                results["original"], results["repeat-aware"], "has_separated_repeat"),
        },
        "modes": {mode: summarize(rows) for mode, rows in results.items()},
    })
    print(f"{args.split}: {len(prompts)} prompts × {len(seeds)} seeds, {changes} outputs changed by reranking")
    for mode, rows in results.items():
        print(mode, len(rows), sum(row["has_separated_repeat"] for row in rows))


if __name__ == "__main__":
    main()
