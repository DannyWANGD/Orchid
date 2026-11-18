"""Measure raw continuation effects and then compare decoders on frozen test prompts."""

import argparse
from collections import Counter
from pathlib import Path
from statistics import median
import sys

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.evaluation import ratio, score_row, summarize
from orchid.generation import Sampling
from orchid.inference import load_poet
from orchid.io import read_json, read_jsonl, read_yaml, sha256_file, write_json, write_jsonl
from orchid.metrics import PINGSHUI_SHA256, TrainMatches, bundled_rhyme_table
from orchid.runner import verified


def summary(rows):
    out = summarize(rows)
    for name in ("distinct_rhyme", "all_rules", "repeated_line"):
        out[name] = ratio(sum(r[name] for r in rows), len(rows))
    out["median_seconds"] = median(r["seconds"] for r in rows if not r["error"])
    out["rhyme_groups"] = dict(Counter(r["ping_matches"][0] for r in rows if r["distinct_rhyme"]))
    return out


def select(results):
    baseline, control, rhyme = (results[k]["summary"] for k in ("original", "random-sft", "rhyme-sft"))
    checks = {
        "more_distinct_rhymes_than_original_and_control":
            rhyme["distinct_rhyme"]["count"] > max(baseline["distinct_rhyme"]["count"],
                                                    control["distinct_rhyme"]["count"]),
        "form_and_keywords_retained": rhyme["joint_ok"]["count"] >= baseline["joint_ok"]["count"],
        "all_rules_retained": rhyme["all_rules"]["count"] >= baseline["all_rules"]["count"],
        "no_more_exact_train_matches":
            rhyme["train_matches"]["exact"]["count"] <= baseline["train_matches"]["exact"]["count"],
    }
    return {"model": "rhyme-sft" if all(checks.values()) else "original", "validation_checks": checks}


def evaluate(args):
    if (args.output / "metrics.json").exists():
        raise FileExistsError(f"Output already exists: {args.output}")
    protocol_path = Path("artifacts/eval_prompts.json")
    protocol = read_json(protocol_path)
    protocol_hash = sha256_file(protocol_path)
    data = read_yaml(Path("configs/data-tang-song.yaml"))["paths"]
    artifacts, processed = Path(data["artifacts"]), Path(data["processed"])
    tokens = read_json(artifacts / "token_stats.json")
    val = processed / "sft/validation.jsonl"
    validation_hash = verified(val, read_json(artifacts / "sft_stats.json")["splits"]["validation"]["file_sha256"])
    sampling, seeds = Sampling(**protocol["sampling"]), protocol["sampling_seeds"]
    if args.stage == "validation":
        validation = list(read_jsonl(val))
        if len(validation) < 90 or any(r["split"] != "validation" for r in validation):
            raise ValueError("Expected the original frozen validation split")
        prompts = [{"id": r["poem_id"], "keywords": r["keywords"], "oov": [],
                    "train_combo": None, "train_keyword_counts": None}
                   for r in (validation[i * len(validation) // 90] for i in range(90))]
        models = {"original": args.original, "random-sft": args.control, "rhyme-sft": args.rhyme}
        modes = {"raw": {}}
        frozen_hash = validation_hash
        selection_hash = None
    else:
        study = read_json(args.selection)
        if study["stage"] != "validation" or study["protocol_sha256"] != protocol_hash:
            raise ValueError("Selection must come from validation under the same prompt protocol")
        chosen = study["selection"]["model"]
        selected = study["models"][chosen]
        models = {chosen: Path(selected["path"])}
        prompts = protocol["suites"]["main"]
        modes = {
            "raw": {},
            "form": {"constrain_form": True},
            "rhyme-only": {"constrain_rhyme": True},
            "rhymed": {"constrain_rhyme": True, "constrain_keywords": True},
        }
        frozen_hash = protocol_hash
        selection_hash = sha256_file(args.selection)
    train = processed / "train.jsonl"
    train_hash = verified(train, tokens["splits"]["train"]["poem_file_sha256"])
    matcher = TrainMatches(read_jsonl(train))
    rhymes = bundled_rhyme_table()
    results, generated = {}, []
    torch.set_num_threads(2)
    for name, path in models.items():
        poet = load_poet(path, args.device, tokenizer_path=artifacts / "tokenizer.json")
        if (poet.meta["lineage"]["data_manifest_sha256"] != tokens["data_manifest_sha256"]
                or poet.meta["lineage"]["validation_input_sha256"] != validation_hash):
            raise ValueError(f"{name} does not use the expected training corpus and frozen validation split")
        if args.stage == "test" and poet.checkpoint_sha256 != selected["sha256"]:
            raise ValueError("Selected checkpoint has changed since evaluation")
        print(f"Evaluating {name} on {len(prompts)} prompts × {len(seeds)} seeds", flush=True)
        for mode, opts in modes.items():
            stats = []
            for prompt in prompts:
                for seed in seeds:
                    row = score_row(poet, sampling, prompt, seed, args.stage, name, matcher, rhymes, **opts)
                    row["mode"] = mode
                    row["distinct_rhyme"] = row["rhyme"] == "pass" and not row["repeated_rhyme_char"]
                    row["all_rules"] = row["joint_ok"] and row["distinct_rhyme"]
                    lines = row["lines"]
                    row["repeated_line"] = row["format_ok"] and len(set(lines)) < len(lines)
                    stats.append(row)
            generated.extend(stats)
            report = summary(stats)
            key = name if args.stage == "validation" else mode
            results[key] = {"path": str(path), "run_id": poet.meta["run_id"],
                            "sha256": poet.checkpoint_sha256, "summary": report}
            print(f"{key}: form={report['format_ok']['count']} rhyme={report['distinct_rhyme']['count']}"
                  f" all={report['all_rules']['count']} / {len(stats)}", flush=True)
    generation_path = args.output / "generations.jsonl"
    write_jsonl(generation_path, generated)
    write_json(args.output / "metrics.json", {
        "provenance": "Measured on the Orchid checkpoints and frozen prompts",
        "stage": args.stage, "device": str(poet.device), "torch": str(torch.__version__), "precision": "fp32",
        "protocol_sha256": protocol_hash, "input_sha256": frozen_hash, "train_sha256": train_hash,
        "generations_sha256": sha256_file(generation_path), "validation_study_sha256": selection_hash,
        "rhyme_sha256": PINGSHUI_SHA256, "seeds": seeds, "sampling": protocol["sampling"],
        "prompt_count": len(prompts), "models": results,
        "selection": select(results) if args.stage == "validation" else study["selection"],
    })


def main():
    p = argparse.ArgumentParser(description=__doc__)
    action = p.add_subparsers(dest="stage", required=True)
    val, test = action.add_parser("validation"), action.add_parser("test")
    for parser in (val, test):
        parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
        parser.add_argument("--output", required=True, type=Path)
    val.add_argument("--original", required=True, type=Path)
    val.add_argument("--control", required=True, type=Path)
    val.add_argument("--rhyme", required=True, type=Path)
    test.add_argument("--selection", required=True, type=Path)
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
