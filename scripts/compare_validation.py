"""Compare write-to-keywords quality on the supervised validation split only."""

import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import sys

import torch

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.evaluation import reference_nll, score_row, strata
from orchid.generation import Sampling
from orchid.inference import load_poet
from orchid.io import read_json, read_jsonl, read_yaml, sha256_file, write_json, write_jsonl
from orchid.metrics import TrainMatches, rhyme_table
from orchid.runner import verified


def run(config, checkpoints, out, prompt_count, seeds, sampling, device, threads, extended_data=None):
    cfg = read_yaml(config)
    paths = {k: Path(v) for k,v in read_yaml(Path(cfg["data"]))["paths"].items()}
    art, proc = paths["artifacts"], paths["processed"]
    stats = read_json(art/"sft_stats.json")
    val_path = proc/"sft/validation.jsonl"
    val_sha = verified(val_path, stats["splits"]["validation"]["file_sha256"])
    data = list(read_jsonl(val_path))
    if any(row["split"] != "validation" for row in data):
        raise ValueError("Expected validation rows only")
    train_file = proc/"train.jsonl"
    verified(train_file, read_json(art/"token_stats.json")["splits"]["train"]["poem_file_sha256"])
    rule = cfg["rhyme"]
    rh_file = Path(rule["path"])
    verified(rh_file, rule["sha256"])
    rhymes = rhyme_table(rh_file)
    vocab_path = art/"tokenizer.json"
    def sources(artifacts, processed):
        current = read_json(artifacts/"sft_stats.json")
        source = processed/"sft/train.jsonl"
        verified(source, current["splits"]["train"]["file_sha256"])
        train_rows = list(read_jsonl(source))
        return (current["data_manifest_sha256"],
                TrainMatches(read_jsonl(processed/"train.jsonl")),
                Counter(w for row in train_rows for w in row["keywords"]),
                {tuple(sorted(row["keywords"])) for row in train_rows})

    key, *payload = sources(art, proc)
    training_sources = {key: payload}
    if extended_data:
        config_ext = read_yaml(extended_data)["paths"]
        eart, eproc = (Path(config_ext[k]) for k in ("artifacts", "processed"))
        e = read_json(eart/"sft_stats.json")
        if e["frozen_holdout"]["splits"]["validation"] != val_sha or e["tokenizer_sha256"] != stats["tokenizer_sha256"]:
            raise ValueError("Models must share the exact validation rows and tokenizer")
        tt = read_json(eart/"token_stats.json")
        verified(eproc/"train.jsonl",tt["splits"]["train"]["poem_file_sha256"])
        ext_key, *ext_payload = sources(eart, eproc)
        training_sources[ext_key] = ext_payload
    n = min(len(data), prompt_count)
    at = [data[i*len(data)//n] for i in range(n)]
    if any(row["split"] != "validation" for row in at):
        raise ValueError("Study is strictly validation only")
    references = [{"poem_id": r["poem_id"]} for r in data]
    result, generated = {}, []
    torch.set_num_threads(threads)
    for path in checkpoints:
        poet = load_poet(path, device or cfg["runtime"]["device"], tokenizer_path=vocab_path)
        name = poet.meta["run_id"]
        model_source = poet.meta["lineage"]["data_manifest_sha256"]
        if model_source not in training_sources:
            raise ValueError(f"Unregistered training data for {name}")
        matcher, counts, combos = training_sources[model_source]
        prompts = [{"id": r["poem_id"], "poem_id": r["poem_id"], "keywords": r["keywords"],
                    "oov": [], "train_combo": tuple(sorted(r["keywords"])) in combos,
                    "train_keyword_counts": [counts[w] for w in r["keywords"]]} for r in at]
        gen = [score_row(poet, sampling, r, seed, "validation", name, matcher, rhymes)
               for r in prompts for seed in seeds]
        nll = reference_nll(poet, val_path, references)
        result[name] = {"checkpoint": str(path), "sha256": poet.checkpoint_sha256,
                        "seed": poet.meta["seed"], "parameters": poet.model.parameter_count(),
                        "nll": nll, "generation": strata(gen)}
        generated.extend(gen)
        headline = result[name]["generation"]["overall"]
        print(name, f"NLL={nll['nll']:.4f}", f"joint={headline['joint_ok']['count']}/{len(gen)}", flush=True)
        del poet
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_jsonl(out/"generations.jsonl",generated)
    write_json(out/"metrics.json", {
        "provenance":"measured validation only; no test scores used",
        "validation_pairs_sha256": val_sha, "prompts": [r["poem_id"] for r in at],
        "sampling": asdict(sampling), "seeds": seeds, "summary": result,
        "device": device or cfg["runtime"]["device"], "precision": "fp32", "cpu_threads": threads,
        "predictions_sha256":sha256_file(out/"generations.jsonl"),
    })


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path("configs/eval.yaml"))
    p.add_argument("--checkpoints",type=Path,nargs="+",required=True)
    p.add_argument("--prompts",type=int,default=90)
    p.add_argument("--seeds",type=int,nargs="+",default=[11,22,33])
    p.add_argument("--device",choices=("cpu","cuda"))
    p.add_argument("--cpu-threads",type=int,default=2)
    p.add_argument("--extended-data",type=Path)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    if min(a.prompts, a.cpu_threads) < 1:
        p.error("Prompt and thread counts must be positive")
    run(a.config,a.checkpoints,a.output,a.prompts,a.seeds,Sampling(),a.device,a.cpu_threads,a.extended_data)


if __name__=="__main__":
    main()
