"""Evaluate unconstrained poetry generation and paired controls on frozen prompts."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import math
from pathlib import Path
import random
from statistics import mean, median, stdev

import torch
from torch.utils.data import Dataset

from .corpus import fetch
from .data import response_item
from .generation import Sampling
from .inference import load_poet
from .io import read_json, read_jsonl, read_yaml, sha256_file, write_bytes, write_json, write_jsonl
from .metrics import TrainMatches, judge_format_keywords, rhyme_status, rhyme_table
from .protocol import freeze
from .runner import verified
from .supervision import Task
from .training import evaluate_nll


class References(Dataset):
    def __init__(self, file, fixed_ids, tokenizer):
        rows = {r["poem_id"]: r for r in read_jsonl(file)}
        self.items = [response_item(tokenizer, rows[i]["keywords"], rows[i]["response"]) for i in fixed_ids]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def reference_nll(poet, ref_path, prompts):
    from .data import collate_responses

    ds = References(ref_path, [p["poem_id"] for p in prompts], poet.tokenizer)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_responses,
                                         num_workers=0, pin_memory=poet.device.type == "cuda")
    return evaluate_nll(poet.model, loader, poet.device)


def ratio(count, total):
    return {"count": count, "total": total, "rate": count / total if total else None}


def score_row(poet, spec, row, sampling_seed, suite, model_name, train_matches, rhymes,
              *, replacement=None, constrain_form=False, constrain_keywords=False):
    words = row["keywords"]
    try:
        out = poet.tokens(words if replacement is None else replacement, spec, sampling_seed,
                          constrain_form=constrain_form, constrain_keywords=constrain_keywords)
        out["error"] = None
    except ValueError as exc:
        out = {"generated": "", "generated_ids": [], "stop": "rejected",
               "seconds": 0.0, "generated_steps": 0, "error": str(exc)}
    quality = judge_format_keywords(out["generated"], words)
    rh = rhyme_status(quality["lines"], rhymes)
    match = train_matches.find(quality["lines"])
    return {
        "model": model_name, "suite": suite, "prompt_id": row["id"], "keywords": words,
        "input_keywords": words if replacement is None else replacement,
        "case": row.get("case"), "poem_id": row.get("poem_id"),
        "train_combo": row["train_combo"], "train_keyword_counts": row["train_keyword_counts"],
        "oov": row["oov"], "sampling_seed": sampling_seed,
        "constrained_form": constrain_form or constrain_keywords, "constrained_keywords": constrain_keywords,
        **out, **quality, **rh, "train_matches": match,
    }


def summarize(rows):
    total = len(rows)
    form = [r for r in rows if r["format_ok"]]
    rhymes = Counter(r["rhyme"] for r in form)
    decided = rhymes["pass"] + rhymes["fail"]
    bodies = defaultdict(set)
    body_all = ["".join(r["lines"]) for r in form]
    for body, row in zip(body_all, form):
        bodies[body].add(row["prompt_id"])
    k = sum(len(r["keywords"]) for r in rows)
    matches = {name: ratio(sum(bool(r["train_matches"].get(name)) for r in form), len(form))
               for name in ("exact", "longer_poem_fragments", "near")}
    return {
        "requests": total,
        **{name: ratio(sum(r[name] for r in rows), total)
           for name in ("parsed", "format_ok", "keywords_ok", "joint_ok", "canonical_punctuation")},
        "keyword_coverage_micro": ratio(sum(len(r["keyword_hits"]) for r in rows), k),
        "stop_reasons": dict(Counter(r["stop"] for r in rows)),
        "rejected_by_case": dict(Counter(r["case"] or "other" for r in rows if r["error"])),
        "same_poem_across_prompts": sum(len(ids) > 1 for ids in bodies.values()),
        "repeat_occurrence": ratio(len(body_all) - len(bodies), len(form)),
        "train_matches": matches,
        "rhyme": {"pass": rhymes["pass"], "fail": rhymes["fail"], "unknown": rhymes["unknown"],
                  "repeated_char": sum(r.get("repeated_rhyme_char", False) for r in form),
                  "unknown_causes": dict(Counter(r["rhyme_cause"] for r in form if r["rhyme"]=="unknown")),
                  "pass_among_decided": ratio(rhymes["pass"], decided),
                  "decided_among_wellformed": ratio(decided, len(form))},
    }


def strata(rows):
    return {
        "overall": summarize(rows),
        "by_keyword_count": {str(n): summarize([r for r in rows if len(r["keywords"]) == n])
                             for n in sorted({len(r["keywords"]) for r in rows})},
        "seen_in_train": summarize([r for r in rows if r["train_combo"]]),
        "new_combination_all_words_seen": summarize(
            [r for r in rows if not r["train_combo"] and all(r["train_keyword_counts"])]),
        "new_word_in_prompt": summarize([r for r in rows if not all(r["train_keyword_counts"])]),
        "oov": summarize([r for r in rows if r["oov"]]),
    }


def sample_mean_delta(first, second, key, *, draws=1000):
    first = {(r["prompt_id"],r["sampling_seed"]): r[key] for r in first}
    groups = defaultdict(list)
    for r in second:
        pair = r["prompt_id"], r["sampling_seed"]
        if pair in first:
            groups[pair[0]].append(int(r[key]) - int(first[pair]))
    if not groups:
        return None
    points = [mean(v) for k, v in sorted(groups.items())]
    rng = random.Random(20261118)
    boot = sorted(mean(rng.choices(points, k=len(points))) for _ in range(draws))
    return {"difference": mean(points), "prompts": len(points), "paired_draws": sum(map(len, groups.values())),
            "percentile95": [boot[math.floor(.025*(draws-1))], boot[math.ceil(.975*(draws-1))]]}


def benchmark(poet, prompts, sampling, seed):
    slots = [p for p in prompts if not p["oov"]][:12]
    if len(slots) < 3:
        raise ValueError("Benchmark requires at least three valid prompts")
    for p in slots[:2]:
        poet.tokens(p["keywords"], sampling, seed)
    if poet.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(poet.device)
    items = [poet.tokens(p["keywords"], sampling, seed) for p in slots[2:]]
    memory = torch.cuda.max_memory_allocated(poet.device) if poet.device.type == "cuda" else None
    return {
        "device": str(poet.device), "precision": "fp32", "warmup_requests": 2,
        "timed_requests": len(items), "load_seconds": poet.load_seconds,
        "median_seconds": median(r["seconds"] for r in items),
        "tokens_per_second": sum(r["generated_steps"] for r in items)/sum(r["seconds"] for r in items),
        "target_steps": sum(r["generated_steps"] for r in items), "peak_allocated_bytes": memory,
    }


def event_file(path, rows):
    write_jsonl(path, rows)
    return {"file": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def run_eval(cfg, output, *, device_override=None, smoke=False, variants_only=False):
    task = Task(**read_yaml(Path(cfg["task"])))
    paths = {k: Path(v) for k,v in read_yaml(Path(cfg["data"]))["paths"].items()}
    art, proc = paths["artifacts"], paths["processed"]
    tok_file, frozen = art / "tokenizer.json", Path(cfg["prompt_config"]["materialized"])
    ps = freeze(cfg, paths, task)
    if output.exists() and (output/"metrics.json").exists():
        raise FileExistsError(f"Output already finished: {output}. Choose a new directory.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"resolved_config.json", {**cfg, "resolved_device": device_override or cfg["runtime"]["device"],
                                                "smoke": smoke, "variants_only": variants_only})
    data_stat = read_json(art / "token_stats.json")
    train_path, test_path = proc / "train.jsonl", proc / "sft" / "test.jsonl"
    verified(train_path, data_stat["splits"]["train"]["poem_file_sha256"])
    # Every SFT response is the normalized body of a train poem; verify rather than assume.
    tr_sft = proc / "sft" / "train.jsonl"
    stats = read_json(art / "sft_stats.json")
    verified(tr_sft, stats["splits"]["train"]["file_sha256"])
    train_rows = list(read_jsonl(train_path))
    by_id = {r["poem_id"]: r for r in train_rows}
    for r in read_jsonl(tr_sft):
        norm = judge_format_keywords(r["response"], r["keywords"])
        if not norm["joint_ok"] or tuple(by_id[r["poem_id"]]["lines"]) != norm["lines"]:
            raise ValueError("Supervised response isn't covered by the train-memory index")
    matches = TrainMatches(train_rows)
    alternate = cfg.get("expanded_models")
    if alternate:
        exp = {k: Path(v) for k,v in read_yaml(Path(alternate["data"]))["paths"].items()}
        exp_stat = read_json(exp["artifacts"]/"token_stats.json")
        if (exp_stat["tokenizer_sha256"] != data_stat["tokenizer_sha256"]
                or any(exp_stat["splits"][s] != data_stat["splits"][s] for s in ("validation", "test"))):
            raise ValueError("Expanded training supports the same frozen evaluation data only")
        exp_stat_sft = read_json(exp["artifacts"]/"sft_stats.json")
        if exp_stat_sft["splits"]["test"]["file_sha256"] != stats["splits"]["test"]["file_sha256"]:
            raise ValueError("Expanded test references do not match the fixed Tang test")
        exp_train = exp["processed"]/"train.jsonl"
        verified(exp_train, exp_stat["splits"]["train"]["poem_file_sha256"])
        by_exp = {r["poem_id"]: r for r in read_jsonl(exp_train)}
        sft_path = exp["processed"]/"sft/train.jsonl"
        verified(sft_path, exp_stat_sft["splits"]["train"]["file_sha256"])
        exp_pairs = list(read_jsonl(sft_path))
        if any(by_exp[r["poem_id"]]["text"] != r["response"] for r in exp_pairs):
            raise ValueError("Expanded SFT contains a response outside the expanded train index")
        exp_combos = {tuple(sorted(r["keywords"])) for r in exp_pairs}
        exp_word_counts = Counter(w for r in exp_pairs for w in r["keywords"])
        exp_match = TrainMatches(by_exp.values())
        del by_exp, exp_pairs
    rule = cfg["rhyme"]
    rh_file = Path(rule["path"])
    if not rh_file.exists():
        candidate = fetch(rule["url"])
        import hashlib
        if hashlib.sha256(candidate).hexdigest() != rule["sha256"]:
            raise ValueError("Rhyme source hash mismatch")
        write_bytes(rh_file, candidate)
    verified(rh_file, rule["sha256"])
    rhymes = rhyme_table(rh_file)
    sampling = Sampling(**cfg["sampling"])
    suites = {k: v[:3] if smoke else v for k,v in ps["suites"].items()}
    seeds = cfg["seeds"][:1] if smoke else cfg["seeds"]
    torch.set_num_threads(cfg["runtime"]["cpu_threads"])
    device = device_override or cfg["runtime"]["device"]
    selected = cfg["selected"]
    models = cfg["models"]
    if selected not in models:
        raise ValueError("Selected SFT model not present")
    if variants_only:
        models = {k:v for k,v in models.items() if k.startswith("X-")}
    if smoke:
        models = {k:v for k,v in models.items() if k in ("R1",selected)}
    cases = {}
    reference, benchmarks, provenance = {}, {}, {}
    by_main_id = {r["id"]: r for r in ps["suites"]["main"]}
    full_main = suites["main"]
    for name, cp in models.items():
        poet = load_poet(cp, device, tokenizer_path=tok_file)
        source_stat, match_index = ((exp_stat, exp_match) if alternate and name in alternate["names"]
                                    else (data_stat, matches))
        if (poet.meta["lineage"]["data_manifest_sha256"] != source_stat["data_manifest_sha256"]
                or poet.meta["lineage"]["split_manifest_sha256"] != source_stat["split_manifest_sha256"]):
            raise ValueError(f"Wrong dataset lineage in {cp}")
        if poet.meta["resolved"]["task_config"] != {
            "form": task.form, "lines": task.lines, "chars_per_line": task.chars_per_line,
            "keywords": task.keywords, "script": task.script,
        }:
            raise ValueError(f"Different evaluation task in {cp}")
        provenance[name] = {"checkpoint": str(cp), "sha256": poet.checkpoint_sha256,
                            "run_id": poet.meta["run_id"], "training_seed": poet.meta["seed"],
                            "model_config": poet.meta["model"], "init": poet.meta["initialization"],
                            "training_lineage": poet.meta["lineage"]}
        reference[name] = reference_nll(poet, test_path, suites["reference"])
        job = {"main": full_main[:8]} if name == "R0" else suites

        def tags(row):
            if source_stat is data_stat:
                return row
            words = row["keywords"]
            return {**row, "train_combo": tuple(sorted(words)) in exp_combos,
                    "train_keyword_counts": [exp_word_counts[w] for w in words]}

        cases[name] = [score_row(poet, sampling, tags(row), seed, suite, name, match_index, rhymes)
                       for suite, inputs in job.items() for row in inputs for seed in seeds]
        if name == selected and not variants_only:
            cases["R2-K"] = [
                score_row(poet, sampling, tags(row), seed,suite="main", model_name="R2-K", train_matches=match_index,
                          rhymes=rhymes, replacement=by_main_id[ps["swaps"][row["id"]]]["keywords"])
                for row in full_main if row["id"] in ps["swaps"] for seed in seeds
            ]
            cases["R3"] = [score_row(poet, sampling, tags(row), seed,"main","R3",match_index,rhymes,constrain_form=True)
                           for row in full_main for seed in seeds]
            provenance["R2-K"] = {**provenance[name], "control": "swap keywords; score against original"}
            provenance["R3"] = {**provenance[name], "control": "force punctuation, line lengths and EOS"}
            if cfg.get("keyword_constraint_control", False):
                cases["R3-K"] = [score_row(poet, sampling, tags(row), seed, "main", "R3-K", match_index, rhymes,
                                           constrain_keywords=True) for row in full_main for seed in seeds]
                provenance["R3-K"] = {**provenance[name], "control": "force form and one word in each of separate lines"}
            if not smoke:
                benchmarks["selected"] = benchmark(poet, ps["suites"]["main"], sampling, seeds[0])
        print(f"{name}: {len(cases[name])} requests, reference NLL {reference[name]['nll']:.5f}", flush=True)
        del poet
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    if not smoke and not variants_only and str(device) != "cpu":
        cpu_model = load_poet(cfg["models"][selected], "cpu", tokenizer_path=tok_file)
        benchmarks["selected_cpu"] = benchmark(cpu_model, ps["suites"]["main"], sampling, seeds[0])
    for r in cases.get("R2-K", []):
        control = judge_format_keywords(r["generated"], r["input_keywords"])
        r["replacement_hit_count"] = len(control["keyword_hits"])
        r["replacement_all_hit"] = control["keywords_ok"]
    metrics = {n: {s: strata([r for r in rows if r["suite"] == s]) for s in sorted({r["suite"] for r in rows})}
               for n, rows in cases.items()}
    multi = {}
    for s in suites:
        names = [n for n,values in metrics.items() if n.startswith("R2-20") and s in values]
        if not names:
            continue
        multi[s] = {}
        for k in ("format_ok","keywords_ok","joint_ok"):
            values = [metrics[n][s]["overall"][k] for n in names]
            vals = [r["rate"] for r in values]
            multi[s][k] = {"mean": mean(vals), "std": stdev(vals) if len(vals)>1 else None,
                           "n_seeds": len(vals), "by_run": dict(zip(names,values))}
    contrasts = {}
    if selected in cases and "R1" in cases:
        comparisons = [("R1",selected), ("R2-K",selected), (selected,"R3")]
        if "R3-K" in cases:
            comparisons.append((selected,"R3-K"))
        for a, b in comparisons:
            contrasts[f"{b} - {a}"] = {key: sample_mean_delta(
                [r for r in cases[a] if r["suite"] == "main"], [r for r in cases[b] if r["suite"] == "main"], key)
                for key in ("format_ok","keywords_ok","joint_ok")}
    generations = event_file(output/"generations.jsonl", (r for items in cases.values() for r in items))
    result = {
        "schema": "orchid-eval-v1", "task": "keywords -> four five-Han-character lines",
        "dataset": "pinned chinese-poetry Tang holdouts; training lineage recorded per model",
        "metrics": ["word exact hit", "four-by-five", "joint", "train matches", "even-line rhyme", "reference NLL"],
        "selected": selected, "seeds": seeds, "sampling": cfg["sampling"], "smoke": smoke,
        "provenance": {"mode":"measured", "timestamp": datetime.now(timezone.utc).isoformat(),
                       "device": str(device), "dataset_contract":"docs/data_contract.md",
                       "data_manifest_sha256": data_stat["data_manifest_sha256"],
                       "tokenizer_sha256": data_stat["tokenizer_sha256"],
                       "split_manifest_sha256": data_stat["split_manifest_sha256"],
                       "prompt_version": ps["version"], "prompt_sha256": sha256_file(frozen),
                       "rhyme_sha256": rule["sha256"], "models": provenance},
        "summary": metrics, "r2_across_seeds": multi, "reference_nll": reference,
        "swap_input_hit_rate": ratio(sum(r["replacement_all_hit"] for r in cases.get("R2-K", [])),
                                     len(cases.get("R2-K", []))),
        "paired_prompt_bootstrap": contrasts, "benchmark": benchmarks, "generations": generations,
    }
    write_json(output/"metrics.json", result)
    return result
