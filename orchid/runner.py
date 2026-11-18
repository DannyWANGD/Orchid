"""Run one versioned experiment, with a resumable cursor and audited inputs."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from math import ceil
from pathlib import Path
import platform
import re
import subprocess
import time
import uuid

import numpy as np
import torch

from .data import Responses, TokenWindows, batch_loader
from .generation import Sampling, sample_tokens
from .io import ROOT, json_text, read_json, read_jsonl, read_yaml, sha256_file, sha256_text, write_json, write_jsonl
from .model import GPT, GPTConfig
from .tokenizer import BOS, CharacterTokenizer
from .training import (
    TrainConfig, batched_steps, build_adamw, evaluate_nll, load_checkpoint, optimize,
    restore_rng, rng_state, runtime_settings, save_checkpoint, schedule, seed_everything, synchronize,
)


def verified(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"Hash mismatch: {path}. Rebuild from the pinned data and tokenizer.")
    return actual


def load_inputs(stage, cfg):
    data = read_yaml(Path(cfg["data"]))
    paths = {k: Path(v) for k, v in data["paths"].items()}
    a, p = paths["artifacts"], paths["processed"]
    ts = read_json(a / "token_stats.json")
    tokenizer = CharacterTokenizer.load(a / "tokenizer.json")
    resolved_model = GPTConfig(vocab_size=len(tokenizer), **read_yaml(Path(cfg["model"])))
    context = resolved_model.context_length
    lineage = {
        "tokenizer_sha256": verified(a / "tokenizer.json", ts["tokenizer_sha256"]),
        "data_manifest_sha256": verified(a / "data_manifest.json", ts["data_manifest_sha256"]),
        "split_manifest_sha256": verified(a / "split_manifest.json", ts["split_manifest_sha256"]),
    }
    sets = {}
    subset = cfg.get("training_subset")
    if subset is not None and stage != "sft":
        raise ValueError("Training subsets are supported only for SFT")
    if stage == "sft":
        ss = read_json(a / "sft_stats.json")
        if any(ss[k] != lineage[k] for k in lineage):
            raise ValueError("Supervision is bound to a different corpus or vocabulary")
        lineage["sft_stats_sha256"] = sha256_file(a / "sft_stats.json")
    for s in ("train", "validation"):
        if stage == "pretrain":
            file = p / "tokens" / f"{s}.npy"
            lineage[s + "_input_sha256"] = verified(file, ts["splits"][s]["token_file_sha256"])
            sets[s] = TokenWindows.load(file, context)
        else:
            file = p / "sft" / f"{s}.jsonl"
            lineage[s + "_input_sha256"] = verified(file, ss["splits"][s]["file_sha256"])
            indices = None
            if s == "train" and subset is not None:
                from .sft_subset import training_subset
                indices, lineage["training_subset"] = training_subset(file, seed=cfg["seed"], **subset)
            sets[s] = Responses(file, tokenizer, context, s, indices=indices)
            if indices is not None and len(sets[s]) != len(indices):
                raise ValueError("Every selected training row must fit in the context")
    return tokenizer, a, lineage, resolved_model, sets


def code_provenance(stage):
    sources = ("io", "data", "model", "runner", "training", "generation", "tokenizer")
    hashes = {f"orchid/{name}.py": sha256_file(ROOT / "orchid" / f"{name}.py") for name in sources}
    if stage == "sft":
        for name in ("sft_subset", "metrics"):
            hashes[f"orchid/{name}.py"] = sha256_file(ROOT / "orchid" / f"{name}.py")
    cli = "pretrain" if stage == "pretrain" else "finetune"
    entry = ROOT / "scripts" / f"{cli}.py"
    if entry.exists():
        hashes[f"scripts/{cli}.py"] = sha256_file(entry)
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        head = None
    return {"git_head": head, "files": hashes, "source_sha256": sha256_text(json_text(hashes, None))}


def environment(device):
    return {
        "python": platform.python_version(), "numpy": np.__version__,
        "torch": str(torch.__version__), "cuda_build": torch.version.cuda,
        "device": str(device),
        "accelerator": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "fp32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def append(path, row):
    with path.open("a", encoding="utf-8") as dest:
        dest.write(json_text(row, None) + "\n")


def artifact_path(root, name):
    return root / (name + ".pt")


def validation_samples(stage, model, token, sample_cfg, sample_path, step, nll, val_path=None):
    if stage == "pretrain":
        opts = {k: v for k, v in sample_cfg.items() if k not in ("prefixes", "seed")}
        spec = Sampling(**opts)
        prompts = [(prefix, [BOS, *token.encode(prefix, strict=True)])
                   for prefix in sample_cfg.get("prefixes", [])]
    else:
        from .data import prompt_ids

        spec = Sampling(**{k: v for k, v in sample_cfg.items() if k not in ("seed", "validation_count")})
        rows = list(read_jsonl(val_path))
        # Fixed equally spaced positions in the validation file; no test inputs.
        count = min(len(rows), int(sample_cfg.get("validation_count", 12)))
        prompts = [(r["keywords"], prompt_ids(token, r["keywords"], strict=False))
                   for r in [rows[j * len(rows) // count] for j in range(count)]]
    was_training = model.training
    model.eval()
    out = []
    try:
        for label, ids in prompts:
            seed = int(sample_cfg.get("seed", 11))
            generated, stop = sample_tokens(model, ids, spec, seed)
            record = {"step": step, "validation_nll": nll, "input": label, "generated": token.decode(generated),
                      "generated_ids": generated, "stop": stop, "seed": seed, "sampling": asdict(spec)}
            if stage == "sft":
                from .metrics import judge_format_keywords
                record.update(judge_format_keywords(record["generated"], label))
            append(sample_path, record)
            out.append(record)
    finally:
        model.train(was_training)
    if stage == "sft":
        return {"n": len(out), "format_ok": sum(r["format_ok"] for r in out),
                "keywords_ok": sum(r["keywords_ok"] for r in out),
                "joint_ok": sum(r["joint_ok"] for r in out)}
    return None


def train(stage, args):
    cfg = read_yaml(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.device:
        cfg["runtime"]["device"] = args.device
    if args.precision:
        cfg["runtime"]["precision"] = args.precision
    if args.micro_batch is not None:
        cfg["training"]["micro_batch"] = args.micro_batch
    resume = load_checkpoint(args.resume) if args.resume else None
    init_sft = getattr(args, "init_sft", None)
    init_path = args.init or init_sft
    if resume and init_path:
        raise ValueError("Specify either --resume (full state) or a weight initialization")
    device, prec = runtime_settings(cfg["runtime"])
    seed_everything(cfg["seed"], device, cfg["runtime"])
    tok, art, lineage, model_cfg, sets = load_inputs(stage, cfg)
    opt_cfg = TrainConfig(**cfg["training"])
    steps_per_epoch = ceil(len(sets["train"]) / (opt_cfg.micro_batch * opt_cfg.accumulation))
    total_steps = steps_per_epoch * opt_cfg.epochs
    if args.stop_after is not None and not 0 < args.stop_after <= total_steps:
        raise ValueError(f"--stop-after must be between 1 and {total_steps}")
    task = read_yaml(Path(cfg["task"]))
    initialization = {"kind": "random_init", "seed": cfg["seed"]}
    if resume:
        initialization = resume["meta"]["initialization"]
    elif stage == "sft":
        if not init_path:
            raise ValueError("SFT requires --init with a best pretraining checkpoint or --init-sft")
        src = load_checkpoint(init_path)
        source_meta = src["meta"]
        required_stage = "sft" if init_sft else "pretrain"
        required_lineage = ["tokenizer_sha256", "data_manifest_sha256", "split_manifest_sha256"]
        if init_sft:
            required_lineage.extend(("sft_stats_sha256", "train_input_sha256", "validation_input_sha256"))
        if (source_meta["stage"] != required_stage or source_meta["model"] != asdict(model_cfg)
                or src["kind"] != "best"
                or (not init_sft and source_meta["initialization"]["kind"] != "random_init")
                or (init_sft and source_meta["resolved"]["task_config"] != task)
                or any(source_meta["lineage"].get(k) != lineage[k] for k in required_lineage)):
            raise ValueError(f"SFT requires this project's matching best {required_stage} checkpoint")
        initialization = {"kind": f"{required_stage}_best", "path": str(init_path),
                          "sha256": sha256_file(init_path), "source_run_id": source_meta["run_id"],
                          "source_step": src["cursor"]["step"]}
    config = json.loads(json_text({
        **cfg, "model_config": asdict(model_cfg), "task_config": task,
        "data_config": read_yaml(Path(cfg["data"])), "training": asdict(opt_cfg),
        "steps_per_epoch": steps_per_epoch, "planned_steps": total_steps,
        "resolved_device": str(device), "resolved_precision": prec,
    }))
    code = code_provenance(stage)
    if resume:
        meta = resume["meta"]
        if (resume["kind"] != "last" or meta["stage"] != stage or meta["resolved"] != config
                or meta["lineage"] != lineage or code["source_sha256"] != meta["code"]["source_sha256"]):
            raise ValueError("Resume requires last.pt with the same resolved config, source code and data")
        if args.run_id and args.run_id != meta["run_id"]:
            raise ValueError("Resuming cannot change the run ID")
        run_id = meta["run_id"]
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_id = args.run_id or f"{stage}-{stamp}-{uuid.uuid4().hex[:6]}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
            raise ValueError("Run ID may use letters, numbers, dot, underscore and hyphen")
        meta = {
            "stage": stage, "run_id": run_id, "seed": cfg["seed"], "model": asdict(model_cfg),
            "config_sha256": sha256_text(json_text(config, None)), "lineage": lineage,
            "resolved": config, "environment": environment(device), "code": code,
            "initialization": initialization, "tokenizer_path": str(art / "tokenizer.json"),
        }
    run = args.runs / run_id
    cp = args.checkpoints / stage / run_id
    metrics_path, sample_path = run / "metrics.jsonl", run / "samples.jsonl"
    model = GPT(model_cfg).to(device)
    optimizer = build_adamw(model, opt_cfg, device)
    scheduler = schedule(optimizer, opt_cfg, total_steps)
    cursor = {"epoch": 0, "rows": 0, "step": 0, "target_tokens": 0, "train_seconds": 0.0}
    quality = {"best_nll": None, "best_step": None, "patience_best": None, "stale": 0}

    def save(kind):
        save_checkpoint(artifact_path(cp, kind), {
            "version": 1, "kind": kind, "meta": meta, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(), "cursor": cursor.copy(), "quality": quality.copy(), "rng": rng_state(device),
        })

    if resume:
        env = environment(device)
        if env != meta["environment"]:
            raise ValueError("Full resume requires the same runtime and precision settings")
        if not (run / "run.json").is_file() or read_json(run / "run.json") != meta:
            raise ValueError("Resume requires the original run directory under --runs")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        cursor.update(resume["cursor"])
        quality.update(resume["quality"])
        restore_rng(resume["rng"], device)
        if args.stop_after is not None and args.stop_after <= cursor["step"]:
            raise ValueError("--stop-after must be greater than the resumed optimizer step")
        if quality["best_step"] and not artifact_path(cp, "best").exists():
            raise ValueError("Missing best checkpoint; preserve the original checkpoint directory")
        for log in (metrics_path, sample_path):
            if log.exists():
                write_jsonl(log, (r for r in read_jsonl(log) if r["step"] <= cursor["step"]))
    else:
        if run.exists() or cp.exists():
            raise FileExistsError(f"Run or checkpoint directory for {run_id} already exists")
        run.mkdir(parents=True)
        write_json(run / "run.json", meta)
        write_json(run / "resolved_config.json", config)
        if stage == "sft":
            model.load_state_dict(src["model"])
            del src
        else:
            save_checkpoint(artifact_path(cp, "random"), {
                "version": 1, "kind": "random", "meta": meta, "model": model.state_dict(), "cursor": cursor.copy(),
            })
    meta_path = run / "summary.json"
    val = batch_loader(sets["validation"], range(len(sets["validation"])), batch_size=opt_cfg.micro_batch,
                       pin_memory=device.type == "cuda")
    val_sft = Path(config["data_config"]["paths"]["processed"]) / "sft" / "validation.jsonl"
    started = time.perf_counter()

    def validate():
        scores = evaluate_nll(model, val, device, prec)
        accuracy = validation_samples(stage, model, tok, cfg.get("sampling", {}), sample_path,
                                      cursor["step"], scores["nll"], val_sft)
        return {"event": "validation", "step": cursor["step"], "epoch": cursor["epoch"],
                "tokens_seen": cursor["target_tokens"], "validation": scores, "sampling_accuracy": accuracy,
                "invocation_wall_seconds": time.perf_counter() - started}

    if not resume:
        r0 = validate()
        append(metrics_path, r0)
        print(f"{run_id} initialized; validation NLL {r0['validation']['nll']:.5f}", flush=True)
    reason = "epoch_budget"
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        model.train()
        while cursor["epoch"] < opt_cfg.epochs:
            epoch = cursor["epoch"]
            perm = torch.randperm(len(sets["train"]), generator=torch.Generator().manual_seed(cfg["seed"] + epoch))
            indices = perm[cursor["rows"]:].tolist()
            loader = batch_loader(sets["train"], indices, batch_size=opt_cfg.micro_batch, pin_memory=device.type == "cuda")
            for window in batched_steps(loader, opt_cfg.accumulation):
                start_step = time.perf_counter()
                nll, count, norm, lr = optimize(model, window, optimizer, scheduler, opt_cfg, device, prec)
                synchronize(device)
                dt = time.perf_counter() - start_step
                cursor["step"] += 1
                cursor["rows"] += sum(len(x) for x, _ in window)
                cursor["target_tokens"] += count
                cursor["train_seconds"] += dt
                epoch_done = cursor["rows"] == len(sets["train"])
                if epoch_done:
                    cursor.update(epoch=epoch + 1, rows=0)
                step = cursor["step"]
                append(metrics_path, {
                    "event": "train", "epoch": epoch + 1, "step": step, "tokens_seen": cursor["target_tokens"],
                    "batch_target_tokens": count, "train_nll": nll, "lr": lr, "grad_norm": norm, "tokens_per_second": count/dt,
                    "seconds": dt, "train_seconds": cursor["train_seconds"], "peak_bytes": (
                        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None),
                })
                early = False
                if epoch_done or (opt_cfg.eval_every and step % opt_cfg.eval_every == 0):
                    validation = validate()
                    score = validation["validation"]["nll"]
                    improved = quality["best_nll"] is None or score < quality["best_nll"]
                    if improved:
                        quality.update(best_nll=score, best_step=step)
                    if quality["patience_best"] is None or score < quality["patience_best"] - opt_cfg.min_improvement:
                        quality.update(patience_best=score, stale=0)
                    else:
                        quality["stale"] += 1
                    append(metrics_path, {**validation, "best": improved, "stale": quality["stale"]})
                    if improved:
                        save("best")
                    save("last")
                    early = opt_cfg.patience > 0 and quality["stale"] >= opt_cfg.patience
                    print(f"{run_id} epoch {epoch+1}, step {step}: train {nll:.4f}, val {score:.4f}"
                          + (" *" if improved else ""), flush=True)
                if early:
                    reason = "validation_patience"
                    break
                if args.stop_after is not None and step >= args.stop_after:
                    if not epoch_done:
                        save("last")
                    reason = "requested_stop_after" if step < total_steps else "epoch_budget"
                    break
            if reason in ("validation_patience", "requested_stop_after"):
                break
    except (FloatingPointError, ValueError, RuntimeError) as exc:
        write_json(run / "diagnostic.json", {"cursor": cursor, "error": str(exc)})
        raise
    summary = {
        "run_id": run_id, "stage": stage, "parameters": model.parameter_count(), "cursor": cursor,
        "validation": quality, "stop_reason": reason, "device": str(device), "precision": prec,
        "invocation_wall_seconds": time.perf_counter() - started, "initialization": initialization,
        "best_checkpoint": str(artifact_path(cp, "best")) if quality["best_step"] is not None else None,
        "best_sha256": sha256_file(artifact_path(cp, "best")) if quality["best_step"] is not None else None,
        "last_checkpoint": str(artifact_path(cp, "last")),
    }
    write_json(meta_path, summary)
    return summary


def cli(stage):
    parser = argparse.ArgumentParser(description=f"Run/resume Orchid {stage}.")
    parser.add_argument("--config", type=Path, default=Path(f"configs/{stage}.yaml"))
    parser.add_argument("--run-id")
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--precision", choices=("fp32", "bf16"))
    parser.add_argument("--micro-batch", type=int)
    parser.add_argument("--stop-after", type=int, help="Save full state after this optimizer step; retain schedule horizon.")
    parser.add_argument("--resume", type=Path, help="Resume with matching data and resolved config from last.pt.")
    if stage == "sft":
        init = parser.add_mutually_exclusive_group()
        init.add_argument("--init", type=Path, help="This project's best pretraining checkpoint.")
        init.add_argument("--init-sft", type=Path, help="A matching best SFT checkpoint, with a fresh optimizer.")
    parser.set_defaults(init=None, init_sft=None)
    args = parser.parse_args()
    print(json_text(train(stage, args)))
