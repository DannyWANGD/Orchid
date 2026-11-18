"""Freeze a demo model from validation requests and materialize the final test config."""

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.io import read_json, read_jsonl, read_yaml, sha256_file, write_json


def choose(inputs):
    candidates = {}
    baseline_scope = None
    for folder in inputs:
        meta = read_json(folder/"metrics.json")
        if not str(meta["provenance"]).startswith("measured validation only"):
            raise ValueError("Only the validation protocol may select models")
        scope = (tuple(meta["prompts"]), tuple(meta["seeds"]), meta["sampling"],
                 meta["validation_pairs_sha256"], meta["device"], meta["precision"])
        if baseline_scope is None:
            baseline_scope = scope
        if scope != baseline_scope:
            raise ValueError("All candidates require exactly the same prompts and sampling settings")
        output = folder/"generations.jsonl"
        if sha256_file(output)!=meta["predictions_sha256"]:
            raise ValueError("Prediction hash mismatch")
        groups=defaultdict(list)
        for r in read_jsonl(output):
            if r["suite"]!="validation":
                raise ValueError("Non-validation generation found")
            groups[r["model"]].append(r)
        for model,val in meta["summary"].items():
            cases=groups[model]
            requests={(r["prompt_id"],r["sampling_seed"]) for r in cases}
            if len(requests)!=len(cases) or requests!={(p,s) for p in scope[0] for s in scope[1]}:
                raise ValueError(f"{model} lacks the exact paired requests")
            accepted=sum(r["joint_ok"] and not r["train_matches"]["exact"]
                         and not r["train_matches"]["longer_poem_fragments"] for r in cases)
            if model in candidates:
                entry = candidates[model]
                if (entry["sha256"], entry["joint_without_exact_or_embedded"]) != (val["sha256"], accepted):
                    raise ValueError(f"Conflicting validation evidence for {model}")
            candidates[model]={**{k:val[k] for k in ("checkpoint","sha256","seed","parameters")},
                               "joint_without_exact_or_embedded":accepted,
                               "total_requests":len(cases),"nll":val["nll"]["nll"],
                               "validation_evidence":str(folder/"metrics.json"),
                               "validation_predictions_sha256":meta["predictions_sha256"]}
    rank=sorted(candidates,key=lambda k:(-candidates[k]["joint_without_exact_or_embedded"],
                                          candidates[k]["nll"],candidates[k]["parameters"],k))
    if not rank:
        raise ValueError("No complete validation experiment")
    return {"rule":"maximum joint without exact/embedded train matches; ties NLL, size, run ID",
            "selected_run":rank[0],"rank":rank,"models":candidates,
            "validation_prompt_count":len(baseline_scope[0]),"sampling_seeds":baseline_scope[1],
            "validation_pairs_sha256":baseline_scope[3], "device":baseline_scope[4], "precision":baseline_scope[5]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--validation-runs",type=Path,nargs="+",required=True)
    p.add_argument("--base-eval",type=Path,default=Path("configs/eval.yaml"))
    p.add_argument("--runs-dir",type=Path,default=Path("runs"),
                   help="Training summaries used to verify the selected model's initialization")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--eval-output",type=Path,required=True)
    a=p.parse_args()
    result=choose(a.validation_runs)
    write_json(a.output,result)
    c=deepcopy(read_yaml(a.base_eval))
    selected=result["selected_run"]
    c["models"]={**c["models"]}
    c["models"]["X-42M"]="checkpoints/sft/sft-tang-42m-2026/best.pt"
    c["models"]["X-92M-10"]="checkpoints/sft/sft-tang-92m-2026/best.pt"
    c["models"]["X-92M-TangSong-Pre"]="checkpoints/pretrain/pretrain-tang-song-92m/best.pt"
    for seed in (2026,2027,2028):
        c["models"][f"X-92M-Tang-Long-{seed}"]=f"checkpoints/sft/sft-tang-92m-26-{seed}/best.pt"
        c["models"][f"X-92M-TangSong-{seed}"]=f"checkpoints/sft/sft-tang-song-92m-{seed}/best.pt"
    # Give the selected checkpoint a distinct ID so its new control scores never
    # replace the archived 14M R2-2027 results.
    selection=result["models"][selected]
    if sha256_file(Path(selection["checkpoint"]))!=selection["sha256"]:
        raise ValueError("Selected validation weights changed")
    matched = [name for name,path in c["models"].items() if path==selection["checkpoint"]]
    c["selected"]=matched[0] if matched else "X-selected"
    c["models"][c["selected"]]=selection["checkpoint"]
    summary = read_json(a.runs_dir / selected / "summary.json")
    parent = summary["initialization"]
    if parent["kind"] != "pretrain_best" or sha256_file(Path(parent["path"])) != parent["sha256"]:
        raise ValueError("Could not verify the selected model's own pretraining baseline")
    paired = [name for name,path in c["models"].items() if path==parent["path"]]
    c["paired_pretrain"]=paired[0] if paired else "X-selected-pre"
    c["models"][c["paired_pretrain"]]=parent["path"]
    c["expanded_models"]={"data":"configs/data-tang-song.yaml",
                          "names":[k for k,v in c["models"].items() if "tang-song" in v]}
    c["keyword_constraint_control"]=True
    c["seed_groups"] = {
        "Tang_14M": [f"R2-{s}" for s in (2026, 2027, 2028)],
        "TangSong_92M": [f"X-92M-TangSong-{s}" for s in (2026, 2027, 2028)],
    }
    write_json(a.eval_output,c)
    print("Selected from validation only:", selected, c["selected"])


if __name__=="__main__":
    main()
