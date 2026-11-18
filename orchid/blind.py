"""Make paired blind human-review packets; import only actual independent ratings."""

from collections import Counter
import csv
from io import StringIO
from pathlib import Path
import random
from statistics import mean

from .io import read_json, read_jsonl, sha256_file, write_bytes, write_json


DIMENSIONS = ("relevance", "fluency", "coherence", "imagery")
HEAD = ("anonymous_id", "pair_id", "display_order", "keywords", "output", *DIMENSIONS, "notes")


def write_csv(path, records):
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HEAD, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    write_bytes(path, buffer.getvalue().encode("utf-8-sig"))


def make_packets(generations, protocol, output, models, *, seed=20261118):
    p = read_json(protocol)
    blind = p["blind"]
    fixed = set(blind["prompt_ids"])
    index = {
        (r["prompt_id"],r["model"]):r for r in read_jsonl(generations)
        if r["suite"]=="main" and r["model"] in models and r["sampling_seed"]==blind["sample_seed"]
        and r["prompt_id"] in fixed
    }
    expected = {(i, m) for i in fixed for m in models}
    if set(index) != expected:
        raise ValueError("Incomplete frozen blind pairs")
    if output.exists():
        raise FileExistsError(f"Keep existing reviews; choose a new output directory: {output}")
    output.mkdir(parents=True)
    # The key is deliberately stored separately; only the review packet goes to each rater.
    rng = random.Random(seed)
    pairs = sorted(fixed)
    rng.shuffle(pairs)
    key = {}
    records = {}
    for n,i in enumerate(pairs,1):
        pair = f"P{n:03d}"
        records[pair] = []
        for m in models:
            row = index[i,m]
            anonymous = f"A{rng.getrandbits(40):010x}"
            if anonymous in key:
                raise AssertionError("Anonymous identifier collision")
            key[anonymous] = {"pair_id":pair,"source_prompt":i,"model":m,
                              "empty": not row["generated"].strip()}
            records[pair].append({"anonymous_id":anonymous,"pair_id":pair,
                                  "keywords":"、".join(row["keywords"]), "output":row["generated"]})
    versions = ("reviewer_1.csv", "reviewer_2.csv")
    for version in versions:
        rows = []
        order = pairs[:]
        order = sorted(records)
        rng.shuffle(order)
        for pair in order:
            perm = records[pair][:]
            rng.shuffle(perm)
            rows.extend({**r, "display_order": j} for j,r in enumerate(perm,1))
        write_csv(output/version, rows)
    key_path=output/"private_key.json"
    write_json(key_path, {"models":models,"items":key,"prompt_sha256":sha256_file(protocol),
                         "generation_sha256":sha256_file(generations),"sample_seed":blind["sample_seed"]})
    write_json(output/"ratings_status.json", {"expected_reviewers":2,"received":0,
              "pairs_per_reviewer":len(pairs),"rating_scale":[1,5],"blind_seed":seed,
              "source_key_sha256":sha256_file(key_path)})
    return versions


def parse_reviews(path, key, reviewer_id):
    with path.open(encoding="utf-8-sig",newline="") as f:
        raw=list(csv.DictReader(f))
    rows={}
    for r in raw:
        i=r["anonymous_id"]
        if i not in key or i in rows or r["pair_id"]!=key[i]["pair_id"]:
            raise ValueError(f"{path}: invalid or repeated review ID {i}")
        ratings=[]
        for dim in DIMENSIONS:
            score = r[dim].strip()
            if score not in ("1","2","3","4","5"):
                raise ValueError(f"{path}: missing or invalid {dim} score at {i}")
            rating=int(score)
            if key[i]["empty"] and rating!=1:
                raise ValueError(f"Empty output must score 1 in every dimension: {i}")
            ratings.append(rating)
        rows[i]={"anonymous_id":i,"reviewer":reviewer_id,**dict(zip(DIMENSIONS,ratings)),
                 "notes":r["notes"]}
    if rows.keys()!=key.keys():
        raise ValueError(f"{path}: expected {len(key)} reviewed items; got {len(rows)}")
    return rows


def score_reviews(key_path, files, output):
    if len(files)!=2 or len({sha256_file(f) for f in files})!=2:
        raise ValueError("Provide two distinct, independently completed review files")
    data=read_json(key_path)
    key, models=data["items"],data["models"]
    reviewed=[parse_reviews(path,key,n) for n,path in enumerate(files,1)]
    result={}
    for m in models:
        ids=[i for i,r in key.items() if r["model"]==m]
        result[m]={d:{"mean":mean(rs), "distribution":dict(Counter(rs))}
                   for d in DIMENSIONS
                   for rs in [[rev[i][d] for rev in reviewed for i in ids]]}
    pairs=sorted({r["pair_id"] for r in key.values()})
    grouped={ (r["pair_id"],r["model"]):i for i,r in key.items() }
    delta={d:[mean(rev[grouped[p,models[1]]][d]-rev[grouped[p,models[0]]][d] for rev in reviewed)
                for p in pairs] for d in DIMENSIONS}
    large=[{"anonymous_id":i,"dimension":d,"ratings":[rev[i][d] for rev in reviewed]}
           for i in key for d in DIMENSIONS if abs(reviewed[0][i][d]-reviewed[1][i][d]) >= 2]
    value={
        "provenance":{"mode":"independent human ratings supplied by reviewer",
                      "inputs":{str(f):sha256_file(f) for f in files},"key_sha256":sha256_file(key_path)},
        "reviewers":2,"pairs":len(pairs),"dimensions":list(DIMENSIONS),"models":models,
        "means_and_distributions":result,"differences":{d:{"mean":mean(v),"by_pair":dict(zip(pairs,v))}
                                                        for d,v in delta.items()},
        "disagreements_two_or_more":large,"empty_items":sum(r["empty"] for r in key.values()),
        "individual_scores":[v for reviewer in reviewed for v in reviewer.values()],
    }
    write_json(output,value)
    return value
