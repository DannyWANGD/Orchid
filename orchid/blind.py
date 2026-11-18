"""Make paired blind human-review packets; import only actual independent ratings."""

from collections import Counter
import csv
from io import BytesIO, StringIO
import random
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

from .io import read_json, read_jsonl, sha256_file, write_bytes, write_json


DIMENSIONS = ("relevance", "fluency", "coherence", "imagery")
HEAD = ("anonymous_id", "pair_id", "display_order", "keywords", "output", *DIMENSIONS, "notes")
FIXED_FIELDS = HEAD[:5]


def write_csv(path, records):
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HEAD, lineterminator="\n")
    writer.writeheader()
    writer.writerows(records)
    write_bytes(path, buffer.getvalue().encode("utf-8-sig"))


def source_pairs(generations, blind, models):
    if len(models) != 2 or len(set(models)) != 2:
        raise ValueError("Blind review needs exactly two different models")
    prompts = blind["prompt_ids"]
    if not prompts or len(prompts) != len(set(prompts)):
        raise ValueError("Blind review needs a nonempty set of distinct prompts")
    expected = {(i, m) for i in prompts for m in models}
    index = {}
    for row in read_jsonl(generations):
        pair = row["prompt_id"], row["model"]
        if row["suite"] == "main" and row["sampling_seed"] == blind["sample_seed"] and pair in expected:
            if pair in index:
                raise ValueError("Duplicate frozen generation for a blind pair")
            index[pair] = row
    if index.keys() != expected:
        raise ValueError("Incomplete frozen blind pairs")
    return index


def make_packets(generations, protocol, output, models, *, seed=20261118):
    blind = read_json(protocol)["blind"]
    index = source_pairs(generations, blind, models)
    if output.exists():
        raise FileExistsError(f"Keep existing reviews; choose a new output directory: {output}")
    output.mkdir(parents=True)
    # The key is deliberately stored separately; only the review packet goes to each rater.
    rng = random.Random(seed)
    pairs = sorted(blind["prompt_ids"])
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


def read_form(path):
    return parse_form_bytes(path.read_bytes(), path)


def parse_form_bytes(content, path):
    reader = csv.DictReader(StringIO(content.decode("utf-8-sig"), newline=""))
    if not reader.fieldnames or sorted(reader.fieldnames) != sorted(HEAD):
        raise ValueError(f"{path}: expected the original review form columns")
    rows = list(reader)
    if any(None in r or None in r.values() for r in rows):
        raise ValueError(f"{path}: malformed review form row")
    return rows


def normalized_fields(row):
    return tuple(row[k].replace("\r\n", "\n") for k in FIXED_FIELDS)


def verified_templates(generations, protocol, key_path, templates):
    """Check organizer templates against the originally frozen prompts and generations."""
    data = read_json(key_path)
    if (data["prompt_sha256"] != sha256_file(protocol)
            or data["generation_sha256"] != sha256_file(generations)):
        raise ValueError("Blind key does not match the frozen prompts and generations")
    blind = read_json(protocol)["blind"]
    if data["sample_seed"] != blind["sample_seed"]:
        raise ValueError("Blind key uses a different sampling seed")
    key = data["items"]
    sources = source_pairs(generations, blind, data["models"])
    source_keys = [(r["source_prompt"], r["model"]) for r in key.values()]
    if len(key) != len(sources) or set(source_keys) != sources.keys():
        raise ValueError("Blind key does not map every frozen output exactly once")
    for pair_id in {r["pair_id"] for r in key.values()}:
        pair = [r for r in key.values() if r["pair_id"] == pair_id]
        if (len(pair) != 2 or len({r["source_prompt"] for r in pair}) != 1
                or {r["model"] for r in pair} != set(data["models"])):
            raise ValueError(f"{pair_id} does not pair the same prompt across both models")
    if len(templates) != 2 or len({p.resolve() for p in templates}) != 2:
        raise ValueError("Expected the two original, blank organizer forms")
    checked = []
    for path in templates:
        expected = {}
        order = {}
        for r in read_form(path):
            anon = r["anonymous_id"]
            if anon not in key or anon in expected:
                raise ValueError(f"{path}: invalid or repeated review ID {anon}")
            entry = key[anon]
            source = sources[entry["source_prompt"], entry["model"]]
            if (r["pair_id"] != entry["pair_id"] or r["keywords"] != "、".join(source["keywords"])
                    or r["output"].replace("\r\n", "\n") != source["generated"].replace("\r\n", "\n")
                    or entry["empty"] != (not source["generated"].strip())):
                raise ValueError(f"{path}: changed prompt or poem for {anon}")
            if any(r[k].strip() for k in (*DIMENSIONS, "notes")):
                raise ValueError(f"{path}: organizer template must be blank")
            order.setdefault(r["pair_id"], []).append(r["display_order"])
            expected[anon] = normalized_fields(r)
        if expected.keys() != key.keys() or any(sorted(items) != ["1", "2"] for items in order.values()):
            raise ValueError(f"{path}: incomplete blind pairs or invalid display positions")
        checked.append(expected)
    return data, checked


def export_reviewer_zips(generations, protocol, key_path, templates, rubric, output):
    data, _ = verified_templates(generations, protocol, key_path, templates)
    if output.exists():
        raise FileExistsError(f"Choose a new reviewer package directory: {output}")
    output.mkdir(parents=True)
    files = []
    for i, path in enumerate(templates, 1):
        buf = BytesIO()
        with ZipFile(buf, "w", compression=ZIP_DEFLATED) as z:
            z.writestr("评分说明.md", rubric.read_bytes())
            z.writestr(f"reviewer_{i}.csv", path.read_bytes())
        dest = output / f"reviewer_{i}.zip"
        write_bytes(dest, buf.getvalue())
        files.append(dest)
    return {"pairs_per_reviewer": len(data["items"]) // 2,
            "packets": {p.name: sha256_file(p) for p in files}}


def parse_reviews(path, key, reviewer_id, template):
    rows={}
    for r in read_form(path):
        i=r["anonymous_id"]
        if i not in template or i in rows:
            raise ValueError(f"{path}: invalid or repeated review ID {i}")
        if normalized_fields(r) != template[i]:
            raise ValueError(f"{path}: original prompt, output, or presentation fields changed at {i}")
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


def score_reviews(generations, protocol, key_path, templates, files, output):
    if len(files)!=2 or len({sha256_file(f) for f in files})!=2:
        raise ValueError("Provide two distinct, independently completed review files")
    data, checked = verified_templates(generations, protocol, key_path, templates)
    key, models=data["items"],data["models"]
    reviewed=[parse_reviews(path,key,n,template) for n,(path,template) in enumerate(zip(files, checked),1)]
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
                      "inputs":{str(f):sha256_file(f) for f in files}, "key_sha256":sha256_file(key_path),
                      "template_sha256":[sha256_file(f) for f in templates],
                      "prompt_sha256":data["prompt_sha256"], "generation_sha256":data["generation_sha256"],
                      "independence":"Organizer assigns each form to a different reviewer; not inferred from CSVs"},
        "reviewers":2,"pairs":len(pairs),"dimensions":list(DIMENSIONS),"models":models,
        "means_and_distributions":result,"differences":{d:{"mean":mean(v),"by_pair":dict(zip(pairs,v))}
                                                        for d,v in delta.items()},
        "disagreements_two_or_more":large,"empty_items":sum(r["empty"] for r in key.values()),
        "individual_scores":[v for reviewer in reviewed for v in reviewer.values()],
    }
    write_json(output,value)
    return value
