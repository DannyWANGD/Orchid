"""Make paired blind human-review packets; import only actual independent ratings."""

from collections import Counter
import csv
from hashlib import sha256
from io import BytesIO, StringIO
import random
from statistics import mean
from zipfile import ZIP_DEFLATED, ZipFile

from .io import ROOT, read_json, read_jsonl, sha256_file, write_bytes, write_json
from .metrics import parse_poem


DIMENSIONS = ("relevance", "fluency", "coherence", "imagery")
HEAD = ("anonymous_id", "pair_id", "display_order", "keywords", "output", *DIMENSIONS, "notes")
FIXED_FIELDS = HEAD[:5]
RUBRICS = {
    "v1": ROOT / "report/blind_review/reviewer_guide.md",
    "v2": ROOT / "report/blind_review/reviewer_guide_v2.md",
}


def unrated_dimensions(text, rubric_version):
    """Prefill only the ratings that the poem structure makes inapplicable."""
    if rubric_version == "v2" and len(parse_poem(text)) != 4:
        return {"coherence": "NA", "imagery": "NA"}
    return {}


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


def make_packets(generations, protocol, output, models, *, seed=None, rubric_version="v1"):
    if rubric_version not in RUBRICS:
        raise ValueError(f"Unknown review rubric: {rubric_version}")
    rubric_info = ({"rubric_version": rubric_version, "rubric_sha256": sha256_file(RUBRICS[rubric_version])}
                   if rubric_version == "v2" else {})
    if seed is None:
        seed = 20261119 if rubric_version == "v2" else 20261118
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
                                  "keywords":"、".join(row["keywords"]), "output":row["generated"],
                                  **unrated_dimensions(row["generated"], rubric_version)})
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
                         "generation_sha256":sha256_file(generations),"sample_seed":blind["sample_seed"],
                         **rubric_info})
    write_json(output/"ratings_status.json", {"expected_reviewers":2,"received":0,
              "pairs_per_reviewer":len(pairs),"rating_scale":[1,5],"blind_seed":seed,
              "source_key_sha256":sha256_file(key_path), **rubric_info})
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
    version = data.get("rubric_version", "v1")
    if version not in RUBRICS:
        raise ValueError(f"Unknown review rubric: {version}")
    if version == "v2" and data.get("rubric_sha256") != sha256_file(RUBRICS[version]):
        raise ValueError("Version 2 rubric does not match the frozen review")
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
            initial = unrated_dimensions(source["generated"], version)
            if any(r[k].strip() != initial.get(k, "") for k in (*DIMENSIONS, "notes")):
                raise ValueError(f"{path}: organizer template contains edited rating fields")
            order.setdefault(r["pair_id"], []).append(r["display_order"])
            expected[anon] = normalized_fields(r)
        if expected.keys() != key.keys() or any(sorted(items) != ["1", "2"] for items in order.values()):
            raise ValueError(f"{path}: incomplete blind pairs or invalid display positions")
        checked.append(expected)
    return data, checked


def export_reviewer_zips(generations, protocol, key_path, templates, rubric, output):
    data, _ = verified_templates(generations, protocol, key_path, templates)
    rubric = rubric or RUBRICS[data.get("rubric_version", "v1")]
    content = rubric.read_bytes()
    if data.get("rubric_version") == "v2" and sha256(content).hexdigest() != data["rubric_sha256"]:
        raise ValueError("Use the rubric that was frozen with this review")
    if output.exists():
        raise FileExistsError(f"Choose a new reviewer package directory: {output}")
    output.mkdir(parents=True)
    files = []
    for i, path in enumerate(templates, 1):
        buf = BytesIO()
        with ZipFile(buf, "w", compression=ZIP_DEFLATED) as z:
            z.writestr("评分说明.md", content)
            z.writestr(f"reviewer_{i}.csv", path.read_bytes())
        dest = output / f"reviewer_{i}.zip"
        write_bytes(dest, buf.getvalue())
        files.append(dest)
    return {"pairs_per_reviewer": len(data["items"]) // 2,
            "packets": {p.name: sha256_file(p) for p in files}}


def parse_reviews(path, key, reviewer_id, template, *, rubric_version="v1"):
    rows={}
    for r in read_form(path):
        i=r["anonymous_id"]
        if i not in template or i in rows:
            raise ValueError(f"{path}: invalid or repeated review ID {i}")
        if normalized_fields(r) != template[i]:
            raise ValueError(f"{path}: original prompt, output, or presentation fields changed at {i}")
        ratings = {}
        initial = unrated_dimensions(r["output"], rubric_version)
        for dim in DIMENSIONS:
            score = r[dim].strip()
            blocked_imagery = (rubric_version == "v2" and dim == "imagery"
                               and min(ratings["fluency"], ratings["coherence"] or 0) < 3)
            if dim in initial or blocked_imagery:
                if score != "NA":
                    raise ValueError(f"{path}: {dim} must be NA at {i}")
                ratings[dim] = None
                continue
            if score not in ("1","2","3","4","5"):
                raise ValueError(f"{path}: missing or invalid {dim} score at {i}")
            rating=int(score)
            if key[i]["empty"] and rating!=1:
                raise ValueError(f"Empty output must score 1 on its applicable dimensions: {i}")
            ratings[dim] = rating
        rows[i]={"anonymous_id":i,"reviewer":reviewer_id,**ratings,
                 "notes":r["notes"]}
    if rows.keys()!=key.keys():
        raise ValueError(f"{path}: expected {len(key)} reviewed items; got {len(rows)}")
    return rows


def rating_summary(reviewed, pairs, models, grouped, *, rubric_version="v1"):
    scores = {}
    for model in models:
        ids = [grouped[pair, model] for pair in pairs]
        scores[model] = {}
        for dim in DIMENSIONS:
            values = [review[anonymous][dim] for review in reviewed for anonymous in ids]
            rated = [n for n in values if n is not None]
            scores[model][dim] = {"mean": mean(rated) if rated else None, "distribution": dict(Counter(rated))}
            if rubric_version == "v2":
                scores[model][dim].update(rated=len(rated), total=len(values))

    differences = {}
    for dim in DIMENSIONS:
        by_pair = {}
        for pair in pairs:
            matched = [(review[grouped[pair, models[0]]][dim], review[grouped[pair, models[1]]][dim])
                       for review in reviewed]
            if all(before is not None and after is not None for before, after in matched):
                by_pair[pair] = mean(after - before for before, after in matched)
        differences[dim] = {"mean": mean(by_pair.values()) if by_pair else None, "by_pair": by_pair}
        if rubric_version == "v2":
            differences[dim].update(paired=len(by_pair), total_pairs=len(pairs))
    return {"means_and_distributions": scores, "differences": differences}


def score_reviews(generations, protocol, key_path, templates, files, output,
                  *, reviewer_kinds=("human", "human")):
    if len(files) != 2 or len({sha256_file(f) for f in files}) != 2:
        raise ValueError("Provide two distinct, independently completed review files")
    if len(reviewer_kinds) != 2 or any(k not in ("human", "ai") for k in reviewer_kinds):
        raise ValueError("Declare each reviewer as human or ai in template order")
    data, checked = verified_templates(generations, protocol, key_path, templates)
    key, models = data["items"], data["models"]
    version = data.get("rubric_version", "v1")
    reviewed = [parse_reviews(path, key, n, template, rubric_version=version)
                for n, (path, template) in enumerate(zip(files, checked), 1)]
    pairs = sorted({r["pair_id"] for r in key.values()})
    grouped = {(r["pair_id"], r["model"]): i for i, r in key.items()}
    large = []
    both_rated = dict.fromkeys(DIMENSIONS, 0)
    for i in key:
        for dim in DIMENSIONS:
            left, right = (review[i][dim] for review in reviewed)
            if left is None or right is None:
                continue
            both_rated[dim] += 1
            if abs(left - right) >= 2:
                large.append({"anonymous_id": i, "dimension": dim, "ratings": [left, right]})
    human_count = reviewer_kinds.count("human")
    value = {
        "provenance": {
            "mode": "two human ratings" if human_count == 2 else "AI-assisted supplementary ratings",
            "inputs": {str(f): sha256_file(f) for f in files},
            "key_sha256": sha256_file(key_path),
            "template_sha256": [sha256_file(f) for f in templates],
            "prompt_sha256": data["prompt_sha256"], "generation_sha256": data["generation_sha256"],
            "independence": "Reviewer types and independence are declared by the organizer",
        },
        "reviewers": 2, "human_reviewers": human_count, "ai_reviewers": 2 - human_count,
        "pairs": len(pairs), "dimensions": list(DIMENSIONS), "models": models,
        "ratings_by_reviewer": [
            {"reviewer": n, "kind": kind, **rating_summary([review], pairs, models, grouped, rubric_version=version)}
            for n, (kind, review) in enumerate(zip(reviewer_kinds, reviewed), 1)
        ],
        "disagreements_two_or_more": large, "empty_items": sum(r["empty"] for r in key.values()),
        "individual_scores": [v for reviewer in reviewed for v in reviewer.values()],
    }
    if version == "v2":
        value["provenance"].update(rubric_version=version, rubric_sha256=data["rubric_sha256"])
        value["both_reviewers_rated"] = both_rated
    if human_count == 2:
        value.update(rating_summary(reviewed, pairs, models, grouped, rubric_version=version))
    write_json(output, value)
    return value
