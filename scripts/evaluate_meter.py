"""Report conservative canonical tone-template and rhyme coverage."""

import argparse
from collections import Counter
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.io import read_jsonl, sha256_file, write_json
from orchid.metrics import rhyme_table
from orchid.prosody import template_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generations",type=Path,required=True)
    p.add_argument("--rhymes",type=Path,default=Path("data/raw/pingshui.json"))
    p.add_argument("--suite",default="main")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    table=rhyme_table(a.rhymes)
    result={}
    seen=Counter()
    for row in read_jsonl(a.generations):
        if row["suite"]!=a.suite:
            continue
        name=row["model"]
        seen[name]+=1
        r=result.setdefault(name,{"tone":Counter(),"tone_rhyme":Counter(),"examples":{}})
        if not row["format_ok"]:
            continue
        scores=template_score(row["lines"],table)
        for k in ("tone","tone_rhyme"):
            s=scores[k]
            r[k][s]+=1
            r["examples"].setdefault(k+"/"+s,
                {"prompt_id":row["prompt_id"],"seed":row["sampling_seed"],"generated":row["generated"],
                 "rhyme":row["rhyme"],**scores})
    for name,r in result.items():
        n=sum(r["tone"].values())
        r["shape"]= {"count":n,"total":seen[name]}
        for k in ("tone","tone_rhyme"):
            for s in ("pass","fail","unknown"):
                r[k].setdefault(s,0)
    write_json(a.output,{"source_sha256":sha256_file(a.generations),"rhyme_sha256":sha256_file(a.rhymes),
                         "scope":"exact canonical five-character tone templates; no historical exceptions",
                         "suite":a.suite,"models":result})
    for name,r in result.items():
        print(name,r["shape"],dict(r["tone"]),dict(r["tone_rhyme"]))


if __name__=="__main__":
    main()
