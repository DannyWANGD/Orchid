"""Fetch and hash the fixed Song poetry shards for the larger-corpus experiment."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import date
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import quote

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.corpus import fetch
from orchid.io import read_json, read_yaml, sha256_file, write_bytes, write_json


def snapshot(cfg, offline):
    source=cfg["source"]
    raw=Path(cfg["paths"]["raw"])
    path=Path(cfg["paths"]["artifacts"])/"song_source.json"
    previous=read_json(path) if path.exists() else None
    if previous and previous["source"]!=source:
        raise ValueError("Pinned Song source changed")
    expected={x["name"]:x["sha256"] for x in previous["files"]} if previous is not None else {}
    prefix=f"https://raw.githubusercontent.com/{source['repo']}/{source['revision']}/"+quote(source["directory"])
    files=[f"poet.song.{v*1000}.json" for v in range(source["song_shards"])]

    def get(name):
        dst=raw/name
        url=f"{prefix}/{name}"
        if name in expected and dst.exists() and sha256_file(dst)==expected[name]:
            return {"name":name,"bytes":dst.stat().st_size,"sha256":expected[name],"url":url}
        if offline:
            raise ValueError(f"Missing verified offline data file {name}")
        content=fetch(url)
        sha=hashlib.sha256(content).hexdigest()
        if expected and sha!=expected.get(name):
            raise ValueError(f"Upstream bytes changed at {url}")
        if not isinstance(json.loads(content),list):
            raise ValueError(f"Missing JSON array at {url}")
        write_bytes(dst,content)
        return {"name":name,"bytes":len(content),"sha256":sha,"url":url}

    with ThreadPoolExecutor(max_workers=12) as pool:
        rows=list(pool.map(get,files))
    write_json(path, {"source":source,"files":rows,
                      "retrieved_on":previous["retrieved_on"] if previous else date.today().isoformat()})
    print(f"{len(rows)} Song shards; {sum(r['bytes'] for r in rows):,} bytes")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,default=Path("configs/song.yaml"))
    p.add_argument("--offline",action="store_true")
    a=p.parse_args()
    snapshot(read_yaml(a.config),a.offline)


if __name__=="__main__":
    main()
