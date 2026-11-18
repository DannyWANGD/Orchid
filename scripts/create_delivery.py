"""Build and verify a local course ZIP with one inference bundle and source."""

import argparse
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import zipfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.inference import verify_bundle
from orchid.io import ROOT, sha256_file, write_json


PROJECT_DIRS = ("orchid", "scripts", "configs", "docs", "tests", "data/lexicon", "data/prompts", "artifacts")
ROOT_FILES = ("README.md", "GPT_BUILD_CHECKLIST.md", "Problem.md", "THIRD_PARTY_NOTICES.md", "requirements.txt",
              ".python-version", ".gitignore")
REPORT_FILES = ("experiment_report.md", "Orchid课程报告.pdf", "pretraining.md", "finetuning.md",
                "scaling.md", "evaluation_baseline.md", "evaluation_expanded.md", "prosody_source_audit.md")


def make_zip(output, bundle):
    bundle = bundle.resolve()
    meta = verify_bundle(bundle)
    seen = set()
    with TemporaryDirectory(prefix="orchid-distribution-", dir=output.parent) as td:
        temporary = Path(td)/"orchid.zip"
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
            def add(src, name):
                key = "orchid-course/" + str(name)
                if key in seen or not src.is_file():
                    raise ValueError(f"Duplicate or missing delivery item: {src}")
                z.write(src, key)
                seen.add(key)

            for name in ROOT_FILES:
                add(ROOT/name, name)
            for directory in (*PROJECT_DIRS, "report/figures", "report/blind_review"):
                for file in sorted((ROOT/directory).rglob("*")):
                    if file.is_file() and "__pycache__" not in file.parts and file.suffix in (
                        ".py", ".json", ".yaml", ".md", ".txt", ".pdf", ".svg", ".csv"
                    ) and not file.name.endswith(("_completed.csv", "_completed.md")):
                        add(file, file.relative_to(ROOT))
            for name in REPORT_FILES:
                path = ROOT/"report"/name
                add(path, path.relative_to(ROOT))
            for name in ("weights.pt", "tokenizer.json", "config.json", "bundle.json"):
                add(bundle/name, "model/"+name)
            for filename in sorted((ROOT/"report/evidence").rglob("*")):
                if (filename.is_file() and filename.suffix in (".json", ".jsonl", ".csv")
                        and not filename.name.startswith("delivery-")):
                    add(filename, filename.relative_to(ROOT))
            z.writestr("orchid-course/QUICKSTART.txt",
                       "Orchid course delivery. Extract the whole zip to preserve source/module paths.\n"
                       "Python 3.11.16, Linux x86_64; install via python -m pip install -r requirements.txt\n"
                       "From this directory, with the venv active:\n"
                       "  python -E scripts/generate.py --checkpoint model --keywords 秋雨 孤舟 --device cpu\n"
                       "  python -E scripts/generate.py --checkpoint model --keywords 霜 孤舟 --device cuda --constrain-keywords\n"
                       "  python -E -m unittest discover -s tests -v\n"
                       "The model runs on CPU without data/ or GPU. CUDA optional for inference.\n"
                       "Verified source and data URLs are in THIRD_PARTY_NOTICES.md and the README.\n")
        with zipfile.ZipFile(temporary) as archive:
            issue = archive.testzip()
            if issue:
                raise IOError(f"Corrupt archive entry: {issue}")
            for name,digest in meta["files"].items():
                with archive.open(f"orchid-course/model/{name}") as f:
                    if hashlib.file_digest(f,"sha256").hexdigest()!=digest:
                        raise ValueError(f"{name} changed while packaging")
        temporary.replace(output)
    return {"zip":str(output), "zip_sha256":sha256_file(output), "bytes":output.stat().st_size,
            "bundle":meta, "file_count":len(seen)+1}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    manifest=make_zip(a.output,a.bundle)
    index=a.output.with_suffix(".json")
    write_json(index,manifest)
    print(f"{a.output}: {manifest['bytes']:,} bytes, {manifest['file_count']} verified files")
    print(index)


if __name__=="__main__":
    main()
