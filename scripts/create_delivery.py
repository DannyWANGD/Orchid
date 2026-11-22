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


PROJECT_DIRS = (
    "orchid", "scripts", "configs", "docs", "tests", "data/lexicon", "data/prompts", "artifacts", "assets",
    "demo", "report/latex",
)
PROJECT_SUFFIXES = (
    ".py", ".json", ".yaml", ".md", ".txt", ".pdf", ".svg", ".csv",
    ".html", ".css", ".js", ".png", ".tex", ".sty", ".bib",
)
ROOT_FILES = ("README.md", "GPT_BUILD_CHECKLIST.md", "Problem.md", "THIRD_PARTY_NOTICES.md", "requirements.txt",
              "requirements-cpu.txt", ".python-version", ".gitignore", ".vscode/settings.example.json")
REPORT_FILES = ("experiment_report.md", "Orchid课程报告.pdf", "pretraining.md", "finetuning.md",
                "scaling.md", "evaluation_baseline.md", "evaluation_expanded.md", "prosody_source_audit.md",
                "prosody_optimization.md", "release_evaluation.md")
BLIND_DOCS = ("README.md", "reviewer_guide.md", "reviewer_guide_v2.md", "meter_review.md", "ratings_status.json",
              "expanded/ratings_status.json")
PRIVATE_FILES = {
    "private_key.json", "reviewer_1.csv", "reviewer_2.csv", "meter_review_completed.md", "cover.local.tex",
}


def public_files(directory, suffixes):
    for file in sorted(directory.rglob("*")):
        if (file.is_file() and not file.is_symlink() and "__pycache__" not in file.parts
                and file.suffix in suffixes and file.name not in PRIVATE_FILES
                and not file.name.endswith("_completed.csv")):
            yield file


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
            if (ROOT/"LICENSE").is_file():
                add(ROOT/"LICENSE", "LICENSE")
            for directory in (*PROJECT_DIRS, "report/figures"):
                for file in public_files(ROOT/directory, PROJECT_SUFFIXES):
                    add(file, file.relative_to(ROOT))
            blind = Path("report/blind_review")
            for name in BLIND_DOCS:
                add(ROOT/blind/name, blind/name)
            for name in REPORT_FILES:
                path = ROOT/"report"/name
                add(path, path.relative_to(ROOT))
            for name in ("weights.pt", "tokenizer.json", "config.json", "bundle.json"):
                add(bundle/name, "model/"+name)
            for filename in public_files(ROOT/"report/evidence", (".json", ".jsonl", ".csv")):
                add(filename, filename.relative_to(ROOT))
            z.writestr("orchid-course/QUICKSTART.txt",
                       "Orchid 92M model kit. Extract the whole zip to preserve source/module paths.\n"
                       "Python 3.11.16, Linux x86_64. Create and activate a virtual environment.\n"
                       "For CPU, run python -m pip install -r requirements-cpu.txt\n"
                       "For CUDA 12.6, run python -m pip install -r requirements.txt\n"
                       "Run these from the extracted project directory.\n"
                       "  python -E scripts/generate.py --checkpoint model --keywords 秋雨 孤舟 --device cpu\n"
                       "  python -E scripts/generate.py --checkpoint model --keywords 霜 孤舟 --device cuda --json\n"
                       "  python -E scripts/serve_demo.py --checkpoint model --device cpu\n"
                       "  python -E -m unittest discover -s tests -v\n"
                       "Open demo/index.html to browse the frozen examples offline.\n"
                       "CPU inference uses the provided weights and rhyme table. Use --mode raw for free sampling.\n"
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
