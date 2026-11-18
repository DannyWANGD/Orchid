"""Small shared helpers for reproducible configuration and artifact writes."""

from collections.abc import Iterable, Iterator
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_yaml(path: Path) -> dict:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as dest:
        tmp = Path(dest.name)
        try:
            dest.write(content)
            dest.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def json_text(value: Any, indent: int | None = 2) -> str:
    return json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True, allow_nan=False)


def write_json(path: Path, value: Any, indent: int | None = 2) -> None:
    write_bytes(path, (json_text(value, indent) + "\n").encode())


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, mode="w", encoding="utf-8", delete=False
    ) as dest:
        tmp = Path(dest.name)
        try:
            for row in records:
                dest.write(json_text(row, indent=None) + "\n")
            dest.close()
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()
