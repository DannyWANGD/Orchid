"""Construct reproducible weak keyword supervision from the frozen Tang splits."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.io import json_text, read_yaml
from orchid.supervision import Task, build_supervision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    args = parser.parse_args()
    cfg = read_yaml(args.config)
    result = build_supervision(read_yaml(Path(cfg["data"])), cfg["supervision"], Task(**read_yaml(Path(cfg["task"]))),
                               read_yaml(Path(cfg["model"]))["context_length"])
    print(json_text(result))


if __name__ == "__main__":
    main()
