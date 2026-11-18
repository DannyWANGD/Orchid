"""Build an expanded pretraining set while keeping the Tang holdouts fixed."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.expansion import prepare
from orchid.io import json_text, read_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data-tang-song.yaml"))
    args = parser.parse_args()
    stats = prepare(read_yaml(args.config))
    print(json_text({k:v for k,v in stats.items() if k!="exclusion_examples"}))


if __name__=="__main__":
    main()
