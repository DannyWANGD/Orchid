"""Fetch and clean the fixed Tang snapshot; split before training."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.corpus import prepare, snapshot
from orchid.io import read_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data.yaml"))
    parser.add_argument("--offline", action="store_true", help="Require the verified files already on disk.")
    args = parser.parse_args()
    config = read_yaml(args.config)
    manifest = snapshot(config, offline=args.offline)
    stats = prepare(config, manifest)
    print(f"{stats['raw']:,} raw; {stats['unique']:,} distinct; {stats['wujue_unique']:,} four-by-five")
    for name, values in stats["splits"].items():
        print(f"{name}: {values['poems']:,} poems; {values['wujue_candidates']:,} four-by-five")


if __name__ == "__main__":
    main()
