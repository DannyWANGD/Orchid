"""Evaluate Orchid on frozen independent poems, references and invalid requests."""

import argparse
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.evaluation import run_eval
from orchid.io import read_yaml


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=Path("configs/eval.yaml"))
    p.add_argument("--output", type=Path, default=Path("runs/fixed-eval-v1"))
    p.add_argument("--device", choices=("cpu","cuda"))
    p.add_argument("--smoke", action="store_true", help="Use a marked three-prompt subset per suite for runtime checks.")
    p.add_argument("--variants-only", action="store_true", help="Evaluate named X-* extensions separately.")
    a = p.parse_args()
    out = run_eval(read_yaml(a.config), a.output, device_override=a.device, smoke=a.smoke,
                   variants_only=a.variants_only)
    for name, suites in out["summary"].items():
        m = suites["main"]["overall"]
        print(f"{name}: joint {m['joint_ok']['count']}/{m['requests']}")
    print(a.output/"metrics.json")


if __name__ == "__main__":
    main()
