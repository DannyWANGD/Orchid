"""Train Orchid GPT from random initialization on the pinned Tang split."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.runner import cli


if __name__ == "__main__":
    cli("pretrain")
