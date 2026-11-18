"""Adapt Orchid's self-trained GPT with response-only supervision."""

from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.runner import cli


if __name__ == "__main__":
    cli("sft")
