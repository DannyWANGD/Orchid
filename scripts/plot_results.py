"""Render standalone course figures directly from training/evaluation logs."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MaxNLocator

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.io import read_json, read_jsonl

STYLE = {"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
         "axes.spines.right": False, "svg.fonttype": "none", "svg.hashsalt": "orchid-v1", "pdf.fonttype": 42}
PALETTE = ["#005b80", "#a64b00", "#67528f", "#2f755e"]


def save(fig, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".svg"):
        path = output.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight",
                    metadata={"CreationDate": None} if suffix == ".pdf" else {"Date": None})
        if suffix == ".svg":
            path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def histories(logs, output):
    all_rows = []
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7.0, 3.5), layout="constrained")
        for i, path in enumerate(logs):
            name = path.parent.name
            color = PALETTE[i % len(PALETTE)]
            rows = list(read_jsonl(path))
            train = [r for r in rows if r["event"] == "train"]
            valid = [r for r in rows if r["event"] == "validation"]
            ax.plot([r["tokens_seen"]/1e6 for r in train], [r["train_nll"] for r in train],
                    lw=0.7, alpha=0.43, color=color, label="Train steps" if len(logs)==1 else None)
            ax.plot([r["tokens_seen"]/1e6 for r in valid], [r["validation"]["nll"] for r in valid],
                    lw=1.8, marker="o", ms=3, color=color,
                    label="Validation" if len(logs)==1 else f"Validation · {name}")
            for r in rows:
                val = r.get("validation", {})
                all_rows.append({
                    "run": name, "kind": r["event"], "step": r["step"], "tokens_seen": r["tokens_seen"],
                    "nll": r.get("train_nll", val.get("nll")), "ppl": val.get("ppl"),
                })
        ax.set(xlabel="Effective target tokens seen (millions)", ylabel="NLL (nats / token)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax.grid(axis="y", color="#e5e8ea", lw=0.7)
        ax.legend(frameon=False, loc="best")
        save(fig, output)
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        columns = ("run", "kind", "step", "tokens_seen", "nll", "ppl")
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)


def architecture(output):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8.0, 3.15), layout="constrained")
        ax.set_xlim(0, 10.2)
        ax.set_ylim(0, 4.65)
        ax.axis("off")
        entries = [(0.2, 1.6, "Character IDs\n[B, T]"), (2.2, 1.65, "Token + position\nembedding"),
                   (4.35, 2.15, "6 × decoder block\n[B, T, 384]"),
                   (7.05, 1.22, "Final LN"), (8.64, 1.48, "Tied head\n[B, T, 8582]")]
        for idx, (x, w, label) in enumerate(entries):
            ax.add_patch(FancyBboxPatch((x, 2.92), w, .90, boxstyle="round,pad=0.04",
                                        ec=PALETTE[0], fc="#e8f2f7", lw=1))
            ax.text(x+w/2, 3.37, label, ha="center", va="center", fontsize=9)
            if idx:
                xp, wp, _ = entries[idx-1]
                ax.annotate("", xy=(x-.06, 3.37), xytext=(xp+wp+.06, 3.37),
                            arrowprops={"arrowstyle": "->", "lw": 1, "color": "#505a60"})
        ax.annotate("shared token matrix", xy=(9.38, 3.87), xytext=(3.02, 4.38),
                    fontsize=9, ha="center", color=PALETTE[0],
                    arrowprops={"arrowstyle": "->", "color": PALETTE[0],
                                "connectionstyle": "angle3,angleA=0,angleB=90"})
        ax.add_patch(FancyBboxPatch((1.0, .42), 8.15, 1.66, boxstyle="round,pad=0.07",
                                    ec="#8d9aa1", fc="#f7f8f8", lw=1))
        ax.plot([5.43, 5.43], [2.12, 2.87], ls=":", lw=1, color="#667781")
        ax.text(1.24, 1.64, "Block: x = x + CausalAttention(LN(x))", va="center", weight="bold")
        ax.text(1.24, 1.18, "6 heads × 64 dimensions; mask future positions before softmax", va="center", fontsize=9)
        ax.text(1.24, .73, "Block: x = x + MLP(LN(x)); 384 → 1536 → 384 with GELU", va="center", fontsize=9)
        save(fig, output)


def writing_comparison(metrics_path, output):
    m = read_json(metrics_path)
    summary = m["summary"]
    selected = m["selected"]
    names = [n for n in ("R1", selected, "R2-K", "R3") if n in summary]
    captions = {"R1": "Pretraining", selected: "After SFT", "R2-K": "Swapped input", "R3": "Forced form"}
    keys = {"format_ok": "Four × five", "keywords_ok": "All keywords", "joint_ok": "Both"}
    data = [{"model": n, "score": k, **summary[n]["main"]["overall"][k]}
            for n in names for k in keys]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.5), sharey=True, layout="constrained")
        for ax, (metric, title) in zip(axes, keys.items()):
            samples = [r for r in data if r["score"] == metric]
            values = [100*r["rate"] for r in samples]
            bars = ax.bar(range(len(values)), values, width=.65, color=PALETTE[:len(values)])
            for b, row in zip(bars, samples):
                ax.text(b.get_x()+b.get_width()/2, max(1.0,b.get_height()+1.4), str(row["count"]),
                        fontsize=8, ha="center", va="bottom")
            ax.set_title(title)
            ax.set_ylim(0,112)
            ax.set_yticks(range(0,101,25))
            ax.set_xticks(range(len(names)), [captions[n] for n in names], rotation=28, ha="right", fontsize=8)
            ax.grid(axis="y",color="#e5e8ea",lw=.7)
            ax.set_axisbelow(True)
        axes[0].set_ylabel("Share of requests on independent prompts (%)")
        save(fig, output)
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=("model", "score", "count", "total", "rate"), lineterminator="\n")
        w.writeheader()
        w.writerows(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="kind", required=True)
    tr = commands.add_parser("history", help="Training and validation NLL vs target tokens.")
    tr.add_argument("--metrics", type=Path, nargs="+", required=True)
    tr.add_argument("--output", type=Path, required=True)
    ar = commands.add_parser("architecture", help="Draw the default GPT architecture.")
    ar.add_argument("--output", type=Path, required=True)
    comp = commands.add_parser("comparison", help="Render task results from a completed metrics JSON.")
    comp.add_argument("--metrics", type=Path, required=True)
    comp.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "history":
        histories(args.metrics, args.output)
    elif args.kind == "architecture":
        architecture(args.output)
    else:
        writing_comparison(args.metrics, args.output)


if __name__ == "__main__":
    main()
