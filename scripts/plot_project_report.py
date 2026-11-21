"""Render the Orchid project report's vector figures from the archived evidence."""

import argparse
import csv
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.io import ROOT, read_json


EVIDENCE = ROOT / "report/evidence"
INK, MUTED, RULE = "#243730", "#536762", "#DAE4DE"
BLUE, GREEN, GOLD, RED = "#0072B2", "#008461", "#B16A00", "#B55038"
PALE_BLUE, PALE_GREEN, PALE_GOLD = "#E9F3F8", "#EAF5EE", "#FBF3DD"
STYLE = {
    "font.family": ["Georgia", "Noto Sans CJK SC", "DejaVu Sans"],
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": RULE,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.axisbelow": True,
    "axes.unicode_minus": False, "pdf.fonttype": 3, "figure.facecolor": "white",
}


def save(figure, out, name):
    figure.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=.1,
                   metadata={"CreationDate": None, "ModDate": None})
    plt.close(figure)


def label(ax, text, axis="y"):
    ax.set_title(text, loc="left", fontsize=11, pad=9, fontweight="bold")
    ax.grid(axis=axis, color=RULE, linewidth=.7)


def box(ax, x, y, w, h, heading, detail, fill, stroke):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.019,rounding_size=0.09",
                           facecolor=fill, edgecolor=stroke, linewidth=.9)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + .65 * h, heading, ha="center", va="center", fontweight="bold")
    ax.text(x + w / 2, y + .3 * h, detail, ha="center", va="center", color=MUTED, fontsize=8.5)


def arrow(ax, start, end, color=MUTED, **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", color=color,
                                 linewidth=1.1, mutation_scale=11, **kwargs))


def workflow(out):
    fig, ax = plt.subplots(figsize=(8.8, 2.65))
    ax.set(xlim=(0, 12), ylim=(0, 3.3))
    ax.axis("off")
    nodes = [
        ("唐宋语料", "分组划分与字表", PALE_BLUE, BLUE),
        ("从零预训练", "预测下一字符", PALE_GREEN, GREEN),
        ("监督微调", "关键词条件与韵律续训", PALE_GREEN, GREEN),
        ("生成与交付", "CLI · 本地网页", PALE_GOLD, GOLD),
    ]
    for i, (heading, detail, fill, color) in enumerate(nodes):
        x = .2 + i * 3.03
        box(ax, x, 1.9, 2.5, .88, heading, detail, fill, color)
        if i:
            arrow(ax, (x - .53, 2.34), (x, 2.34))
    arrow(ax, (1.45, 1.88), (1.45, 1.05), BLUE)
    arrow(ax, (10.55, 1.88), (10.55, 1.05), GOLD)
    box(ax, .2, .15, 2.5, .85, "冻结唐诗留出集", "宋诗只加入训练", PALE_BLUE, BLUE)
    box(ax, 4.15, .15, 3.7, .85, "验证集选模", "参考 NLL 与固定验证题", PALE_BLUE, BLUE)
    box(ax, 9.3, .15, 2.5, .85, "冻结题评估", "规则 · 阅卷 · 时延", PALE_BLUE, BLUE)
    arrow(ax, (2.73, .57), (4.12, .57), BLUE)
    arrow(ax, (7.88, .57), (9.27, .57), BLUE)
    save(fig, out, "workflow")


def transformer(out):
    fig, (net, mask) = plt.subplots(1, 2, figsize=(8.8, 3.8), width_ratios=(1.45, 1),
                                    layout="constrained")
    net.set(xlim=(0, 7.2), ylim=(0, 5.4))
    net.axis("off")
    net.set_title("A  92M 解码器", loc="left", fontweight="bold")
    items = [
        (4.25, "字符标记 + 位置嵌入", "每位置 768 维", PALE_BLUE, BLUE),
        (2.9, "LayerNorm → 因果多头注意力 → 残差加法", "12 头，每头 64 维", PALE_GREEN, GREEN),
        (1.7, "LayerNorm → GELU 前馈 → 残差加法", "前馈层宽度 3,072", PALE_GREEN, GREEN),
        (.3, "末端 LayerNorm + 词表投影", "8,582 个 logits，投影与词嵌入绑权", PALE_GOLD, GOLD),
    ]
    for i, (y, title, detail, fill, color) in enumerate(items):
        box(net, .25, y, 5.55, .82, title, detail, fill, color)
        if i:
            arrow(net, (3.03, items[i - 1][0] - .02), (3.03, y + .86))
    net.plot([6.03, 6.25, 6.25, 6.03], [3.77, 3.77, 1.63, 1.63], color=GREEN, lw=1.2)
    net.text(6.43, 2.7, "× 12", color=GREEN, va="center")
    mask.set_title("B  因果注意力掩码", loc="left", fontweight="bold")
    mask.imshow(np.tril(np.ones((6, 6))), vmin=0, vmax=1,
                cmap=ListedColormap(["#F0F2EF", GREEN]), interpolation="nearest")
    mask.set(xticks=range(6), yticks=range(6), xticklabels=range(1, 7), yticklabels=range(1, 7),
             xlabel="键位置 j", ylabel="查询位置 i")
    mask.set_xticks(np.arange(-.5, 6), minor=True)
    mask.set_yticks(np.arange(-.5, 6), minor=True)
    mask.grid(which="minor", color="white", linewidth=1)
    mask.tick_params(which="minor", bottom=False, left=False)
    save(fig, out, "transformer")


def supervision(out):
    source = read_json(ROOT / "artifacts/sft_mask_example.json")
    all_positions = source["positions"]
    length = source["prompt_len"]
    columns = [*all_positions[length - 5:length + 3], None, all_positions[-1]]
    fig, ax = plt.subplots(figsize=(8.8, 2.15))
    ax.set(xlim=(-.3, 12.2), ylim=(0, 2.65))
    ax.axis("off")
    ax.text(.05, 2.38, "监督配对样例 · " + "、".join(source["keywords"]), fontweight="bold")
    for i, row in enumerate(columns):
        if row is None:
            ax.text(i + 2, 1.25, "…", ha="center", color=MUTED)
            continue
        mask = row["target"] is None
        fill, edge = (PALE_BLUE, BLUE) if mask else (PALE_GREEN, GREEN)
        box(ax, i + 1.58, .74, .89, 1.22, row["input"], row["target"] or "—", fill, edge)
    ax.text(.05, 1.52, "输入", va="center", color=MUTED)
    ax.text(.05, 1.12, "目标", va="center", color=MUTED)
    ax.text(.05, .18, "蓝色 · 条件标签被屏蔽       绿色 · 正文、标点及结束符计入损失", color=MUTED)
    save(fig, out, "supervision")


def corpus(out):
    tang = read_json(ROOT / "artifacts/data_stats.json")
    broad = read_json(ROOT / "artifacts/tang-song/expansion_stats.json")
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.0), layout="constrained")
    label(left, "A  唐诗清洗", "x")
    left.barh(["原始记录", "通过初筛", "按正文去重"], [tang["raw"], tang["clean"], tang["unique"]],
              color=[BLUE, "#4B98BF", GREEN], height=.48)
    left.invert_yaxis()
    left.bar_label(left.containers[0], labels=[f"{tang[k]:,}" for k in ("raw", "clean", "unique")],
                   padding=4, fontsize=9)
    left.set(xlim=(0, 69000), xlabel="记录数")
    left.ticklabel_format(axis="x", style="plain")
    label(right, "B  语料扩展的训练集", "x")
    train, song = broad["tang_train"], broad["song_kept"]
    right.barh(["唐诗版", "唐宋版"], [train, train], color=BLUE, height=.48, label="唐诗")
    right.barh(["唐宋版"], [song], left=[train], color=GREEN, height=.48, label="加入的宋诗")
    for y, value in enumerate((train, train + song)):
        right.annotate(f"{value:,}", (value, y), xytext=(4, 0), textcoords="offset points",
                       va="center", fontsize=9)
    right.set(xlim=(0, 365000), xlabel="训练诗作数")
    right.invert_yaxis()
    right.legend(loc="lower right", frameon=False, fontsize=8)
    save(fig, out, "corpus")


def points(filename, name):
    with (ROOT / "report/figures" / filename).open() as source:
        return [(int(r["tokens_seen"]), float(r["nll"])) for r in csv.DictReader(source)
                if r["run"] == name and r["kind"] == "validation"]


def training(out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.55), layout="constrained")
    label(left, "A  同唐诗语料与训练预算")
    models = [("pretrain-tang-v1", "14M", BLUE), ("pretrain-tang-42m", "42M", GOLD),
              ("pretrain-tang-92m", "92M", GREEN)]
    for run, name, color in models:
        x, y = zip(*points("scale-length.csv", run))
        left.plot(np.array(x) / 1e6, y, "o-", color=color, markersize=3, linewidth=1.6,
                  label=f"{name} · {y[-1]:.3f}")
    left.set(xlabel="累计目标 token（百万）", ylabel="唐诗验证 NLL")
    left.legend(frameon=False, fontsize=8)
    label(right, "B  固定留出集的唐宋扩容")
    curves = [("pretrain-tang-92m-26", "仅唐诗", BLUE), ("pretrain-tang-song-92m", "唐诗＋宋诗", GREEN)]
    for run, name, color in curves:
        x, y = zip(*points("corpus-expansion.csv", run))
        right.plot(np.array(x) / 1e6, y, "o-", color=color, markersize=3, linewidth=1.6, label=name)
        j = int(np.argmin(y))
        right.scatter(x[j] / 1e6, y[j], s=36, facecolor="white", edgecolor=color, zorder=3)
        right.annotate(f"{y[j]:.3f}", (x[j] / 1e6, y[j]), xytext=(0, -16),
                       textcoords="offset points", ha="center", color=color, fontsize=8.5)
    right.set(xlabel="累计目标 token（百万）", ylabel="唐诗验证 NLL", ylim=(3.35, 9.5))
    right.legend(frameon=False, fontsize=8)
    save(fig, out, "training")


def finetune(out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.3), layout="constrained")
    label(left, "A  正文与结束符的验证损失")
    for seed, color, symbol in ((2026, BLUE, "s"), (2027, GOLD, "^"), (2028, GREEN, "o")):
        x, y = zip(*points("sft-tang-song.csv", f"sft-tang-song-92m-{seed}"))
        left.plot(range(1, len(x)), y[1:], marker=symbol, color=color, lw=1.5, label=str(seed))
    left.set(xlabel="微调轮次", ylabel="验证 NLL", xticks=[1, 2, 3, 4])
    left.legend(frameon=False, fontsize=8, title="训练种子", title_fontsize=8)
    label(right, "B  韵律续训的自由生成")
    study = read_json(EVIDENCE / "prosody-study/validation/metrics.json")["models"]
    cells = (("原权重", "original"), ("普通续训", "random-sft"), ("同韵续训", "rhyme-sft"))
    counts = [study[name]["summary"]["distinct_rhyme"]["count"] for _, name in cells]
    bars = right.bar([label for label, _ in cells], counts, color=[BLUE, GOLD, GREEN], width=.58)
    right.bar_label(bars, labels=[f"{n}/270" for n in counts], padding=3)
    right.set(ylabel="偶句异字同韵数", ylim=(0, 224))
    save(fig, out, "finetune")


def rules(out):
    data = read_json(EVIDENCE / "release-evaluation-20261121.json")["quality"]["modes"]
    fig, ax = plt.subplots(figsize=(8.8, 3.2), layout="constrained")
    label(ax, "同一 92M 权重 · 50 题 × 3 次采样")
    modes = ("raw", "form", "rhyme-only", "rhymed")
    x = np.arange(len(modes))
    for i, (key, name, color) in enumerate((("format_ok", "四句五字", BLUE),
                                           ("distinct_rhyme", "偶句异字同韵", GOLD),
                                           ("keywords_ok", "原词全覆盖", "#6BA3BC"),
                                           ("all_rules", "三项同时达标", GREEN))):
        bars = ax.bar(x + (i - 1.5) * .18, [data[m]["counts"][key] for m in modes],
                      color=color, label=name, width=.16)
        ax.bar_label(bars, fontsize=8, padding=2)
    ax.set(xticks=x, xticklabels=["自由", "仅字数", "字数＋押韵", "默认"], ylim=(0, 177),
           ylabel="达标输出数")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.13), ncols=4, frameon=False, fontsize=8)
    save(fig, out, "rules")


def blind(out):
    reviews = read_json(EVIDENCE / "blind-v2-ai-92m-20261121.json")["ratings_by_reviewer"]
    dimensions = ("relevance", "fluency", "coherence", "imagery")
    fig, ax = plt.subplots(figsize=(8.8, 3.05), layout="constrained")
    label(ax, "v2 同题评分 · 默认押韵模式减去自由生成", "x")
    positions = np.arange(len(dimensions))
    for i, (color, review) in enumerate(zip((BLUE, GREEN), reviews)):
        data = review["differences"]
        values = [data[dim]["mean"] for dim in dimensions]
        y = positions + (i - .5) * .31
        ax.barh(y, values, .27, color=color, label=f"AI 评阅 {review['reviewer']}")
        for dim, v, row_y in zip(dimensions, values, y):
            ax.annotate(f"{v:+.2f}  ({data[dim]['paired']})", (v, row_y),
                        xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=8.5)
    ax.axvline(0, color=MUTED, linewidth=.85)
    ax.set(yticks=positions, yticklabels=["切题", "可读性", "连贯性", "意境"], xlim=(-.55, 2.27),
           xlabel="配对均差（括号内为可比题数）")
    ax.invert_yaxis()
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    save(fig, out, "blind")


def ranking(out):
    base = EVIDENCE / "ranking-ablation-20261121"
    val = read_json(base / "validation/metrics.json")["modes"]
    test = read_json(base / "challenge/metrics.json")["modes"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.15), layout="constrained")
    label(left, "A  同句出现隔位重字")
    for i, (mode, title, color) in enumerate((("original", "原排序", BLUE), ("repeat-aware", "新排序", GREEN))):
        cells = [rows[mode]["counts"]["has_separated_repeat"] for rows in (val, test)]
        bars = left.bar(np.arange(2) + (i - .5) * .28, [100 * r["rate"] for r in cells],
                        width=.25, color=color, label=title)
        left.bar_label(bars, labels=[f"{r['count']}/{r['total']}" for r in cells], fontsize=8.5, padding=3)
    left.set(xticks=[0, 1], xticklabels=["旧验证题", "新题"], ylabel="有隔位重字的输出（%）", ylim=(0, 19.5))
    left.legend(loc="upper right", frameon=False, fontsize=8)
    label(right, "B  新题生成延迟", "x")
    for i, (mode, name, color) in enumerate((("raw", "自由", MUTED), ("one-candidate", "单候选", GOLD),
                                             ("original", "四候选原排序", BLUE),
                                             ("repeat-aware", "四候选新排序", GREEN))):
        m = test[mode]
        mid, tail = m["median_seconds"], m["p95_seconds"]
        right.hlines(i, mid, tail, color=color, linewidth=1.4)
        right.scatter([mid, tail], [i, i], facecolors=[color, "white"], edgecolors=color, s=31, zorder=3)
        right.annotate(f"{mid:.3f}", (mid, i), xytext=(0, 7), textcoords="offset points",
                       ha="center", fontsize=8)
    right.set(yticks=range(4), yticklabels=["自由", "单候选", "四候选原排序", "四候选新排序"],
              xlim=(0, .39), ylim=(-.45, 3.5), xlabel="秒（实心 p50 · 空心 p95）")
    right.invert_yaxis()
    save(fig, out, "ranking")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "report/latex/figures")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        for plot in (workflow, transformer, corpus, supervision, training, finetune, rules, blind, ranking):
            plot(args.output)


if __name__ == "__main__":
    main()
