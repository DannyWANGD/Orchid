"""Render the Orchid project report's vector figures from the archived evidence."""

import argparse
import csv
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from orchid.io import ROOT, read_json


EVIDENCE = ROOT / "report/evidence"
GOLD, PEACH, CORAL, ROSE, PLUM = "#F9DB64", "#EFC8B9", "#ED867C", "#DD808A", "#794E83"
INK, MUTED, RULE = "#382741", "#635165", "#E9DEEB"
PALE_PLUM, PALE_PEACH, PALE_GOLD = "#F4EDF5", "#FBF0EA", "#FFF7D6"
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
        ("唐宋语料", "分组划分与字表", PALE_PEACH, PEACH),
        ("从零预训练", "预测下一字符", PALE_GOLD, GOLD),
        ("监督微调", "关键词条件与韵律续训", PALE_PEACH, CORAL),
        ("生成与交付", "CLI · 本地网页", PALE_PLUM, PLUM),
    ]
    for i, (heading, detail, fill, color) in enumerate(nodes):
        x = .2 + i * 3.03
        box(ax, x, 1.9, 2.5, .88, heading, detail, fill, color)
        if i:
            arrow(ax, (x - .53, 2.34), (x, 2.34))
    arrow(ax, (1.45, 1.88), (1.45, 1.05), PLUM)
    arrow(ax, (10.55, 1.88), (10.55, 1.05), PLUM)
    box(ax, .2, .15, 2.5, .85, "冻结唐诗留出集", "宋诗只加入训练", PALE_PLUM, ROSE)
    box(ax, 4.15, .15, 3.7, .85, "验证集选模", "参考 NLL 与固定验证题", PALE_PLUM, ROSE)
    box(ax, 9.3, .15, 2.5, .85, "冻结题评估", "规则 · 评阅 · 时延", PALE_PLUM, PLUM)
    arrow(ax, (2.73, .57), (4.12, .57), PLUM)
    arrow(ax, (7.88, .57), (9.27, .57), PLUM)
    save(fig, out, "workflow")


def transformer(out):
    fig, (net, mask) = plt.subplots(1, 2, figsize=(8.8, 3.8), width_ratios=(1.45, 1),
                                    layout="constrained")
    net.set(xlim=(0, 7.2), ylim=(0, 5.4))
    net.axis("off")
    net.set_title("A  92M 解码器", loc="left", fontweight="bold")
    items = [
        (4.25, "字符标记 + 位置嵌入", "每位置 768 维", PALE_PEACH, PEACH),
        (2.9, "LayerNorm → 因果多头注意力 → 残差加法", "12 头，每头 64 维", PALE_PLUM, ROSE),
        (1.7, "LayerNorm → GELU 前馈 → 残差加法", "前馈层宽度 3,072", PALE_PLUM, PLUM),
        (.3, "末端 LayerNorm + 词表投影", "8,582 个 logits，投影与词嵌入绑权", PALE_GOLD, CORAL),
    ]
    for i, (y, title, detail, fill, color) in enumerate(items):
        box(net, .25, y, 5.55, .82, title, detail, fill, color)
        if i:
            arrow(net, (3.03, items[i - 1][0] - .02), (3.03, y + .86))
    net.plot([6.03, 6.25, 6.25, 6.03], [3.77, 3.77, 1.63, 1.63], color=PLUM, lw=1.2)
    net.text(6.43, 2.7, "× 12", color=PLUM, va="center")
    mask.set_title("B  因果注意力掩码", loc="left", fontweight="bold")
    mask.imshow(np.tril(np.ones((6, 6))), vmin=0, vmax=1,
                cmap=ListedColormap([PALE_PEACH, PLUM]), interpolation="nearest")
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
        fill, edge = (PALE_PEACH, PEACH) if mask else (PALE_PLUM, PLUM)
        box(ax, i + 1.58, .74, .89, 1.22, row["input"], row["target"] or "—", fill, edge)
    ax.text(.05, 1.52, "输入", va="center", color=MUTED)
    ax.text(.05, 1.12, "目标", va="center", color=MUTED)
    ax.text(.05, .18, "浅桃色 · 条件标签被屏蔽       淡紫色 · 正文、标点及结束符计入损失", color=MUTED)
    save(fig, out, "supervision")


def corpus(out):
    tang = read_json(ROOT / "artifacts/data_stats.json")
    broad = read_json(ROOT / "artifacts/tang-song/expansion_stats.json")
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.0), layout="constrained")
    label(left, "A  唐诗清洗", "x")
    left.barh(["原始记录", "通过初筛", "按正文去重"], [tang["raw"], tang["clean"], tang["unique"]],
              color=[PEACH, CORAL, PLUM], height=.48)
    left.invert_yaxis()
    left.bar_label(left.containers[0], labels=[f"{tang[k]:,}" for k in ("raw", "clean", "unique")],
                   padding=4, fontsize=9)
    left.set(xlim=(0, 69000), xlabel="记录数")
    left.ticklabel_format(axis="x", style="plain")
    label(right, "B  语料扩展的训练集", "x")
    train, song = broad["tang_train"], broad["song_kept"]
    right.barh(["唐诗版", "唐宋版"], [train, train], color=ROSE, height=.48, label="唐诗")
    right.barh(["唐宋版"], [song], left=[train], color=PLUM, height=.48, label="加入的宋诗")
    for y, value in enumerate((train, train + song)):
        right.annotate(f"{value:,}", (value, y), xytext=(4, 0), textcoords="offset points",
                       va="center", fontsize=9)
    right.set(xlim=(0, 365000), xlabel="训练诗作数", xticks=[0, 100000, 200000, 300000])
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
    models = [("pretrain-tang-v1", "14M", ROSE, "s--"), ("pretrain-tang-42m", "42M", CORAL, "^-."),
              ("pretrain-tang-92m", "92M", PLUM, "o-")]
    for run, name, color, marker_style in models:
        x, y = zip(*points("scale-length.csv", run))
        left.plot(np.array(x) / 1e6, y, marker_style, color=color, markeredgecolor=PLUM,
                  markeredgewidth=.5, markersize=3, linewidth=1.6,
                  label=f"{name} · {y[-1]:.3f}")
    left.set(xlabel="累计目标 token（百万）", ylabel="唐诗验证 NLL")
    left.legend(frameon=False, fontsize=8)
    label(right, "B  固定留出集的唐宋扩容")
    curves = [("pretrain-tang-92m-26", "仅唐诗", CORAL, "s--"),
              ("pretrain-tang-song-92m", "唐诗＋宋诗", PLUM, "o-")]
    for run, name, color, marker_style in curves:
        x, y = zip(*points("corpus-expansion.csv", run))
        right.plot(np.array(x) / 1e6, y, marker_style, color=color, markeredgecolor=PLUM,
                   markeredgewidth=.5, markersize=3, linewidth=1.6, label=name)
        j = int(np.argmin(y))
        right.scatter(x[j] / 1e6, y[j], s=36, facecolor="white", edgecolor=color, zorder=3)
        right.annotate(f"{y[j]:.3f}", (x[j] / 1e6, y[j]), xytext=(0, -16),
                       textcoords="offset points", ha="center", color=INK, fontsize=8.5)
    right.set(xlabel="累计目标 token（百万）", ylabel="唐诗验证 NLL", ylim=(3.35, 9.5))
    right.legend(frameon=False, fontsize=8)
    save(fig, out, "training")


def finetune(out):
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.3), layout="constrained")
    label(left, "A  正文与结束符的验证损失")
    for seed, color, symbol in ((2026, ROSE, "s--"), (2027, CORAL, "^-."),
                                (2028, PLUM, "o-")):
        x, y = zip(*points("sft-tang-song.csv", f"sft-tang-song-92m-{seed}"))
        left.plot(range(1, len(x)), y[1:], symbol, color=color, lw=1.5, label=str(seed),
                  markeredgecolor=PLUM, markeredgewidth=.5)
    left.set(xlabel="微调轮次", ylabel="验证 NLL", xticks=[1, 2, 3, 4])
    left.legend(frameon=False, fontsize=8, title="训练种子", title_fontsize=8)
    label(right, "B  韵律续训的自由生成")
    study = read_json(EVIDENCE / "prosody-study/validation/metrics.json")["models"]
    cells = (("原权重", "original"), ("普通续训", "random-sft"), ("同韵续训", "rhyme-sft"))
    counts = [study[name]["summary"]["distinct_rhyme"]["count"] for _, name in cells]
    bars = right.bar([name for name, _ in cells], counts, color=[PEACH, CORAL, PLUM], width=.58)
    right.bar_label(bars, labels=[f"{n}/270" for n in counts], padding=3)
    right.set(ylabel="偶句异字同韵数", ylim=(0, 224))
    save(fig, out, "finetune")


def rules(out):
    data = read_json(EVIDENCE / "release-evaluation-20261121.json")["quality"]["modes"]
    fig, ax = plt.subplots(figsize=(8.8, 3.2), layout="constrained")
    label(ax, "同一 92M 权重 · 50 题 × 3 次采样")
    modes = ("raw", "form", "rhyme-only", "rhymed")
    x = np.arange(len(modes))
    for i, (key, name, color) in enumerate((("format_ok", "四句五字", PEACH),
                                           ("distinct_rhyme", "偶句异字同韵", GOLD),
                                           ("keywords_ok", "原词全覆盖", CORAL),
                                           ("all_rules", "三项同时达标", PLUM))):
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
    for i, (color, review) in enumerate(zip((ROSE, PLUM), reviews)):
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


def decoding_slots(out):
    pairs = read_json(EVIDENCE / "release-evaluation-20261121.json")["qualitative_examples"]["pairs"]
    example = next(row for row in pairs if row["prompt_id"] == "M017")
    lines = re.findall(r"([^，。]+)([，。])", example["rhymed"])
    if len(lines) != 4 or any(len(line) != 5 for line, _ in lines):
        raise ValueError("Expected four five-character lines for M017")

    fig, ax = plt.subplots(figsize=(8.8, 3.4))
    ax.set(xlim=(0, 12), ylim=(0, 5.35))
    ax.axis("off")
    ax.text(.1, 5.07, "偶句韵脚怎样进入生成过程", fontsize=11, fontweight="bold", va="center")
    ax.text(11.85, 5.07, "「霜、孤舟」冻结样例", color=MUTED, fontsize=9, ha="right", va="center")
    ax.text(.12, 4.47, "句次", color=MUTED, va="center")
    for i, title in enumerate(("第 1 字", "第 2 字", "第 3 字", "第 4 字", "第 5 字", "标点")):
        ax.text(1.41 + i * .91, 4.47, title, ha="center", va="center", color=MUTED, fontsize=8.7)
    for i, (line, punctuation) in enumerate(lines):
        y = 3.45 - i * .91
        ax.text(.12, y + .36, f"第 {i + 1} 句", color=MUTED, va="center")
        for j, character in enumerate(line + punctuation):
            is_rhyme = i in (1, 3) and j == 4
            fill, stroke, ink = (
                (PLUM, PLUM, "white") if is_rhyme else
                (PALE_GOLD, GOLD, INK) if j == 5 else (PALE_PEACH, PEACH, INK)
            )
            x = 1.04 + j * .91
            patch = FancyBboxPatch((x, y), .74, .70, boxstyle="round,pad=.016,rounding_size=.08",
                                   facecolor=fill, edgecolor=stroke, linewidth=.75)
            ax.add_patch(patch)
            ax.text(x + .37, y + .36, character, color=ink, ha="center", va="center", fontsize=14)
    for y, head, body in (
        (2.39, "第 11 步选「尋」", "从可配对的单一平声韵部字中取样"),
        (.57, "第 23 步选「林」", "取「尋」的同部搭档，并排除「尋」本字"),
    ):
        arrow(ax, (6.5, y + .39), (7.03, y + .39), PLUM)
        box(ax, 7.05, y, 4.65, .79, head, body, PALE_PLUM, PLUM)
    ax.text(.13, .16, "每句五字加标点。第 25 步输出结束符。", color=MUTED, fontsize=8.7)
    save(fig, out, "decoding-slots")


def rhyme_strictness(out):
    quality = read_json(EVIDENCE / "release-evaluation-20261121.json")["quality"]["modes"]["rhymed"]
    canonical = read_json(EVIDENCE / "meter-prosody.json")["modes"]["rhymed"]["tone_rhyme"]
    total = sum(canonical.values())
    fig, ax = plt.subplots(figsize=(8.8, 2.3), layout="constrained")
    label(ax, f"同批 {total} 首默认输出 · 偶句用韵与完整常格", "x")
    count = quality["counts"]["distinct_rhyme"]
    ax.barh([0], [count], color=PLUM, height=.48)
    ax.text(count / 2, 0, f"{count}/{total} 通过", color="white", fontweight="bold", va="center", ha="center")
    offset = 0
    for key, color in (("pass", PLUM), ("unknown", GOLD), ("fail", PEACH)):
        number = canonical[key]
        ax.barh([1], [number], left=[offset], color=color, height=.48)
        offset += number
    ax.text((total + offset - canonical["fail"]) / 2, 1,
            f"{canonical['fail']} 未通过", color=INK, va="center", ha="center")
    ax.set(yticks=[0, 1], yticklabels=["偶句异字同平韵", "逐字平仄与常格用韵"],
           xlim=(0, total), ylim=(1.57, -.55), xticks=[0, 30, 60, 90, 120, total], xlabel="诗作数")
    handles = [
        Line2D([0], [0], marker="s", linestyle="none", color=color, markersize=7,
               label=f"{title} {canonical[key]}")
        for key, color, title in (("pass", PLUM, "通过"), ("unknown", GOLD, "待判"), ("fail", PEACH, "未通过"))
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.3),
              ncols=3, frameon=False, fontsize=8)
    save(fig, out, "rhyme-strictness")


def latency(out):
    timings = read_json(EVIDENCE / "release-evaluation-20261121.json")["benchmark"]
    fig, ax = plt.subplots(figsize=(8.8, 2.9), layout="constrained")
    label(ax, "CPU 与 GPU 的常驻生成时延", "x")
    cases = (("cuda", "raw", "GPU · 自由"), ("cuda", "rhymed", "GPU · 默认"),
             ("cpu", "raw", "CPU · 自由"), ("cpu", "rhymed", "CPU · 默认"))
    for y, (device, mode, _) in enumerate(cases):
        benchmark = timings[mode]["benchmarks"][device]
        median, p95 = benchmark["median_seconds"], benchmark["p95_seconds"]
        color = CORAL if mode == "raw" else PLUM
        ax.hlines(y, median, p95, color=color, linewidth=2)
        ax.scatter(median, y, s=35, facecolor=color, edgecolor=PLUM, zorder=3)
        ax.scatter(p95, y, s=39, facecolor="white", edgecolor=PLUM, marker="s", zorder=3)
        ax.annotate(f"{median:.3f} / {p95:.3f}", (p95, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8.5)
    ax.axhline(1.5, color=RULE, lw=.8)
    ax.set(yticks=range(len(cases)), yticklabels=[name for _, _, name in cases], ylim=(3.5, -.51),
           xscale="log", xlim=(.034, 5.5), xlabel="生成耗时（秒，对数轴） · 实心 p50／空心 p95")
    ticks = [.05, .1, .2, .5, 1, 2, 5]
    ax.set_xticks(ticks, [f"{tick:g}" for tick in ticks])
    save(fig, out, "latency")


def ranking(out):
    base = EVIDENCE / "ranking-ablation-20261121"
    val = read_json(base / "validation/metrics.json")["modes"]
    test = read_json(base / "challenge/metrics.json")["modes"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.8, 3.15), layout="constrained")
    label(left, "A  同句出现隔位重字")
    for i, (mode, title, color) in enumerate((("original", "原排序", ROSE), ("repeat-aware", "新排序", PLUM))):
        cells = [rows[mode]["counts"]["has_separated_repeat"] for rows in (val, test)]
        bars = left.bar(np.arange(2) + (i - .5) * .28, [100 * r["rate"] for r in cells],
                        width=.25, color=color, label=title)
        left.bar_label(bars, labels=[f"{r['count']}/{r['total']}" for r in cells], fontsize=8.5, padding=3)
    left.set(xticks=[0, 1], xticklabels=["旧验证题", "新题"], ylabel="有隔位重字的输出（%）", ylim=(0, 19.5))
    left.legend(loc="upper right", frameon=False, fontsize=8)
    label(right, "B  新题生成延迟", "x")
    for i, (mode, _, color) in enumerate((("raw", "自由", MUTED), ("one-candidate", "单候选", GOLD),
                                          ("original", "四候选原排序", ROSE),
                                          ("repeat-aware", "四候选新排序", PLUM))):
        m = test[mode]
        mid, tail = m["median_seconds"], m["p95_seconds"]
        right.hlines(i, mid, tail, color=color, linewidth=1.4)
        right.scatter([mid, tail], [i, i], facecolors=[color, "white"], edgecolors=PLUM, s=31, zorder=3)
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
        for plot in (workflow, transformer, corpus, supervision, training, finetune, rules,
                     blind, decoding_slots, rhyme_strictness, latency, ranking):
            plot(args.output)


if __name__ == "__main__":
    main()
