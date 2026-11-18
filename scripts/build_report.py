"""Build the Chinese course PDF from its Markdown and the original vector figures."""

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from urllib.parse import quote

from markdown_it import MarkdownIt


PREAMBLE = r"""
\documentclass[UTF8,a4paper,11pt,fontset=none]{ctexart}
\setCJKmainfont{Noto Serif CJK SC}
\setCJKsansfont{Noto Sans CJK SC}
\setCJKmonofont{Noto Sans CJK SC}
\usepackage[a4paper,hmargin=24mm,vmargin=21mm]{geometry}
\usepackage{graphicx,booktabs,tabularx,hyperref}
\usepackage{xcolor}
\hypersetup{colorlinks=true,linkcolor=blue!42!black,urlcolor=blue!50!black}
\setlength{\parindent}{0pt}
\setlength{\parskip}{7pt plus 2pt}
\setlength{\floatsep}{9pt}
\setlength{\intextsep}{9pt}
\definecolor{mutedink}{HTML}{334C60}
\newcommand{\reportcode}[1]{{\small\ttfamily\color{mutedink}#1}}
\begin{document}
"""

ESCAPES = {
    "\\": r"\textbackslash{}", "&":r"\&", "%":r"\%", "$":r"\$", "#":r"\#",
    "_":r"\_", "{":r"\{", "}":r"\}", "~":r"\textasciitilde{}", "^":r"\textasciicircum{}",
    "₀":r"\ensuremath{_0}", "₁":r"\ensuremath{_1}", "ᵢ":r"\ensuremath{_i}",
    "₊":r"\ensuremath{_+}", "ₜ":r"\ensuremath{_t}", "ⱼ":r"\ensuremath{_j}",
    "ₕ":r"\ensuremath{_h}", "₂":r"\ensuremath{_2}", "√":r"\ensuremath{\sqrt{\vphantom{d}}}",
    "≤":r"\ensuremath{\le}", "→":r"\ensuremath{\rightarrow}",
    "β":r"\ensuremath{\beta}", "ε":r"\ensuremath{\epsilon}",
}


def tex(text):
    return "".join(ESCAPES.get(c,c) for c in text)


def inline(tokens):
    interior=[]
    for item in tokens or []:
        t=item.type
        if t=="text":
            interior.append(tex(item.content))
        elif t in ("softbreak","hardbreak"):
            interior.append(" " if t=="softbreak" else r"\par ")
        elif t=="code_inline":
            interior.append(r"\reportcode{" + tex(item.content) + "}")
        elif t in ("strong_open","em_open"):
            interior.append(r"\textbf{" if t=="strong_open" else r"\emph{")
        elif t in ("strong_close","em_close","link_close"):
            interior.append("}")
        elif t=="link_open":
            href=item.attrGet("href")
            uri=quote(href,safe=":/?=&#._-%")
            interior.append(r"\href{"+uri.replace("%",r"\%").replace("#",r"\#")+"}{")
        else:
            raise ValueError(f"Unsupported inline syntax in report: {t}")
    return "".join(interior)


def render(text):
    out=[PREAMBLE]
    parser=MarkdownIt("commonmark").enable("table")
    tokens=parser.parse(text)
    i=0
    while i<len(tokens):
        token=tokens[i]
        if token.type=="heading_open":
            label=tokens[i+1].content
            if token.tag=="h1":
                out.append(r"\begin{center}{\LARGE\bfseries " + tex(label) + r"}\end{center}")
            else:
                out.append(r"\section*{" + tex(label) + "}")
            i+=3
        elif token.type=="paragraph_open":
            children=tokens[i+1].children
            if len(children)==1 and children[0].type=="image":
                figure=children[0]
                path=Path(figure.attrGet("src")).with_suffix(".pdf")
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Use a relative path in figures: {path}")
                out.extend([r"\begin{center}",r"\includegraphics[width=0.96\linewidth]{"+str(path)+"}",
                            r"\par{\small "+tex(figure.content)+"}",r"\end{center}"])
            elif children:
                out.append(inline(children)+"\n")
            i+=3
        elif token.type=="table_open":
            rows=[]
            i+=1
            while tokens[i].type!="table_close":
                x=tokens[i]
                if x.type=="tr_open":
                    rows.append([])
                if x.type=="inline":
                    rows[-1].append(inline(x.children))
                i+=1
            if any(len(r)!=len(rows[0]) for r in rows):
                raise ValueError("Ragged Markdown table")
            cols = len(rows[0])
            out.append(r"\begin{center}\small\begin{tabularx}{\linewidth}{l"+(">{\raggedright\arraybackslash}X"* (cols-1))+"}")
            out.append(r"\toprule")
            out.append(" & ".join(r"\textbf{"+x+"}" for x in rows[0])+r"\\ \midrule")
            out.extend(" & ".join(row)+r"\\" for row in rows[1:])
            out.append(r"\bottomrule\end{tabularx}\end{center}")
            i+=1
        elif token.type in ("bullet_list_open","bullet_list_close","list_item_open","list_item_close","inline",
                            "blockquote_open","blockquote_close"):
            if token.type=="inline":
                out.append(inline(token.children))
            else:
                pairs={"bullet_list_open":r"\begin{itemize}", "bullet_list_close":r"\end{itemize}",
                       "list_item_open":r"\item ", "list_item_close":"",
                       "blockquote_open":r"\begin{quote}", "blockquote_close":r"\end{quote}"}
                out.append(pairs[token.type])
            i+=1
        elif token.type=="fence":
            out.extend([r"\begin{quote}\ttfamily\small", *[tex(c)+r"\par" for c in token.content.splitlines()],
                        r"\end{quote}"])
            i+=1
        else:
            raise ValueError(f"Unsupported block syntax in report: {token.type}")
    out.append(r"\end{document}")
    return "\n".join(out)+"\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=Path("report/experiment_report.md"))
    p.add_argument("--output", type=Path, default=Path("report/Orchid课程报告.pdf"))
    a=p.parse_args()
    source=a.source.resolve()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix="orchid-pdf-") as tmp:
        dest=Path(tmp)
        (dest/"orchid.tex").write_text(render(source.read_text(encoding="utf-8")), encoding="utf-8")
        run=subprocess.run(
            ["xelatex","-halt-on-error","-interaction=nonstopmode","-output-directory",tmp,str(dest/"orchid.tex")],
            text=True,capture_output=True,cwd=source.parent,
        )
        if run.returncode:
            raise SystemExit(run.stdout[-6500:] + run.stderr[-2000:])
        a.output.write_bytes((dest/"orchid.pdf").read_bytes())
        log=(dest/"orchid.log").read_text(encoding="utf-8",errors="replace")
        issues=[line for line in log.splitlines() if line.startswith(("Overfull","Missing character:"))]
        for issue in issues:
            print(issue)
        print(f"PDF: {a.output}")


if __name__ == "__main__":
    main()
