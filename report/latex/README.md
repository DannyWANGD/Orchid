# Orchid 项目报告

正文为 11 pt，中文用 Fandol 宋体，英文用 Georgia，命令用 Courier New 并加圆角底色。训练、冻结评估、评阅、排序补测和时延图表来自项目记录，网页截图来自实际浏览器。项目封面插图由内置 ImageGen 生成，构图为月色下的群山、孤舟和兰花。

## 编译

准备 XeLaTeX、BibTeX、latexmk、Fandol、Georgia、Courier New 和 Noto CJK 字体。在本目录运行以下命令。

```bash
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

封面从可选的 `cover.local.tex` 读取姓名、学号、班级和方向。署名版 LaTeX 源码包内含该文件。Git 源码与模型包用项目名称和日期完成封面，可用于公开复现构建。

## 更新图表与补测

从仓库根目录运行以下命令，替换命令中的模型路径为已导出的权重包目录。

```bash
python -E scripts/evaluate_ranking.py --split challenge --checkpoint dist/orchid-92m-prosody \
  --device cuda --output runs/ranking-replay
python -E scripts/plot_project_report.py
python -E scripts/prepare_final_review.py --output runs/blind-92m-replay
```

图表脚本读取封存证据。新运行应先核对指纹，再按需更新证据和报告。封存排序结果和 AI v2 汇总可直接复查。[新排序证据](../evidence/ranking-ablation-20261121/) · [92M AI 盲评](../evidence/blind-v2-ai-92m-20261121.json) · [评分口径](../blind_review/reviewer_guide_v2.md)
