# 诗歌盲评（待两名独立评阅者填写）

只把 `reviewer_1.csv`、`reviewer_2.csv` 分别交给两名评阅者。每份有 30 组、共 60 条输出，包含错格、漏词及提前停止的样例；每份的顺序已随机化。请在表中给切题（relevance）、通顺（fluency）、连贯（coherence）、意象与意境（imagery）各填整数 1–5，可在 notes 说明。评分维度的完整锚点见 [清单第 10.4 节](../../GPT_BUILD_CHECKLIST.md#104-人工盲评诗写得怎么样)；输出为空或明确拒绝，各维记 1。

| 维度 | 1 | 3 | 5 |
|---|---|---|---|
| 切题 | 无关或生硬堆词 | 有关联，部分漏用或牵强 | 围绕诗题形成明确主题 |
| 通顺 | 多数句难读 | 大意通顺，有个别不自然用词 | 读得懂，语言凝练 |
| 连贯 | 句间无关、矛盾 | 有共同场景，转折或收尾显突兀 | 四句能读成一首完整诗 |
| 意象 | 抽象套话，没形成画面 | 有场景或情绪，表现普通 | 画面具体，情境自然，有余味 |

作答前可用以下两条**构造示例**校准：题目「江、月」；较通顺的例子「客舫臨江泊，孤燈映短篷。月沉沙岸冷，霜薄候歸鴻。」；只有堆词的例子「月月江江月，江江月月江。月江江月月，江月月江江。」不必把押韵或字面命中重复算入人评分，可按语义判断切题。二人分别填写，不传阅彼此的评分或模型映射。

当前收到 **0/2** 份。程序不会把空表或自动评分计为人工结果。组织者留存单独的 `runs/poetry-blind-v1/private_key.json`，不随评分表发给评阅者；若需重建，用 README 的导出命令和相同的完整基线生成记录。

```bash
python -E scripts/blind_review.py aggregate \
    --key runs/poetry-blind-v1/private_key.json \
    --ratings reviewer_1_completed.csv reviewer_2_completed.csv \
    --output runs/human_review_results.json
```

如只安排一轮评分，优先使用 `expanded/reviewer_1.csv` 与 `expanded/reviewer_2.csv`：它们比较交付的 92M 模型在同一预训练权重的微调前后，仍是冻结的 30 道题和匿名的 60 条输出。14M 基线的上层两份表可以作为附加实验。不要交叉混用两套表。扩容版表单完成后，把文件放在此目录的 `expanded/`，以 `reviewer_1_completed.csv`、`reviewer_2_completed.csv` 命名；汇总时对应的模型映射为本机 `runs/poetry-blind-expanded/private_key.json`。

若需要重建扩容版映射与空表：

```bash
python -E scripts/blind_review.py make \
    --generations report/evidence/retained-expanded-eval-v1/generations.jsonl \
    --models X-92M-TangSong-Pre X-92M-TangSong-2028 \
    --output runs/poetry-blind-expanded-recreated
```

收到两份填好的扩容表后，在仓库根目录运行：

```bash
python -E scripts/blind_review.py aggregate \
    --key runs/poetry-blind-expanded/private_key.json \
    --ratings report/blind_review/expanded/reviewer_1_completed.csv \
              report/blind_review/expanded/reviewer_2_completed.csv \
    --output runs/human_review_expanded.json
```

只把评分表和评分锚点发给两位评阅者。格律抽查是另一项工作；建议在盲评交回后，用[抽查示例](meter_review.md)请熟悉诗律的读者确认。
