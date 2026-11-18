# 诗歌盲评：组织者说明

## 已完成的 14M v1 主对照

提交者确认沿用原第 2 份真人评分。第 1 份由独立 AI 子 agent 按同一 v1 量表完成，两份对应相同的 30 道匿名原输出配对，各有 240 个通过校验的评分。主报告采用真人评分，并报告 AI 参照及评分差异。[v1 同量表结果](../evaluation_baseline.md#一位真人与一位-ai-的盲评结果)

## v2 量表复核

同一批 14M 封存诗题和原诗已重排匿名编号与呈现顺序。v2 的切题看语义，可读性看表达能否理解，连贯性只评可解析的四句，意境只在可读性和连贯性各达到 3 分时评。小于门槛的项目为 `NA`。第 1 份 AI 回件的 60 条已校验，原件保存在 `runs/poetry-blind-v2/reviewer_1_ai_completed.csv`。v2 用于复核这些规则的适用范围。[v2 评分说明](reviewer_guide_v2.md) · [AI 汇总](../evidence/blind-v2-ai-14m.json) · [评阅者 1 材料](../../dist/orchid-blind-14m-v2/reviewer_1.zip) · [评阅者 2 材料](../../dist/orchid-blind-14m-v2/reviewer_2.zip)

首次重建 v2 或把新回件存入 `runs/poetry-blind-v2/` 后汇总，分别使用以下命令。重建时选择新目录保留原表。仍需保存每个来源各自的覆盖率和结果。

```bash
python -E scripts/blind_review.py make \
    --generations report/evidence/fixed-eval-v1/generations.jsonl \
    --rubric-version v2 --output runs/poetry-blind-v2
python -E scripts/blind_review.py distribute \
    --generations report/evidence/fixed-eval-v1/generations.jsonl \
    --key runs/poetry-blind-v2/private_key.json \
    --templates runs/poetry-blind-v2/reviewer_1.csv runs/poetry-blind-v2/reviewer_2.csv \
    --output dist/orchid-blind-14m-v2
python -E scripts/blind_review.py aggregate \
    --generations report/evidence/fixed-eval-v1/generations.jsonl \
    --key runs/poetry-blind-v2/private_key.json \
    --templates runs/poetry-blind-v2/reviewer_1.csv runs/poetry-blind-v2/reviewer_2.csv \
    --ratings runs/poetry-blind-v2/reviewer_1_completed.csv runs/poetry-blind-v2/reviewer_2_completed.csv \
    --output runs/review_v2_14m.json
```

汇总中的 `rated / total` 是数值评分数和全部输出数，`paired / total_pairs` 是一对两首均有数值的题数和全部题数。若汇总两位真人，配对差仅采用两位在该题均可比较的记录。没有配对的维度显示 `null`。如回件含 AI，按顺序声明 `--reviewer-kinds`，并保留各来源的结果。

只收到一份时可用 `summarize` 核验。它使用相同的封存文件、私有映射和模板，增加 `--rating`、`--reviewer 1` 或 `2`、`--reviewer-kind human` 或 `ai`。输出只有分布、适用数与配对差的汇总。

格律抽查使用单独的[五言格律抽查表](meter_review.md)。表中提供八首原诗。同一位读者若也参与质量盲评，请先交匿名评分 CSV，再做格律抽查。格律回件保存为本目录的 `meter_review_completed.md`。

## v1 材料与回件位置

14M 真人备份为 `reviewer_2_completed.csv`。[v1 量表](reviewer_guide.md)

**原 92M 的 v1 材料尚无回件。** 两位各有一份匿名 CSV 和评分说明。每轮次有 30 道题、共 60 条输出，两份表的题序和组内显示顺序独立打乱。[原 92M 评阅者 1 材料](../../dist/orchid-blind-92m/reviewer_1.zip) · [原 92M 评阅者 2 材料](../../dist/orchid-blind-92m/reviewer_2.zip)

评阅者仅需各自的 ZIP，并在查看结果前独立完成。14M 回件放在本目录，原 92M 回件放在 `expanded/`，均用 `reviewer_1_completed.csv` 和 `reviewer_2_completed.csv` 命名。回件保存在本机，课程包使用空白表。按同一版本汇总，组织者确认两位独立完成。新押韵版已完成自动对照，已有评分对应旧轮次的原诗。[押韵改进实测](../prosody_optimization.md)

### 重建 v1 表与单独分发包

以下命令从封存的 92M 原始输出重建本地私有映射与表。使用新目录，避免覆盖已经存在的材料；现有的映射若仍在，可以直接跳到汇总。

```bash
python -E scripts/blind_review.py make \
    --generations report/evidence/retained-expanded-eval-v1/generations.jsonl \
    --models X-92M-TangSong-Pre X-92M-TangSong-2028 \
    --output runs/poetry-blind-92m-recreated

python -E scripts/blind_review.py distribute \
    --generations report/evidence/retained-expanded-eval-v1/generations.jsonl \
    --key runs/poetry-blind-92m-recreated/private_key.json \
    --templates report/blind_review/expanded/reviewer_1.csv \
                report/blind_review/expanded/reviewer_2.csv \
    --output dist/orchid-blind-92m-recreated
```

### 回收与汇总 v1

材料是空表时不产生评分结果。汇总会核对封存生成文件、诗题清单、私有映射、原始表及回件，确保评分对象、题目和原始文本一致。有缺项、无效分数或被改写的输出时先人工查明，保留两名评阅者各自的原始分和说明。

```bash
python -E scripts/blind_review.py aggregate \
    --generations report/evidence/retained-expanded-eval-v1/generations.jsonl \
    --key runs/poetry-blind-expanded/private_key.json \
    --templates report/blind_review/expanded/reviewer_1.csv \
                report/blind_review/expanded/reviewer_2.csv \
    --ratings report/blind_review/expanded/reviewer_1_completed.csv \
              report/blind_review/expanded/reviewer_2_completed.csv \
    --output runs/human_review_expanded.json
```

汇总 14M 时，生成文件用 `report/evidence/fixed-eval-v1/generations.jsonl`，私有映射与两份空白原件用 `runs/poetry-blind-v1/`，回件用本目录的两个 `*_completed.csv`。私有映射如需重建，使用该生成文件及模型 `R1 R2-2027`。完整汇总保留四维均值、分布、逐题配对差值与至少相差 2 分的条目。

## 14M 真人与 AI 评阅对照

第 1 份匿名材料已由独立上下文的 AI 子 agent 按同一量表填写，回件另存为 `reviewer_1_ai_completed.csv`。汇总时显式声明来源，分别得到各评阅者的四维均值、分布和逐题配对差值。原始分数、逐首备注和私有映射留在本机。[评分结果](../evaluation_baseline.md#一位真人与一位-ai-的盲评结果)

```bash
python -E scripts/blind_review.py aggregate \
    --generations report/evidence/fixed-eval-v1/generations.jsonl \
    --key runs/poetry-blind-v1/private_key.json \
    --templates runs/poetry-blind-v1/reviewer_1.csv runs/poetry-blind-v1/reviewer_2.csv \
    --ratings report/blind_review/reviewer_1_ai_completed.csv \
              report/blind_review/reviewer_2_completed.csv \
    --reviewer-kinds ai human --output runs/review_human_ai_14m.json
```
