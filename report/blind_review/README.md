# 诗歌盲评：组织者说明

**14M 已收到 1/2 份有效回件。** 第 2 份回件的 60 条匿名 ID 和 240 个评分均通过校验，本机备份为 `reviewer_2_completed.csv`。交给同版另一位评阅者的是 [14M 评阅者 1 独立材料](../../dist/orchid-blind-14m/reviewer_1.zip)。

**原 92M 收到 0/2 份回件。** [92M 评阅者 1 材料](../../dist/orchid-blind-92m/reviewer_1.zip) 和 [92M 评阅者 2 材料](../../dist/orchid-blind-92m/reviewer_2.zip) 各有一份匿名 CSV 和评分说明。每轮次有 30 道题、共 60 条输出，两份表的题序和组内显示顺序独立打乱。评阅者评切题、通顺、连贯、意象与意境，四项各 1～5 分。[评分说明](reviewer_guide.md)

评阅者仅需各自的 ZIP，并在查看结果前独立完成。14M 回件放在本目录，原 92M 回件放在 `expanded/`，均用 `reviewer_1_completed.csv` 和 `reviewer_2_completed.csv` 命名。回件保存在本机，课程包使用空白表。按同一版本汇总，组织者确认两位独立完成。新押韵版已完成自动对照，已有评分对应旧轮次的原诗。[押韵改进实测](../prosody_optimization.md)

## 重建表与单独分发包

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

## 回收与汇总

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
