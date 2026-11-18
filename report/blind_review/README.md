# 诗歌盲评：组织者说明

**优先评 92M 交付版。** 请将 [评阅者 1 材料](../../dist/orchid-blind-92m/reviewer_1.zip) 和 [评阅者 2 材料](../../dist/orchid-blind-92m/reviewer_2.zip) 分别交给两位评阅者。每个 ZIP 只有一份匿名 CSV 和[评分说明](reviewer_guide.md)。表里有 30 道题、共 60 条输出；两份表的题序和组内显示顺序独立打乱。评阅者评切题、通顺、连贯、意象与意境，四项各 1～5 分。

请勿把本组织者页面、课程包、模型对应关系、生成日志或[格律抽查说明](meter_review.md)一同发给评阅者。回件只需填完的 UTF-8 CSV；请放在 `report/blind_review/expanded/`，分别命名为 `reviewer_1_completed.csv` 和 `reviewer_2_completed.csv`。这两份回件默认不进 Git，也不进入交付 ZIP。组织者负责确认由两人独立完成，这不能由 CSV 自动证明。基线 14M 的两份表保留为可选附加对照；翻阅任何结果前先评完。

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

私有映射如已丢失，汇总时把 `--key` 改为上面重建的路径。收到完整评分后再填入报告，同时保留四维均值、分布、逐题配对差值与至少相差 2 分的条目。当前收到 **0/2** 份真实评分。
