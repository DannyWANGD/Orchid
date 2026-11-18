# 五言四句与押韵改进实测

## 从盲评发现的问题

第 2 位评阅者交回的是 14M 原始解码轮次。60 条匿名 ID、原诗与题目均匹配，四维评分的 240 个单元格均通过校验。早期预训练模型的 30 条输出无一符合五言四句。监督微调后的 30 条中，29 条符合五言四句，偶句韵脚有 27 条不合、2 条待判。两组均无可确认的异字同韵。基线 14M 已收到一份有效回件，同版的另一份评分仍待回收。[盲评组织者说明](blind_review/README.md)

## 从训练改善自由生成

原唐宋 92M 监督集有 12,394 条五言四句，5,472 条按当前口径可确认偶句异字同韵，3,514 条不合，3,400 条待判，另有 8 条重复韵脚字。普通续训对照抽取 4,096 条，其中 1,809 条可靠同韵。韵律组抽取 4,096 条可靠同韵样本，配平一、二、三词样本数（2,028、1,342、726）。两组从原 92M 的同一最佳 SFT 权重重新初始化优化器，在 RTX 4080 SUPER 上均以 BF16 训练 6 轮、384 步、614,400 个目标 token，按固定步数比较。每组峰值 PyTorch 分配显存约 2.19 GiB。[方案与语料口径](../docs/prosody_protocol.md) · [普通续训日志](evidence/prosody-study/random-training.json) · [韵律续训日志](evidence/prosody-study/rhyme-training.json)

| 92M 权重，均自由生成 | 五言四句 | 结构和用词 | 偶句异字同韵 | 三项皆达标 |
|---|---:|---:|---:|---:|
| 原交付 | 269/270 | 87/270 | 117/270 | 42/270 |
| 普通样本续训 | 266/270 | 103/270 | 119/270 | 49/270 |
| 同韵样本续训 | 268/270 | 109/270 | **188/270** | **80/270** |

验证取 90 道原封存验证题，每题 3 个预设种子。原版与普通续训分别确认 43.3% 和 44.1% 的偶句异字同韵，韵律续训确认 69.6%。三组均无训练集整首命中。韵律续训满足预定条件，采用第 6 轮权重。其第 1 轮验证 NLL 为 3.6390，第 6 轮为 3.7103，反映了普通正文损失和自由押韵目标的区别。当前只有一组配对训练种子。[验证指标](evidence/prosody-study/validation/metrics.json) · [逐条生成](evidence/prosody-study/validation/generations.jsonl)

## 从解码保证交付形式

模型逐字生成，每句五个字，标点后再开下一句，第四句后发送结束符。第二句末在韵部唯一且有搭档的平声字中采样，第四句末从同部的其他字中采样。默认生成四个候选，尝试不同的关键词落点，按模型原始对数概率和重复惩罚选出一首。用韵取固定的平水韵表，第 1、3 句平仄、首句入韵和拗救需要另行判定。[王力《诗律》](https://www.aisixiang.com/data/98039.html) · [平水韵原表](https://zh.wikisource.org/zh-hant/%E5%B9%B3%E6%B0%B4%E9%9F%BB) · [九歌生成系统](https://aclanthology.org/P19-3005/)

| 同一韵律续训权重，按生成方式 | 五言四句 | 偶句异字同韵 | 三项皆达标 | CUDA 生成中位耗时 |
|---|---:|---:|---:|---:|
| 自由生成 | 148/150 | 92/150 | 17/150 | 0.032 秒 |
| 只限制外形 | 150/150 | 93/150 | 17/150 | 0.051 秒 |
| 限制外形和押韵 | 150/150 | 150/150 | 25/150 | 0.050 秒 |
| 默认，加入关键词和候选排序 | **150/150** | **150/150** | **150/150** | 0.189 秒 |

测试采用原先封存的 50 道新题，每题 3 个采样种子，权重在此之前锁定。原 92M 的同题历史结果按异字规则重新计为 56/150，新权重自由生成是 92/150。两词、三词题各有 51 次试验，新权重自由生成仍各为 0 次全用词。默认版的两组都到 51/51，所需词组由解码约束保证。默认版 150 首中未检测到重复整句或与训练集整首、长诗片段及近重复的匹配。规则计数既不能衡量意境，也不能判定每个用词读起来自然，最新版的诗意、连贯和平仄仍需人工评价。[测试指标](evidence/prosody-study/test/metrics.json) · [逐条生成](evidence/prosody-study/test/generations.jsonl)

## 平仄常格诊断

对同一批封存输出逐字复查四种五言常格。上平、下平记为平；上、去、入记为仄。一个字的可能读法均保留。押首句韵的常格还检查第一句与偶句的同韵和韵脚字是否各异。平仄分母是对应模式中已写成四句五字的数量。[王力《诗律》](https://www.aisixiang.com/data/98039.html) · [逐模式诊断数据](evidence/meter-prosody.json)

| 生成方式 | 五言四句 | 偶句异字同韵 | 逐字平仄合／不合／待判 | 平仄、按常格押韵合／不合／待判 |
|---|---:|---:|---:|---:|
| 自由生成 | 148/150 | 92/150 | 0／126／22 | 0／131／17 |
| 只限制外形 | 150/150 | 93/150 | 0／129／21 | 0／133／17 |
| 限制外形和押韵 | 150/150 | 150/150 | 1／127／22 | 1／130／19 |
| 默认，加入关键词和候选排序 | 150/150 | 150/150 | 1／143／6 | 1／146／3 |

默认模式的 M028／种子 22「古路斜陽下，高樓遠樹邊。登樓殘月暮，一一是秋天。」匹配仄起首句不入韵，韵脚「邊／天」。M001／种子 11「霜蟹肥堪釀，霜螯美可烹。一千頭上雪，三百首陽英。」有可确认的共同平声韵脚，仍不匹配逐字常格。读音歧义和可变通的字位见专项抽查，严格常格数据保持原规则。[格律抽查表](blind_review/meter_review.md)

## 本地复现

运行前准备 [唐宋 92M](../README.md#单卡扩容92m-唐宋版) 的数据和原权重。使用新的运行 ID 和输出目录复跑，单独保留历史证据。

```bash
for arm in random rhyme; do
    python -E scripts/finetune.py --config "configs/sft-prosody-$arm.yaml" \
        --init-sft checkpoints/sft/sft-tang-song-92m-2028/best.pt \
        --run-id "sft-prosody-$arm-repeat"
done
python -E scripts/evaluate_prosody.py validation --original dist/orchid-92m \
    --control checkpoints/sft/sft-prosody-random-repeat/last.pt \
    --rhyme checkpoints/sft/sft-prosody-rhyme-repeat/last.pt --output runs/prosody-repeat/validation
python -E scripts/evaluate_prosody.py test \
    --selection runs/prosody-repeat/validation/metrics.json --output runs/prosody-repeat/test
python -E scripts/evaluate_meter.py --generations runs/prosody-repeat/test/generations.jsonl \
    --suite test --group-by mode --output runs/prosody-repeat/test/meter.json
python -E scripts/generate.py --checkpoint dist/orchid-92m-prosody --keywords 秋雨 孤舟 --json
```

本机版从韵律续训的 `last.pt` 导出，权重与包的指纹见 [解压验收](evidence/delivery-92m-prosody.json)。课题的两个旧匿名评阅轮次继续用原诗和原评分。已登记的回件可用于 14M 原始模型对照。
