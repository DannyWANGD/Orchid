# 数据约定

实验问题：用唐诗从零训练小型 GPT 后，监督微调能否让同一模型按给定的 1～3 个短关键词创作五言四句诗？预训练预测诗文中的下一个字符标记；监督微调仅在条件后的正文和 EOS 上计损失。

| 数据 | 来源、固定版本及取得方式 | 实验用途 |
|---|---|---|
| 唐诗 JSON | 公开复用：[chinese-poetry 固定提交](https://github.com/chinese-poetry/chinese-poetry/tree/b8594f81a89752241442f2ce267d6f66f96704ee/%E5%85%A8%E5%94%90%E8%AF%97)，按提交 URL 读取 `poet.tang.0.json` 至 `poet.tang.57000.json`；2026-11-18 已只读复核 | 以整首诗为单位划分预训练与 SFT 的共同 train/validation/test，标签由冻结规则派生 |
| 独立诗题 | 人工编写若干无参考诗的 1～3 词组合，确定模型后封存并固定版本 | 主测试；绝不进入训练或超参数选择 |
| 平水韵表 | [固定版本繁体 JSON](https://github.com/rbnyng/pingshui_rhyme/blob/0d6b05a932ff1828d8403e7a0de67dd1f6bd81d0/pingshui_rhyme/data/organized_ping_ze_rhyme_dict.json)，SHA-256 见清单 §10.3 | 仅做偶句同平声韵的评价，不输入训练程序 |

训练数据保留繁体和来源索引。正文按 Unicode NFC 和 §4.2 的句读规则初筛，不从长诗截出四句充作监督答案；来源缺字和夹注另存复核。宋诗、宋词、《唐诗三百首》选本、标题与作者不混入训练正文。许可与来源见 [第三方登记](../THIRD_PARTY_NOTICES.md)。

先全局分组（相同正文、已识别近重复及四句连续内嵌），再用种子 `2026` 按约 `90/5/5` 一次划分。词表只看 train 文本，窗口与监督配对继承所属诗作的集合，不跨集合拼接。近重复规则覆盖有限；评估另外检查生成诗与训练诗全体的匹配。

实验产物关联 `data_manifest.json`、`split_manifest.json`、`data_stats.json` 和 tokenizer 哈希。论文分数、样本文本、调试小批次不替代真实运行结果。只有实际在该快照上执行的训练与评估，才报告为实测。
