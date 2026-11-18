# 预训练实测

Orchid 在固定唐诗训练集上以种子 2026 随机初始化，没有加载外部权重。使用 RTX 4080 SUPER、BF16 autocast、微批 32、累积 4 和初始峰值学习率 (3\times10^{-4})。第一个 epoch 完成后落盘，随后从 `last.pt` 继续到第 870 个优化步，日志步号没有断裂或重复。完成全部 10 个 epoch，共处理 28,487,670 个有效目标 token；优化耗时 58.9 秒，验证、采样、写盘额外计时。最佳权重位于第 870 步。

| 检查点 | 验证 NLL | 验证 PPL |
|---|---:|---:|
| 随机初始化 | 9.1361 | 9,284.1 |
| 第 1 epoch | 6.3573 | 576.7 |
| 第 5 epoch | 5.2748 | 195.4 |
| 第 10 epoch | 4.9540 | 141.7 |

验证集在各次均有 158,708 个有效目标；验证指标含正文标点以及诗作边界的 BOS/EOS，按 token 汇总，不和微调的正文掩码 NLL 直接比较。曲线 [PDF](figures/pretrain.pdf) / [CSV](figures/pretrain.csv)，[逐步日志](evidence/pretrain-tang-v1/metrics.jsonl)，[固定前缀原始采样](evidence/pretrain-tang-v1/samples.jsonl)，[配置与哈希](evidence/pretrain-tang-v1/checkpoints.json)。

给定「春」，最佳检查点一次续写为「春風自有情，獨坐故人歸。」后输出 EOS；仅两句，还不能算完成五言四句任务。早期也会重复标点。验证 NLL 在最后一个 epoch 仍下降，但按预定 10 轮结束；本轮先构建并验证监督微调，是否增加诗料由验证诊断决定。
