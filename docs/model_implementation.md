# GPT 实现与训练机制检查

模型规格在 [model.yaml](../configs/model.yaml)，代码见 [model.py](../orchid/model.py)。实际字表 8,582 个 token，6 层，宽度 384，每层 6 个 64 维注意力头，最大上下文 256；共有 **14,041,344** 个可训练参数。输出投影与字嵌入共享同一个参数，优化器没有重复登记。

```mermaid
flowchart LR
    A["字符 ID [B,T]"] --> B["字嵌入 + 可学习位置嵌入 [B,T,384]"]
    B --> C["6 个 GPT 块：先归一化，注意力或 MLP，再加残差"]
    C --> D["最后归一化"]
    D --> E["用字嵌入矩阵转置得到 logits [B,T,8582]"]
```

每个块把 Q、K、V 拆成 `[B,6,T,64]`。用 QKᵀ/√64 按查询与键的相似度打分，遮住未来位置，softmax 后对 V 加权。模型可在同一批输入上用手写公式或 SDPA；显式掩码中 `True` 表示允许关注。训练样本只从右侧补齐，因果掩码已让有效查询不读取右侧 PAD，损失还把这些位置的标签屏蔽。前馈网络将宽度扩展至 1,536，以 GELU 激活后投影回 384。初始化用 N(0,0.02)，线性偏置为 0，LayerNorm 比例为 1、偏移为 0。

训练使用 AdamW、余弦调度、梯度裁剪和 BF16 autocast，模型参数仍是 FP32。每个优化步先数完各微批的有效标签，再以同一个总数除各微批的交叉熵和。验证累计整套有效标签，不把大小不同的批次等权平均。checkpoint 保存下一批位置、优化器、调度器、各随机数生成器状态和输入哈希。

2026-11-18 的[检查记录](../artifacts/training_checks.json)包括：

| 检查 | 测得 |
|---|---|
| RTX 4080 SUPER，完整配置 FP32，批量 2 × 256 | 前向、反向均有限，参数为 FP32 |
| 24 首 train 诗作，BF16 调试 100 步 | NLL 9.1333 → 0.1348；给定 5 字前缀，贪心完整复现 24/24 |
| BF16，微批 32，累计 4；预热 50 步，计时 200 步 | 481,016 有效 token/s，峰值分配 1.61 GiB，87 步/epoch |
| 预计 10 个 epoch 与 checkpoint 存储 | 约 59 秒优化计算（不计采样、验证、写盘）；约 393 MB 用于 random/best/last |

24 首实验只验证实现能学习，不衡量新题泛化。CPU FP32 上另跑了包含 dropout 的完整中断恢复测试：第 3 步中断后续训，最终参数和 AdamW 状态在零容差下与连续运行相同，调度、数据位置和日志步数一致。该结果不承诺其他设备或软件版本逐位一致。

```bash
python -E -m unittest discover -s tests -v
python -E scripts/check_training.py --config configs/pretrain.yaml --output runs/check_training.json
```
