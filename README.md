# Orchid：古诗 GPT

Orchid 是一个从随机初始化训练的字符级 GPT。课程任务是给定 1～3 个关键词，生成一首五言四句诗。预训练用古诗正文预测下一个 token，监督微调学习如何按给定关键词写诗。

课程要求见 [Problem.md](Problem.md)，进度和验收标准见 [GPT_BUILD_CHECKLIST.md](GPT_BUILD_CHECKLIST.md)。[数据约定](docs/data_contract.md)说明语料和测试边界；具体参考及许可证见 [第三方登记](THIRD_PARTY_NOTICES.md)。

在 Ubuntu 22.04、Python 3.11.16 上，用仓库的 `.venv` 或用同版本 Python 新建虚拟环境。`requirements.txt` 固定了 CUDA 12.6 构建的 PyTorch。

```bash
source .venv/bin/activate
unset PYTHONPATH
python -m pip check
python -I scripts/check_env.py --require-cuda --verify-pins --output artifacts/environment.json
```

如果是新建的空环境，激活后先运行 `python -m pip install -r requirements.txt`。`check_env.py` 会写出版本、显卡、BF16 能力和 CUDA 前向/反向检查结果；只使用 CPU 时去掉 `--require-cuda`。VS Code 可参考 [工作区设置模板](.vscode/settings.example.json)，在「Python: Select Interpreter」中选择 `.venv/bin/python`。如果 ROS 2 扩展自动写回 Python 3.10 的分析路径，可在 Orchid 工作区单独停用该扩展；运行命令用 `python -E` 避开 Python 环境变量。

从仓库根目录获取和准备语料：

```bash
python -E scripts/prepare_data.py --config configs/data.yaml
python -E scripts/build_tokenizer.py --config configs/data.yaml
```

首次运行下载原始唐诗到 `data/raw/tang/`。添加 `--offline` 可用本地已校验的分片复跑。同一快照和配置重建 `data/processed/`，其中包含三个独立诗集与 token 流；`artifacts/` 保存源文件清单、冻结的划分、字表和统计。训练只学习 train 的 8,582 个 token（含保留标记）；验证和测试里的生僻字可以变成 `<unk>`，比例见 [token 统计](artifacts/token_stats.json)。

正式预训练可按实际记录分两段运行（首轮试训后完整续训，调度仍以 10 轮为上限）：

```bash
python -E scripts/pretrain.py --config configs/pretrain.yaml --run-id pretrain-tang-v1 --stop-after 87
python -E scripts/pretrain.py --config configs/pretrain.yaml --resume checkpoints/pretrain/pretrain-tang-v1/last.pt
```

日志写入 `runs/pretrain-tang-v1/`；权重写入 `checkpoints/pretrain/pretrain-tang-v1/{random,best,last}.pt`。重新运行实验时使用新的 `--run-id`。详见 [预训练实测](report/pretraining.md)。

从上述最佳权重派生监督数据并运行三个微调种子：

```bash
python -E scripts/build_sft_data.py --config configs/sft.yaml
for seed in 2026 2027 2028; do
    python -E scripts/finetune.py --config configs/sft.yaml \
        --init checkpoints/pretrain/pretrain-tang-v1/best.pt \
        --seed "$seed" --run-id "sft-tang-$seed"
done
```

每轮在验证集中记录正文 NLL 与固定条件样例。本次实测的演示权重是 `checkpoints/sft/sft-tang-2027/best.pt`；选择规则与三个种子见 [微调实测](report/finetuning.md)。
