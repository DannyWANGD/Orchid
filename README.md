# Orchid：古诗 GPT

Orchid 是一个从随机初始化训练的字符级 GPT。课程任务是给定 1～3 个关键词，生成一首五言四句诗。预训练用古诗正文预测下一个 token，监督微调学习如何按给定关键词写诗。

课程要求见 [Problem.md](Problem.md)，进度和验收标准见 [GPT_BUILD_CHECKLIST.md](GPT_BUILD_CHECKLIST.md)。[数据约定](docs/data_contract.md)说明语料和测试边界；具体参考及许可证见 [第三方登记](THIRD_PARTY_NOTICES.md)。

## 使用新版生成古诗

```bash
python -E scripts/generate.py --checkpoint dist/orchid-92m-prosody --keywords 秋雨 孤舟 --device cuda --json
```

新版默认写四句，每句五字，使用全部关键词，第 2、4 句以同一平水韵平声部的不同字结尾。输入可包含 1～3 个繁体关键词，每个 1～2 字。用 `--mode raw` 查看模型自由生成，`--mode rhyme-only` 只限定字数和押韵。原有 `--constrain-form` 和 `--constrain-keywords` 分别复现旧实验的字数和固定词位。JSON 会给出韵脚字、韵部和生成时间。

解压 [92M 押韵版课程包](dist/Orchid-course-prosody.zip) 后，将命令中的检查点改为 `model`。无 CUDA 时用 `--device cpu`。续训、解码对照和逐条输出见[押韵改进实测](report/prosody_optimization.md)。

## 环境与基础训练

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

基线冻结自动评估与模型交付：

```bash
python -E scripts/evaluate.py --config configs/eval.yaml --device cpu --output runs/fixed-eval-v1
python -E scripts/package_model.py --checkpoint checkpoints/sft/sft-tang-2027/best.pt --output dist/orchid-14m
python -E scripts/generate.py --checkpoint dist/orchid-14m --keywords 秋雨 孤舟 --device cpu --mode raw --json
python -E scripts/generate.py --checkpoint dist/orchid-14m --keywords 秋雨 孤舟 --device cuda --mode raw --json
```

CPU 和 CUDA 在相同种子下可能生成不同诗作，因此正式自动对照固定设备。`dist/orchid-14m/bundle.json` 记录从检查点导出的权重、字表和配置哈希。详见[封存评估](report/evaluation_baseline.md)和[盲评说明](report/blind_review/README.md)。

### 历史基线包与盲评

本机 ZIP 示例：[14M 基线包](dist/Orchid-course-baseline.zip)。在新目录解压进入 `orchid-course/`，安装固定依赖后运行：

```bash
python -E scripts/generate.py --checkpoint model --keywords 青山 白雲 --device cpu --constrain-keywords --json
```

评阅者收到的是各自的匿名诗、题目及评分说明。14M 的 v1 归档一份真人回件和一份 AI 参照，v2 匿名包用于新一轮。汇总会核对同版本两份表，并按维度保存评分分布及适用数。[盲评说明](report/blind_review/README.md)

报告源文件 [Markdown](report/experiment_report.md)，可读版 [PDF](report/Orchid课程报告.pdf)。本机有 XeLaTeX 与 Noto CJK 字体时，从仓库根目录运行 `python -E scripts/build_report.py` 重建 PDF。

### 单卡扩容：92M 唐宋版

扩容使用同一公开唐诗与宋诗快照，验证和测试仍是固定唐诗集合。应先完整跑通上面的基础数据与监督构建。

```bash
python -E scripts/download_song.py --config configs/song.yaml
python -E scripts/prepare_tang_song.py --config configs/data-tang-song.yaml
python -E scripts/build_sft_data.py --config configs/sft-tang-song.yaml
python -E scripts/pretrain.py --config configs/pretrain-tang-song.yaml --run-id pretrain-tang-song-92m
for seed in 2026 2027 2028; do
    python -E scripts/finetune.py --config configs/sft-tang-song.yaml \
        --init checkpoints/pretrain/pretrain-tang-song-92m/best.pt \
        --seed "$seed" --run-id "sft-tang-song-92m-$seed"
done
```

预训练每半轮验证，最多 16 轮，连续三次无改善时早停。本次在第 9 轮保存最佳、第 10.5 轮停，实际优化约 31 分钟；途中能直接用同一配置加 `--resume checkpoints/pretrain/pretrain-tang-song-92m/last.pt` 恢复。三个 SFT 种子各实跑 4 轮。交付模型是固定验证题选出的种子 2028：

```bash
python -E scripts/package_model.py --checkpoint checkpoints/sft/sft-tang-song-92m-2028/best.pt --output dist/orchid-92m
python -E scripts/generate.py --checkpoint dist/orchid-92m --keywords 霜 孤舟 --device cpu --mode raw --json
python -E scripts/generate.py --checkpoint dist/orchid-92m --keywords 霜 孤舟 --device cuda --constrain-keywords
```

如要复跑多模型冻结对照，使用 `python -E scripts/evaluate.py --config configs/eval-expanded.json --device cuda --output runs/retained-expanded-eval-v1`；该配置同时登记 14M/42M/92M 对照的本地权重。只查看现成结果无需训练。每个运行 ID 的配置、成绩与 checkpoint 哈希在 [report/evidence](report/evidence/)。

原 92M 权重、封存对照与两份匿名表可从旧版留档查看。当前报告纳入了新版韵律续训、推理解码和同题对照。[原 92M 本地 ZIP](dist/Orchid-course.zip) · [封存测试](report/evaluation_expanded.md) · [课程 PDF](report/Orchid课程报告.pdf)
