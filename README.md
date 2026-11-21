# Orchid：古诗 GPT

Orchid 是一个从随机初始化训练的字符级 GPT。课程任务是给定 1～3 个关键词，生成一首五言四句诗。预训练用古诗正文预测下一个 token，监督微调学习如何按给定关键词写诗。

课程要求见 [Problem.md](Problem.md)，进度和验收标准见 [GPT_BUILD_CHECKLIST.md](GPT_BUILD_CHECKLIST.md)。[数据约定](docs/data_contract.md)说明语料和测试边界；具体参考及许可证见 [第三方登记](THIRD_PARTY_NOTICES.md)。

## 试用 92M 生成模型

获取 `Orchid-model-kit.zip`，解压后进入 `orchid-course/`。模型包自带 92M 推理权重和离线韵表，用 Python 3.11.16 在 Linux x86_64 上即可运行。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -E scripts/generate.py --checkpoint model --keywords 秋雨 孤舟 --device cpu --json
```

用 NVIDIA GPU 时，在独立虚拟环境中安装 `requirements.txt`，生成时传 `--device cuda`。源码工作目录里的已导出权重通常位于 `dist/orchid-92m-prosody`，可直接传给 `--checkpoint`。

默认模式写四句，每句五字，使用全部关键词，第 2、4 句以同一平水韵平声部的不同字结尾。请提供 1～3 个互异关键词，每词 1～2 个字，字形以训练字表为准。加 `--mode raw` 可看自由生成，加 `--mode rhyme-only` 可只限定字数和押韵。原有 `--constrain-form` 和 `--constrain-keywords` 用于复现只限字数及固定词位的旧实验。JSON 会给出韵脚字、韵部和生成时间。

`Orchid-source.zip` 含源码、配置、数据指纹和封存结果。复跑下列训练步骤可从语料重建模型，有现成模型包即可从推理和复测开始。查看[当前模型评估与展示建议](report/release_evaluation.md)、[源码和模型包的构建与验证](docs/open_source_cleanup.md)。

## 环境与基础训练

训练和报告使用 Ubuntu 22.04、Python 3.11.16 与 `requirements.txt` 中固定的 PyTorch CUDA 12.6 及其依赖。在源码根目录创建环境后运行下面的命令。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -I scripts/check_env.py --require-cuda --verify-pins --output runs/environment.json
```

`check_env.py` 记录版本、显卡、BF16 能力和 CUDA 前后向结果。CPU 推理环境可运行 `python -I scripts/check_env.py --verify-pins --requirements requirements-cpu.txt`。VS Code 可按[工作区设置模板](.vscode/settings.example.json)选择 `.venv/bin/python`。示例运行命令用 `python -E` 避开全局 Python 环境变量。

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

如有留档的 14M 基线包，可在新目录解压进入 `orchid-course/`，安装包内固定依赖后运行。

```bash
python -E scripts/generate.py --checkpoint model --keywords 青山 白雲 --device cpu --constrain-keywords --json
```

评阅者收到的是各自的匿名诗、题目及评分说明。14M 主对照采用提交者确认的 v1 真人回件与同版 AI 参照。v2 已核验一份 AI 回件以复核量表。汇总核对同版本表，并按维度保存评分分布及适用数。[盲评说明](report/blind_review/README.md)

报告源文件 [Markdown](report/experiment_report.md)，可读版 [PDF](report/Orchid课程报告.pdf)。本机有 XeLaTeX 与 Noto CJK 字体时，从仓库根目录运行 `python -E scripts/build_report.py` 重建 PDF。

### 单卡扩容：92M 唐宋版

扩容使用固定的公开唐诗与宋诗快照，验证和测试沿用唐诗留出集。从源码开始时，先运行上面的 `prepare_data.py`、`build_tokenizer.py` 和 `build_sft_data.py --config configs/sft.yaml`，再执行以下命令。

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
python -E scripts/package_model.py --checkpoint checkpoints/sft/sft-tang-song-92m-2028/best.pt \
    --tokenizer artifacts/tang-song/tokenizer.json --output dist/orchid-92m
python -E scripts/generate.py --checkpoint dist/orchid-92m --keywords 霜 孤舟 --device cpu --mode raw --json
python -E scripts/generate.py --checkpoint dist/orchid-92m --keywords 霜 孤舟 --device cuda --constrain-keywords
```

如要复跑多模型冻结对照，使用 `python -E scripts/evaluate.py --config configs/eval-expanded.json --device cuda --output runs/retained-expanded-eval-v1`；该配置同时登记 14M/42M/92M 对照的本地权重。只查看现成结果无需训练。每个运行 ID 的配置、成绩与 checkpoint 哈希在 [report/evidence](report/evidence/)。

继续生成最新版权重时，按[韵律续训的复现步骤](report/prosody_optimization.md#本地复现)训练和导出。旧版对照见[封存测试](report/evaluation_expanded.md)和[课程 PDF](report/Orchid课程报告.pdf)。
