<div align="center">
  <img src="assets/orchid-cover.png" alt="Orchid 五言诗生成项目封面" width="960">
  <h1>Orchid</h1>
  <p><strong>从零训练的五言诗生成与韵律控制</strong></p>
  <p>
    <a href="#离线看样例">🌸 离线看样例</a> ·
    <a href="#试用-92m-生成模型">🖥️ 实时生成</a> ·
    <a href="#实验结果">📊 实验结果</a> ·
    <a href="#从源码复现">🧪 从源码复现</a> ·
    <a href="report/latex/README.md">📖 项目报告</a>
  </p>
</div>

Orchid 是从随机初始化开始训练的字符级 GPT。给定 1～3 个短词，模型可以自由生成，也可以按四句、每句五字的句式写诗，让第 2、4 句用同一平水韵平声部的两个不同韵脚。网页把两种结果放在一起，方便逐句比较。

## 离线看样例

打开本地的 [`demo/index.html`](demo/index.html)，点选页面上的 **霜 · 孤舟**、**清霜**等题目，即可查看四组封存样例。浏览器直接读取同目录的样式、脚本和样例数据，离线演示只需带上完整的 `demo/` 目录。

| 解码方式 | 「霜、孤舟」封存输出 |
| --- | --- |
| 自由生成 | 一望長江闊，千尋古戍閑。<br>煙波猶未定，不得上孤山。 |
| 默认押韵生成 | 霜葉滿空落，悲心無處尋。<br>孤舟寒夜水，獨自照秋林。 |

这组输出取自冻结测试 M017，种子为 11。默认模式的韵脚「尋／林」同属下平声十二侵。两栏都展示句式、韵脚、原词覆盖和生成耗时。

<details>
<summary>🖼️ 查看网页实拍</summary>

<img src="report/latex/figures/demo.png" alt="Orchid 网页中的关键词、种子和两首并排的诗" width="920">

</details>

## 试用 92M 生成模型

### 📦 1. 找到推理权重

| 手头的文件 | 运行命令的位置 | 检查点路径 |
| --- | --- | --- |
| `Orchid-model-kit-demo.zip` | 解压后进入 `orchid-course/` | `model` |
| 源码及已导出的 92M 韵律权重 | 仓库根目录 | `dist/orchid-92m-prosody` |

模型包内的 `model/` 包含权重、字表、配置和校验清单。可以在 ZIP 所在目录执行下面的命令。若从仓库根目录解压本机留档，将 ZIP 路径改为 `dist/Orchid-model-kit-demo.zip`。

```bash
unzip Orchid-model-kit-demo.zip -d orchid-demo
cd orchid-demo/orchid-course
```

### 🧰 2. 安装 CPU 或 CUDA 依赖

以下命令以 Linux x86_64 和 Python 3.11.16 为例。已有可用的 Orchid 虚拟环境时，直接激活即可。新建环境时，安装了 `uv` 的机器可运行下列命令。

```bash
uv venv --python 3.11.16 --seed .venv
source .venv/bin/activate
python --version
```

也可用本机已安装的 Python 3.11.16 执行 `python3.11 -m venv .venv`，再激活环境。根据演示设备选择一组依赖。

**💻 CPU 推理**

```bash
python -m pip install -r requirements-cpu.txt
DEVICE=cpu
```

**⚡ NVIDIA GPU 推理**

```bash
python -m pip install -r requirements.txt
python -I scripts/check_env.py --require-cuda --verify-pins --output runs/environment.json
DEVICE=cuda
```

训练环境使用 `requirements.txt` 固定的 PyTorch CUDA 12.6 构建。在 VS Code 中可按[工作区设置模板](.vscode/settings.example.json)选择 `.venv/bin/python`。运行命令中的 `python -E` 可隔离全局 Python 环境变量。

### ✍️ 3. 启动网页并现场演示

下例从解压后的模型包运行。若在仓库根目录操作，将第一行改为 `MODEL=dist/orchid-92m-prosody`。在已激活的现成环境中运行时，可把 `--device "$DEVICE"` 直接写成 `--device cpu` 或 `--device cuda`。

```bash
MODEL=model
python -E scripts/serve_demo.py --checkpoint "$MODEL" --device "$DEVICE"
```

1. 终端打印服务地址后，浏览器打开 [本地演示页](http://127.0.0.1:8765/)。
2. 页面显示「模型就绪」后，输入 `霜 孤舟`，保留种子 `11`，点击 **生成两首**。输入可含 1～3 个互异的短词，每词 1～2 个汉字，用空格或顿号分隔。繁体词可先用 `清霜`、`煙波` 或 `孤舟`。
3. 对照两栏的诗句，先看第 2、4 句末字和韵部，再看原词覆盖与耗时。原词覆盖逐字检查，切题、可读性、连贯性与意境按量表分别记录。
4. 保持关键词和种子，切换 **减少隔位重字** 后再点击生成。该排序调整押韵模式挑选候选诗的方式。
5. 演示结束，在终端按 `Ctrl+C` 停止服务。如果端口 8765 被占用，启动命令可加 `--port 8766`，浏览器打开终端打印的新地址。

### ⌨️ 4. 直接从命令行出诗

延用上一步的 `MODEL` 和 `DEVICE`。相同题目和种子可用于展示不同解码设置。

```bash
python -E scripts/generate.py --checkpoint "$MODEL" --device "$DEVICE" \
  --keywords 秋雨 孤舟 --seed 11

python -E scripts/generate.py --checkpoint "$MODEL" --device "$DEVICE" \
  --keywords 秋雨 孤舟 --seed 11 --mode rhymed --ranking repeat-aware --json
```

| 想看什么 | 命令参数 |
| --- | --- |
| 自由抽样 | `--mode raw` |
| 四句五字与偶句押韵 | `--mode rhyme-only` |
| 四句五字、偶句押韵、逐字覆盖原词 | `--mode rhymed`，省略 `--mode` 时采用此模式 |
| 押韵候选少选同句隔位重字 | `--mode rhymed --ranking repeat-aware` |
| 输出结构化结果 | `--json`，含诗句、命中原词、模型哈希、加载与生成耗时，押韵模式另含韵脚和韵部 |

复现旧实验时可用 `--constrain-form` 或 `--constrain-keywords` 分别启用字数限制与固定词位解码。CPU 和 CUDA 在相同种子下可能生成不同的诗，逐字复测时应固定设备。

## 实验结果

📊 冻结测试使用 50 道题和 3 个种子，每种解码生成 150 条。三项达标同时要求四句五字、输入词原样覆盖，以及第 2、4 句异字同平声韵。

| 解码 | 四句五字 | 偶句异字同平声韵 | 原词全覆盖 | 三项达标 |
| --- | ---: | ---: | ---: | ---: |
| 自由生成 | 148/150 | 92/150 | 25/150 | 17/150 |
| 字数与押韵约束 | 150/150 | 150/150 | 25/150 | 25/150 |
| 默认完整约束 | 150/150 | 150/150 | 150/150 | 150/150 |

🔬 另在 36 道新题上比较候选排序，每题采样 3 次。采用减少隔位重字的排序后，出现同句隔位重字的输出从 13/108 降至 9/108，两种排序的三项达标均为 108/108。[冻结测试与计时](report/release_evaluation.md) · [排序补测记录](report/evidence/ranking-ablation-20261121/challenge/metrics.json)

## 从源码复现

源代码包 `Orchid-source-demo.zip` 包含配置、数据指纹、训练与复测脚本以及封存结果。在仓库根目录执行以下命令。需要重新训练时，先按上面的 CUDA 步骤建好环境。

### 📚 数据与字表

```bash
python -E scripts/prepare_data.py --config configs/data.yaml
python -E scripts/build_tokenizer.py --config configs/data.yaml
python -E scripts/build_sft_data.py --config configs/sft.yaml
```

数据脚本下载固定版本的唐诗分片到 `data/raw/tang/`。`prepare_data.py --offline` 可从已校验的本地分片复跑。`data/processed/` 存放三套独立诗集及 token 序列，`artifacts/` 存放来源清单、冻结划分、字表和统计。字符级字表仅用训练集构建，共 8,582 个 token（含保留标记）。[数据约定](docs/data_contract.md) · [token 统计](artifacts/token_stats.json)

下面的历史运行名用于匹配配置和导出命令。若本机已有同名运行，另选运行名并相应调整检查点与评估配置中的路径。

<details>
<summary>🔁 复现 14M 基线</summary>

#### 预训练、恢复与监督微调

```bash
python -E scripts/pretrain.py --config configs/pretrain.yaml --run-id pretrain-tang-v1 --stop-after 87
python -E scripts/pretrain.py --config configs/pretrain.yaml --resume checkpoints/pretrain/pretrain-tang-v1/last.pt

for seed in 2026 2027 2028; do
    python -E scripts/finetune.py --config configs/sft.yaml \
        --init checkpoints/pretrain/pretrain-tang-v1/best.pt \
        --seed "$seed" --run-id "sft-tang-$seed"
done
```

预训练在第 87 步写盘后恢复，完整调度以 10 轮为上限。预训练日志位于 `runs/pretrain-tang-v1/`，权重位于 `checkpoints/pretrain/pretrain-tang-v1/{random,best,last}.pt`。14M 演示使用微调种子 2027。[预训练实测](report/pretraining.md) · [微调选模](report/finetuning.md)

#### 自动评估和推理包

```bash
python -E scripts/evaluate.py --config configs/eval.yaml --device cpu --output runs/fixed-eval-v1
python -E scripts/package_model.py \
    --checkpoint checkpoints/sft/sft-tang-2027/best.pt --output dist/orchid-14m
python -E scripts/generate.py --checkpoint dist/orchid-14m \
    --keywords 秋雨 孤舟 --device cpu --mode raw --json
```

若使用留档的 14M 模型包，解压到独立目录后传入 `--checkpoint model`。每个推理包的 `bundle.json` 记录权重、字表与配置的哈希。[基线封存评估](report/evaluation_baseline.md)

</details>

### 单卡扩容：92M 唐宋版

完成上面的**数据与字表**三步后，下载宋诗、保持唐诗留出集、构建扩容监督数据，随后训练 92M 版本。

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

python -E scripts/package_model.py --checkpoint checkpoints/sft/sft-tang-song-92m-2028/best.pt \
    --tokenizer artifacts/tang-song/tokenizer.json --output dist/orchid-92m
```

预训练每半轮验证，最多 16 轮，连续三次无改善时早停。封存运行在第 9 轮保存最佳权重、第 10.5 轮停止，实际优化约 31 分钟。可用同一配置加 `--resume checkpoints/pretrain/pretrain-tang-song-92m/last.pt` 恢复。三个微调种子各跑 4 轮，固定验证选出 2028。后续按实测方案完成普通续训与同韵续训对照，导出 `dist/orchid-92m-prosody`。[韵律续训与导出步骤](report/prosody_optimization.md#本地复现)

### 🧪 复跑评估与构建模型包

已有推理包即可复跑新题排序补测。全套固定规则测试还需上一步的唐宋数据及原划分。

```bash
python -E scripts/evaluate_ranking.py --split challenge --checkpoint dist/orchid-92m-prosody \
    --device cuda --output runs/ranking-replay
python -E scripts/create_delivery.py --bundle dist/orchid-92m-prosody \
    --output dist/Orchid-model-kit-demo.zip
```

多模型冻结对照可运行 `python -E scripts/evaluate.py --config configs/eval-expanded.json --device cuda --output runs/retained-expanded-eval-v1`，配置需要本地 14M、42M 和 92M 对照权重。完整配置、逐条输出及指纹均已归档。[完整固定测试命令](report/release_evaluation.md#复跑命令) · [发布包构建与验证](docs/open_source_cleanup.md) · [机器可读证据](report/evidence/)

## 课程材料

- 📖 [项目报告与 LaTeX 构建](report/latex/README.md)
- 📝 [课程要求](Problem.md) · [建设与验收清单](GPT_BUILD_CHECKLIST.md)
- 📚 [数据及第三方来源](THIRD_PARTY_NOTICES.md)
