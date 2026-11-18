# Orchid：从零构建古诗 GPT 的任务清单

编制日期：2026-11-17；语料与评估复核日期：2026-11-18。适用平台：Ubuntu 22.04 / Linux x86_64、Python 3.11.16、单张 RTX 4080 SUPER（约 16GB 显存）。课程依据：[Problem.md](Problem.md)。

**项目目标：用 PyTorch 搭建小型 GPT，从随机初始化完成古诗语料预训练，再对自己的预训练模型进行监督微调，实现按关键词生成五言四句诗，并提供本地推理、评估与课程报告。**

`[x]` 表示已完成的工作或验证；`[ ]` 表示待完成任务。第 4.2 节保留实施前的语料复核结果。实际运行数据、配置和产物随各板块登记。

## 1. 确定任务范围和完成标准

### 1.1 默认实施路线

| 项目 | 本轮采用的方案 | 选择原因 |
|---|---|---|
| 最小生成任务 | 给定 1～3 个关键词（每词 1～2 字），创作一首五言四句诗 | 字数和用词可自动检查；切题、连贯和诗意由人评判 |
| 预训练语料 | 先使用清洗后的唐诗；数据不足时，再按相同规则加入宋诗 | 先保证数据质量和可复现性，实际数量由脚本统计 |
| 字形 | 默认保留源语料繁体；输入与评估沿用同一字形规范 | 源数据主要为繁体，直接转换可能改变词义或产生错误 |
| 分词 | 字符级 tokenizer：一个汉字或标点通常对应一个 token，特殊标记单独编码 | 便于理解和检查字数，避免直接套用英语 GPT-2 词表 |
| 模型 | Decoder-only、6 层、隐藏维度 384、6 个注意力头、上下文 256 | 保留 GPT 的核心机制，适合作为单卡课程实验起点 |
| 参数量 | 由实际词表和实现决定；词表 8,000 时，本文配置约 1,382 万参数 | 课程允许小参数量模型，不需要追求十亿级规模 |
| 微调 | 加载自己的预训练检查点，进行全参数 SFT | 模型较小，完整展示监督数据和目标函数即可 |
| 推理 | 先交付命令行程序，支持 GPU 和 CPU | 便于复现与验收；网页界面属于可选增强 |
| 数据与结果 | 使用真实公开语料，保存实际训练和评估记录 | 所有性能数字都需要追溯到运行记录 |

本项目把任务称为**五言四句诗**：四句各五字，意思连贯，能围绕给定意象写一首诗。第 10 节另查偶句韵脚；如果目标升级为**五言律绝**，需要按第 12 节另筛语料并校验平仄。诗体区分参见[王力《诗律》](https://www.aisixiang.com/data/98039.html)。

- [x] SCOPE-01 固化本轮任务定义：输入关键词，输出五言四句诗；在配置与报告中使用同一个定义。
- [x] SCOPE-02 写出两个阶段的差别：预训练学习诗歌文本的下一个 token；SFT 学习在给定条件下输出诗作。
- [x] SCOPE-03 明确自建模型、自行预训练的主体路线；见 [第三方登记](THIRD_PARTY_NOTICES.md)。
- [x] SCOPE-04 约定实验完成标准：数据可追溯、模型能训练和续训、两个阶段有各自检查点、评估可重跑、本地推理可使用、报告覆盖全部评分项。

### 1.2 与课程评分逐项对应

| 课程要求 | 占比 | 必须保留的证据 |
|---|---:|---|
| 知识点梳理 | 35% | 架构图、关键公式、配置解释、数据处理说明、预训练与微调原理、对应代码位置 |
| 小参数量 GPT 搭建 | 20% | 模型代码、参数量统计、前向与因果性验证、单卡运行记录 |
| 预训练 | 20% | 数据清单、词表、训练程序、训练/验证曲线、检查点、固定输入的生成样例 |
| 微调 | 20% | 监督样本、标签与损失掩码示例、微调配置、微调前后对照、任务指标 |
| 成果与体会 | 5% | 失败案例、局限、实际困难和解决方法、个人收获与分工 |

报告是主要评分依据；源码虽非强制提交项，仍建议整理。用到的公开数据和开源资料需在报告中给出网址；大文件按课程要求提供下载方式。

## 2. 环境与依赖：先固定已经可用的组合

### 2.1 当前已知状态

| 组件 | 已核实版本或状态 |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.14.1+cu126 |
| PyTorch CUDA 构建 | 12.6 |
| CUDA Toolkit Python 包 | 12.6.3 |
| CUDA Python bindings | 12.9.7 |
| Triton | 3.8.0 |
| NVIDIA 驱动 | 560.35.05 |
| 显卡 | RTX 4080 SUPER，16376 MiB |
| GPU 验证 | 用户已运行 CUDA 矩阵乘法及反向传播，验证通过 |
| 依赖验证 | 本次检查的 182 个版本固定项均与本地安装一致；隔离 ROS 路径后 `pip check` 通过 |

CUDA Toolkit、bindings、cuDNN 等组件的版本号不要求全部相同，应遵循 PyTorch 声明的依赖关系。`requirements.txt` 已固定 `torch==2.14.1+cu126`，并增加官方 `cu126` 软件源。参考：[PyTorch 安装版本表](https://pytorch.org/get-started/previous-versions/)、[NVIDIA CUDA 兼容表](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)。

- [x] ENV-01 用户完成兼容版本安装，并验证 GPU 前向、反向计算。
- [x] ENV-02 更新 [requirements.txt](requirements.txt)：同步 PyTorch、Triton、CUDA Toolkit、bindings 和配套 CUDA 12 库的实际版本。
- [x] ENV-03 从依赖文档移除旧 CUDA 13 库的固定项，保留原有其他工具依赖；没有卸载现有环境中的残留包。
- [x] ENV-04 检查固定项与本地版本一致，并通过当前环境中的依赖检查和不安装软件的解析检查。
- [x] ENV-05 `scripts/check_env.py` 验证版本与 BF16 能力，实测证据见 [环境记录](artifacts/environment.json)。
- [x] ENV-06 提供选择 `.venv/bin/python` 的工作区模板和隔离 ROS 的实测命令；扩展重写的本地设置不入库。
- [x] ENV-07 2026-11-18 从空的 Python 3.11.16 venv 执行 `pip install -r requirements.txt`，182 个固定项、`pip check` 和 CUDA 前向/反向均通过。
- [x] ENV-08 基线沿用现有依赖；后续仅在实际使用新工具时增加并固定版本。

现有环境的日常检查命令：

```bash
cd /home/ubt2204/Desktop/Orchid
source .venv/bin/activate
unset PYTHONPATH
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

`unset PYTHONPATH` 只影响当前终端。依赖文件保留了原来较完整的教学工具集合，包含 Notebook 和 TensorFlow CPU 等工具；这不表示项目训练需要同时使用多个深度学习框架。重建时以版本固定文件和安装源为准，不直接把带有 ROS 路径或旧 CUDA 残留的 `pip freeze` 输出覆盖进去。

## 3. 组织工程并登记开源借用

项目的主要实现和产物结构如下：

```text
Orchid/
├── Problem.md
├── GPT_BUILD_CHECKLIST.md
├── requirements.txt
├── README.md
├── configs/                 # model、pretrain、sft、generation 配置
├── orchid/
│   ├── tokenizer.py         # 字表、特殊标记、编码/解码
│   ├── data.py              # Dataset、拼接、collate、标签与掩码
│   ├── model.py             # GPT、注意力、前馈网络
│   ├── training.py          # 优化步、评估、日志与检查点
│   ├── generation.py        # 自回归采样
│   └── metrics.py           # 格式、关键词、重复与记忆检查
├── scripts/
│   ├── check_env.py
│   ├── prepare_data.py
│   ├── build_tokenizer.py
│   ├── build_sft_data.py
│   ├── pretrain.py
│   ├── finetune.py
│   ├── evaluate.py
│   ├── generate.py
│   └── plot_results.py
├── tests/                   # 关键机制与数据边界检查
├── data/{raw,processed}/    # 原始快照与处理结果
├── artifacts/               # 数据清单、划分、字表、环境记录
├── runs/                    # 按 run_id 保存配置、日志、结果
├── checkpoints/             # pretrain 与 sft 分开保存
├── report/                  # 报告、图表、生成样例
└── THIRD_PARTY_NOTICES.md   # 借用位置、版本、许可证与改动
```

- [x] ENG-01 建立上述核心模块和统一入口，训练程序使用 `if __name__ == "__main__"`，导入模块时不启动训练或下载。
- [x] ENG-02 配置文件覆盖模型、数据路径、种子、训练和采样参数；运行时保存最终解析后的配置，不能只保存默认配置。
- [x] ENG-03 每次运行使用独立 `run_id`；记录代码提交或源码摘要、配置哈希、数据清单哈希、tokenizer 哈希与环境版本。
- [x] ENG-04 `.gitignore` 排除原始数据、处理缓存、大检查点及运行目录；脚本、配置和清单可提交。
- [x] ENG-05 建立 [开源借用登记](THIRD_PARTY_NOTICES.md)：源码引用、数据版本、具体用途、改动和许可证。
- [x] ENG-06 以 `LLMs-from-scratch` 第 2～5、7 章为主要学习参考，借鉴 nanoGPT 的配置、训练日志与采样组织方式，不混用两个项目的 tokenizer 和检查点格式。

已核对的开源快照：

| 项目 | 用途 | 提交 SHA | 已读取的许可证 |
|---|---|---|---|
| [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch/tree/0311973e2c974b9e0223933103dc400c7a536179) | 主参考：GPT 结构、训练流程和监督微调数据组织 | `0311973e2c974b9e0223933103dc400c7a536179` | [Apache 2.0，代码相关范围以原文为准](https://github.com/rasbt/LLMs-from-scratch/blob/0311973e2c974b9e0223933103dc400c7a536179/LICENSE.txt) |
| [nanoGPT](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291) | 训练配置、梯度累积、生成程序的简洁参考 | `3adf61e154c3fe3fca428ad6bc3818b27a3b8291` | [MIT](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/LICENSE) |
| [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry/tree/b8594f81a89752241442f2ce267d6f66f96704ee) | 唐诗、宋诗原始数据 | `b8594f81a89752241442f2ce267d6f66f96704ee` | [仓库标示 MIT](https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/LICENSE) |

nanoGPT 作者已在 README 中于 2025 年 11 月标注项目较旧并转向 nanochat。本项目使用它作为简洁实现参考，仍需在当前 PyTorch 上验证所借用逻辑。[README 状态说明](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/README.md)

## 4. 准备数据：先划分诗作，再生成训练窗口

### 4.1 数据来源与结构

第一版使用固定快照 `b8594f81a89752241442f2ce267d6f66f96704ee` 的 `chinese-poetry/全唐诗/poet.tang.*.json`。预训练用清洗后训练集的多种诗体；SFT 只用其中四句各五字且能提取到可靠关键词的诗。宋诗 `poet.song.*.json` 可用于后续扩充；宋词、曲和现代译注不混入第一版。若扩充，先跨来源去重并确定全局划分，再开始正式实验。

数据至少核对 `author`、`title`、`paragraphs`，有 `id` 时保留。源文本以繁体为主，有缺字和断句问题。**`paragraphs` 的条数不是诗句数**：例如《帝京篇十首 一》有四个数组元素，按标点切开是八句；《於太原召侍臣賜宴守歲》有两个元素，切开是四句。保留原始字段供溯源，模型正文不混入标题、作者和注释。[语料说明](https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/%E5%85%A8%E5%94%90%E8%AF%97/README.md)、[原始样本](https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/%E5%85%A8%E5%94%90%E8%AF%97/poet.tang.0.json)

### 4.2 快照复核结果（2026-11-18）

| 统计项 | 实测数 | 如何使用 |
|---|---:|---|
| 唐诗 `poet.tang.0.json` 至 `poet.tang.57000.json`，按 1000 递增 | 58 个分片，57,607 条原始记录 | 数据盘点的起点 |
| 通过下面的正文初筛 | 51,945 条；按正文去重后 51,138 条、共 2,594,580 个正文字（不含标点） | 全部诗体的预训练候选，尚未划分 |
| 四句各五字 | 3,976 条；按正文去重后 **3,791 条** | SFT 候选上限；尚未完成近重复分组或筛关键词，未判定严格格律 |
| 标签覆盖 | 57,607 条中无 `keywords` 字段；4,132 条有非空 `tags`，其中 300 条属于 3,976 条五言四句原始候选 | `tags` 混合体裁、选本和主题，如“五言绝句”“唐诗三百首”“写人”，不能直接当作关键词监督 |

复核口径：`paragraphs` 须是非空数组，每项均为非空白字符串。用换行拼接并做 Unicode NFC，按 `[，。！？；,.!?;\r\n]+` 切句；去掉每段首尾空白、忽略空段，每个剩余句子须匹配 `[\p{Unified_Ideograph}〇]+`（Python `regex`）。四句各五字再单独计数。精确去重键是 `''.join(lines)`。这是保守的自动初筛：有 `□`、括注或句中不明符号的条目进入待复查记录，不靠删字通过。3,791 条不等于 3,791 条带关键词的训练样本。

同一快照的 [《唐诗三百首》JSON](https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/%E5%85%A8%E5%94%90%E8%AF%97/%E5%94%90%E8%AF%97%E4%B8%89%E7%99%BE%E9%A6%96.json) 使用同一口径后有 **345 条不同正文，342 条已在主库**；44 条五言四句正文中有 43 条重叠。可选用其中的诗做格式单元样例或评阅者校准；先确认所选诗作及近似变体不在最终测试样本中。整个选集不作为独立测试集，也不重复加入训练。

### 4.3 清洗、去重与划分任务

已落地 [语料与复跑抽核](docs/corpus_audit.md)，保存了 [源文件清单](artifacts/data_manifest.json)、[冻结划分](artifacts/split_manifest.json) 和 [真实统计](artifacts/data_stats.json)。

- [x] DATA-01 获取固定提交的数据快照，生成 `artifacts/data_manifest.json`：来源、提交 SHA、访问日期、实际文件列表、各文件 SHA-256 和数据使用说明。
- [x] DATA-02 原始文件放在 `data/raw/`，处理结果另存；为每首诗保留来源文件、原始 ID、作者、标题和原文，能够回查。
- [x] DATA-03 固定清洗规则：UTF-8、Unicode NFC、空白与标点处理、有效汉字范围。缺字 `□`、乱码、括注进入排除或人工校对记录；可查[《御定全唐詩》四库全书本](https://zh.wikisource.org/zh-hant/%E5%BE%A1%E5%AE%9A%E5%85%A8%E5%94%90%E8%A9%A9_%28%E5%9B%9B%E5%BA%AB%E5%85%A8%E6%9B%B8%E6%9C%AC%29)，若更改正文须记录卷次、异文与依据。
- [x] DATA-04 默认保留繁体。如改为简体，在划分前统一转换，记录转换工具/版本/配置，对提示词和评估应用同一规则，并重新去重。
- [x] DATA-05 分开保存“包含标点的规范文本”和“去标点正文”；用去标点正文的稳定哈希分组，合并跨文件、异名或异作者署名的完全重复诗作。同正文断句不一时核对后选规范版本，并保留各来源索引。
- [x] DATA-06 先对五言四句候选查近重复：同一位置至少三句相同，或去标点正文的字符编辑距离 ≤ 2；同时查候选是否作为完整四句连续出现在更长的训练候选里。把命中的诗及变体归到同一组，人工抽检同名、同首句和边界样本；跨来源按同一规则处理，保存合并理由。
- [x] DATA-07 在去重组层面按约 `90% / 5% / 5%` 划分 train / validation / test，固定种子 `2026`，按组内是否有五言四句候选分层，最终保存具体诗作 ID 和分组 ID。
- [x] DATA-08 冻结验证/测试诗作清单。测试诗作及其变体不得进入任何阶段训练；所有窗口、SFT 样本和数据增强继承原诗所属集合。
- [x] DATA-09 预训练从 train 诗作生成文本流；SFT 只从 train 中符合五言四句基本格式的诗作构造监督数据。两阶段复用训练诗作可以，但需在报告说明。
- [x] DATA-10 先用第 4.2 节口径复跑数据统计。用固定种子人工抽查至少 50 首五言四句候选和 50 首被排除的诗，检查断句、缺字和整首是否完整；不从八句诗截四句凑样本。
- [x] DATA-11 统计原始、清洗、精确去重后的诗数，及各 split 的正文字符量、长度分布、五言四句候选和实际 SFT 样本数。记录规则改动后计数变化的原因。
- [x] DATA-12 断言各 split 的精确正文哈希和组 ID 无交集，并记录近重复检查结果。先划分，后分词、拼接、切窗口和构造 SFT 样本。

**产物：**原始数据快照清单、清洗和去重日志、`train/val/test` 诗作文件、`split_manifest.json`、`data_stats.json`。**验收：**重复运行得到相同划分；测试诗作未进入预训练或 SFT 训练材料；格式筛选可人工复核。

## 5. 实现 tokenizer 与训练数据加载

- [x] TOK-01 只从训练文本建立字符词表；提前加入固定特殊标记，如 `<pad>`、`<unk>`、`<bos>`、`<eos>`、`<wujue>`、`<kw>`、`<kwsep>`、`<sep>`。关键词分隔符也使用固定标记，避免模板字符意外变成 UNK；其 ID 和词表顺序确定后固定。
- [x] TOK-02 保存 `tokenizer.json`：字符与 ID 映射、特殊 ID、规范化规则、版本和 SHA-256；验证和测试出现的未登录字使用 `<unk>`，单独报告其比例。
- [x] TOK-03 让特殊标记作为单一 token 编码，而不是被拆成尖括号和英文字母；普通文本中的同名字符串需要明确转义或拒绝策略。
- [x] TOK-04 验证训练文本编码后解码可还原规范文本；验证特殊标记、繁体字、生僻字、换行和未登录字；不能使用验证/测试内容扩大词表。
- [x] TOK-05 预训练序列采用 `<bos>正文<eos>`；各 split 单独拼接，再取长度 `T+1` 的 token 片段，构造 `input_ids=tokens[:-1]`、`labels=tokens[1:]`。
- [x] TOK-06 记录拼接允许跨训练诗作的上下文，使用 EOS 标出边界；尾部不足一块时采用 padding 并屏蔽相应标签，不拼接其他集合补足长度。
- [x] TOK-07 SFT 使用单样本右侧 padding，保留每个样本的 prompt 长度与有效长度；`<pad>` 和 `<eos>` 使用不同 ID，EOS 保留训练信号。
- [x] TOK-08 检查 `input_ids`、`labels` 形状与索引类型、词表边界、padding 位置、目标偏移和有效 token 数。保存几条可读的 token/label 对照样例。

字符级分词不需要先做现代汉语分词。预训练和微调必须使用同一个 tokenizer；固定词表后扩充语料，需要评估 OOV，不能无记录地改变词表或嵌入尺寸。

**产物：**[tokenizer.json](artifacts/tokenizer.json)、[token 统计](artifacts/token_stats.json)、缓存文件、DataLoader 与 collate 实现。**验收：**训练批次中每个目标 token 与原文对应正确；验证/测试不会参与字表学习。

## 6. 搭建 GPT 并验证核心机制

### 6.1 基线结构

实现与实测见 [模型说明](docs/model_implementation.md) 和 [训练检查记录](artifacts/training_checks.json)；实际字表 8,582，参数量 14,041,344。

```text
token IDs
  → token embedding + learned position embedding
  → 6 × [Pre-LayerNorm → causal self-attention → residual
         → Pre-LayerNorm → MLP(GELU) → residual]
  → final LayerNorm
  → vocabulary projection
  → logits [batch, sequence_length, vocab_size]
```

| 参数 | 基线值 |
|---|---:|
| `n_layer` | 6 |
| `d_model` | 384 |
| `n_head` / `head_dim` | 6 / 64 |
| `d_ff` | 1536，即 `4 × d_model` |
| `context_length` | 256 |
| `dropout` | 0.1，推理时关闭 |
| Q/K/V、attention 输出、MLP 线性层 | 使用 bias |
| LayerNorm | 有可学习 scale/bias，`eps=1e-5` |
| 输出投影 | 无 bias，与 token embedding 共享权重 |
| 位置编码 | 可学习绝对位置嵌入 |

按上表且权重共享，参数量为 `384 × V + 10,745,856`，`V` 为实际词表大小；例如 `V=8000` 时为 `13,817,856`。这是配置推导值，完成实现后还要由程序统计确认。主参考：[GPT 教学实现](https://github.com/rasbt/LLMs-from-scratch/blob/0311973e2c974b9e0223933103dc400c7a536179/ch04/01_main-chapter-code/gpt.py)。

- [x] MODEL-01 使用 `nn.Module`、`nn.Embedding`、线性层、自动微分等 PyTorch 组件搭建模型，初始化时不加载现成大模型权重。固定初始化方案：嵌入/线性层权重可采用 `N(0, 0.02)`、bias 为 0、LayerNorm scale 为 1/shift 为 0；共享权重只初始化一次，额外的残差投影缩放如启用则记录。
- [x] MODEL-02 验证多头拆分与合并维度，检查 `d_model % n_head == 0`，在报告解释 `QKᵀ / sqrt(head_dim)`、softmax 和 V 加权。
- [x] MODEL-03 实现下三角因果注意力；给出“修改未来 token，不改变此前位置 logits”的自动检查，测试时关闭 dropout。
- [x] MODEL-04 实现位置索引与最大长度检查，越界时明确报错；解释 token embedding 与 position embedding 各自的作用。
- [x] MODEL-05 实现两次 Pre-LayerNorm、两条残差、GELU 前馈网络和最终归一化；将结构、维度与配置对应起来。
- [x] MODEL-06 实现输出权重共享，检查两处引用确实为同一个参数，优化器不会重复登记该参数。
- [x] MODEL-07 输入原始 logits 计算交叉熵，先不做 softmax；flatten 时保持 logits 与 labels 对齐，`ignore_index=-100`。[交叉熵接口](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html)
- [x] MODEL-08 如果使用 SDPA 加速，检查布尔掩码语义：SDPA 的 `True` 表示允许关注。无 padding 时可用 `is_causal=True`；需要自定义掩码时，将 causal 与有效 key 位置合成 `attn_mask`，设置 `is_causal=False`，并避免真实 token 的注意力行被全部屏蔽。推理显式传 `dropout_p=0.0`。[SDPA 文档](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [x] MODEL-09 做少量关键机制检查：输出形状、因果性、padding 标签不计损失、手写小样例损失一致、保存加载后输出一致。
- [x] MODEL-10 在仅用于调试的 16～32 首训练诗上尝试过拟合，确认损失明显下降、可复现部分样本；该调试结果单独标注，不能作为泛化成绩。
- [x] MODEL-11 输出实际参数量、参数 dtype 和前向/反向峰值显存；先跑通 FP32 小批次，再启用混合精度。

**产物：**`model.py`、模型配置、参数量表、架构图和关键机制检查结果。**验收：**因果性与损失检查通过，调试小集合可学习，随机初始化来源可追溯。

## 7. 实现通用训练、验证与续训

### 7.1 第一轮建议配置

| 设置 | 预训练起点 | SFT 起点 |
|---|---|---|
| 优化器 | AdamW | 新建 AdamW，不沿用预训练优化器动量 |
| 学习率 | `3e-4` | `3e-5` |
| `betas` / `eps` | `(0.9, 0.95)` / `1e-8` | 同左 |
| 权重衰减 | `0.1`，bias/归一化等一维参数不衰减 | 同左 |
| 微批量大小 | 32 | 32 |
| 梯度累积步数 | 4 | 1，数据足够时再调整 |
| 上下文上限 | 256 | 256，按批内最长样本右侧补齐 |
| 梯度裁剪 | 全局 norm 上限 `1.0` | 同左 |
| 精度 | GPU 支持并验证后用 BF16 autocast | 同左 |
| 学习率计划 | 约 3% warmup，再 cosine 到峰值的 10% | 同左，短运行避免 warmup 占满训练 |
| 初始上限 | 先完成 1 个 epoch 试训，再按验证情况训练，最多 10 个 epoch | 先 1 个 epoch，最多 5 个 epoch |

以上数值是起点，不是效果保证。若达到完整 256 长度，预训练每优化步最多处理 `32 × 4 × 256 = 32768` 个目标 token；padding 等屏蔽位置需扣除。总步数根据实际训练 token、微批量和累积步数计算，学习率调度记录的单位为**优化步**。

- [x] TRAIN-01 实现 `zero_grad → forward → loss → backward → clip → optimizer.step → scheduler.step`，梯度累积窗口内只在正确时机清零与更新。
- [x] TRAIN-02 按有效目标 token 总数归一化损失；梯度累积时处理各微批有效长度差异和最后一个不足窗口，避免把所有批次平均值等权相加。
- [x] TRAIN-03 先检查 `torch.cuda.is_bf16_supported()`；使用 `torch.amp.autocast("cuda", dtype=torch.bfloat16)`，参数维持合适的主精度。若改 FP16，使用 GradScaler，裁剪前先 unscale。[AMP 文档](https://docs.pytorch.org/docs/2.14/amp.html)
- [x] TRAIN-04 支持 CPU FP32 调试；混合精度、SDPA 或 `torch.compile` 分别启用和验证，保存每项开关，避免同时改多项后无法定位差异。
- [x] TRAIN-05 固定 Python、NumPy、PyTorch 和数据加载的种子；记录确定性设置及精度设置，说明跨硬件或版本不保证逐位相同。[复现说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html)
- [x] TRAIN-06 验证时 `model.eval()` 和无梯度推理；累计整套验证集的有效 token 损失，再恢复 `model.train()`。损失异常、NaN/Inf 或零有效标签时立即保存诊断并停止该次运行。
- [x] TRAIN-07 日志至少保存：epoch、优化步、有效 tokens seen、训练/验证 NLL、学习率、梯度 norm、tokens/s、显存峰值和耗时；区分有效 token 与填充位置。
- [x] TRAIN-08 检查点包括：模型、优化器、调度器、需要时的 scaler、epoch/step、随机数状态、数据顺序/位置、配置、词表哈希和数据清单哈希。`best`、`last` 分开保存；将 NumPy 等随机数状态转换为可兼容保存/加载的 tensor 或基础类型，并验证当前版本 `torch.load` 的加载选项与保存格式匹配。[检查点教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [x] TRAIN-09 做一次短程中断与恢复检查：优化步、学习率、数据位置和日志连续；固定条件下与未中断运行比较，记录数值容差。仅加载权重属于重新开始训练，不能冒称完整续训。
- [x] TRAIN-10 用 50 步预热和约 200 步计时估算速度，GPU 计时前后同步；据此计算完整运行时间与磁盘需求，再决定正式训练预算。

**产物：**可配置训练循环、日志、检查点、续训检查记录。**验收：**短程训练稳定、验证口径正确、中断后可恢复，不依赖手工改代码继续训练。

## 8. 完成并展示预训练

实测证据见 [预训练记录](report/pretraining.md)。RTX 4080 SUPER 从随机初始化跑完 10 轮，验证 NLL 9.1361 → 4.9540；SFT 只使用登记的最佳检查点。

- [x] PRE-01 保存随机初始化的基线样例和固定评估输入；标记模型阶段 `random_init`，保存种子和 tokenizer 哈希。
- [x] PRE-02 在正式 train 集上运行自监督预训练。目标是预测真实下一个 token；保存随机初始化至训练结束的配置与运行记录。
- [x] PRE-03 每约 0.5～1 个 epoch 评估固定 validation 集，按验证 NLL 保存最佳检查点；连续 3 次验证无改善可提前停止，最小改善量写入配置。
- [x] PRE-04 定期对同一组短前缀采样，保存早期、中期与最佳检查点的完整输出，包括失败样例，避免只保留最后挑选的好诗。
- [x] PRE-05 绘制训练/验证 NLL 随有效 tokens seen 的曲线；若训练降、验证升，优先检查过拟合与重复数据，再调整训练时长和正则化。
- [x] PRE-06 输出预训练验证 NLL 和 `PPL = exp(NLL)`，说明 tokenizer、样本和损失覆盖范围；不用不同词表或不同掩码下的 PPL 直接比较优劣。
- [x] PRE-07 冻结并登记 `pretrain/best`：模型结构、词表、数据版本、种子、训练 token 总数、实际 GPU 时间和验证结果，作为 SFT 唯一初始化来源。
- [x] PRE-08 若效果不足，按“数据与标签检查 → 增加干净语料 → 调整训练预算 → 小范围改模型”的顺序处理，记录每次只改哪些因素。

**产物：**预训练程序、`pretrain/best` 与 `last`、日志曲线、阶段生成样例和说明。**验收：**相对随机初始化在同一验证口径下有所学习，并如实展示泛化和格式问题；失败结果同样保留，不能用下载的权重替换该阶段证据。

## 9. 构建监督数据并进行条件生成微调

已完成三次 SFT，见 [监督抽核](docs/sft_audit.md) 与 [微调实测](report/finetuning.md)。严格按验证集选了种子 2027 用于演示；测试评估汇总三个种子。

### 9.1 样本约定

建议序列形式：

```text
<bos><wujue><kw>霜<kwsep>孤舟<sep>野渡凝秋霜，孤舟繫短橈。疏鐘催客醒，殘月下寒潮。<eos>
```

这是构造示例，只展示序列格式，不纳入数据集。`<wujue>` 在协议里表示四句各五字；训练正文用“逗号、句号、逗号、句号”，每句五个汉字。每个指定词须在诗的某一句内连续出现，同时评估用得是否自然。

从诗中抽取意象词是**弱监督**，它让模型练习围绕给定用词写作。词出现在诗里，只表明它是候选，更不意味着整首诗的主题已经标好。九歌（Jiuge）的[论文](https://aclanthology.org/P19-3005/)可参考关键词输入设计；本项目自行制作“关键词 → 诗”的配对。

- [x] SFT-01 训练配对只从 train 的五言四句诗构造；对 validation/test 复用冻结的关键词规则并保持原诗分组。第 10 节另设独立编写的无参考诗题。
- [x] SFT-02 第一版把训练诗里的 1～2 字意象词做成白名单（如“霜”“孤舟”），记录 `lexicon`、停词和版本；在同一句内找实际出现的词，优先保留长词，去掉重复词及其被包含的短词，如“孤舟”和“舟”不算两个条件。不匹配的诗跳过；用固定种子从剩下的词取 1～3 个并打乱其顺序（首次可抽 1/2/3 等概率，再按实际词数截取）。
- [x] SFT-03 保存配对 JSONL：`poem_id`、`group_id`、`split`、`form`、`keywords`、`label_source`、`prompt`、`response`、来源和处理版本，并记录关键词抽取规则和词表的哈希。
- [x] SFT-04 第一版每诗只取一个配对，标题/作者只作元数据；分层抽查至少 50 条训练配对，覆盖 1/2/3 词及高低频词，记录是否是意象、是否真的出现在诗中、能否合理引出诗意。报告各 split 的配对数量、词数分布、无匹配与未登录词比例。
- [x] SFT-05 实现仅对正文及 EOS 计算损失的 labels：**目标 token 属于 prompt 或 PAD 的位置置为 `-100`**；按右移后的目标位置判定，不把正文第一个字误屏蔽。
- [x] SFT-06 保存并检查一条带 token ID、目标字符、prompt 边界和 labels 的样例；断言有效目标不为空、正文完整、EOS 仍参与损失。超长样本按规则过滤，保留过滤计数。
- [x] SFT-07 核对预训练检查点与 tokenizer/config 哈希，加载自己的 `pretrain/best`；新建微调优化器与学习率计划，保存初始化检查点的标识。
- [x] SFT-08 按第 7 节初始配置微调，优先观察固定条件下的验证正文 NLL、格式与关键词表现；验证集选择检查点，测试集留到选择完成后使用。
- [x] SFT-09 至少从同一个预训练检查点运行 3 个 SFT 训练种子，例如 `2026/2027/2028`；若只完成 1 次则注明证据限制。共享预训练的重复不能表述为三次独立从零训练。
- [x] SFT-10 保留微调前后在同一条件、同一采样设置下的输出；检查条件相关性是否改善，以及重复、照抄或语言质量是否退化。

参考书第 7 章可用于理解数据整理，但其所检查的基础 `custom_collate_fn` 主要屏蔽 padding，未实现本项目要求的全部 prompt 损失屏蔽，且示例会加载现成 GPT-2 权重。因此需要按本节改造。[已核对的微调代码](https://github.com/rasbt/LLMs-from-scratch/blob/0311973e2c974b9e0223933103dc400c7a536179/ch07/01_main-chapter-code/gpt_instruction_finetuning.py)

**产物：**监督数据、关键词规则、掩码示例、SFT 程序、检查点和前后对比。**验收：**初始化来自本项目预训练；标签位置正确；任务评估与训练诗作隔离。

## 10. 评估：分别回答“像不像诗”“是否听条件”“是否在背诵”

### 10.1 比较对象与固定协议

已完成 [14M 封存测试](report/evaluation_baseline.md) 与 [92M 扩容版同尺寸微调前后对照](report/evaluation_expanded.md)。两个版本各有 30 组[匿名盲评](report/blind_review/README.md)材料；优先评 92M，当前尚未收到两位独立评阅者的评分。

| 编号 | 对象 | 用途 |
|---|---|---|
| R0 | 同结构随机初始化模型 | 检查训练确实带来学习；保留小规模样例与 NLL 即可 |
| R1 | 本项目预训练模型 | 微调前基线 |
| R2 | R1 初始化后得到的 SFT 模型 | 主结果 |
| R2-K | R2 收到另一组关键词，仍按原诗题评分 | 检查改换关键词是否影响写作结果 |
| R3，可选 | R2 加格式约束解码 | 检验解码规则带来的收益，单独报告 |

默认固定采样：`temperature=0.8`、`top_k=50`、`top_p=1.0`、`max_new_tokens=64`；遇 EOS 停止。R1/R2 使用相同 tokenizer、prompt、参数、生成预算与采样种子。预训练没见过完整的关键词模板，R1→R2 的差异包含学会使用模板的收益；R2-K 单独检验指定词是否有作用。

- [x] EVAL-01 冻结两套测试诗题并分开报告。**主测试：**由 Codex 独立编写 50 道无参考诗题，覆盖 1/2/3 个关键词及不同意象（如“霜、孤舟”），优先用训练见过的词、未出现过的组合。**补充测试：**从 test 诗作选 100 首，每组近重复至多一首，提取条件，保留原诗用于 NLL 和错误分析。两套均固定 ID、词序和来源，调参另用验证诗题。
- [x] EVAL-02 按排序后的关键词集合检查与训练配对的组合重叠；同时记录词是否在训练配对中出现及字是否在 tokenizer 中。新组合结果和含未登录字、超长等边界请求分开列；凡已列入测试的请求，都保留在对应汇总的分母里。
- [x] EVAL-03 对 R1/R2 每条条件使用 3 个固定采样种子，例如 `11/22/33`，保留全部原始输出、运行参数和失败原因；采样种子重复不等于训练种子重复。
- [x] EVAL-04 在同一组监督验证/测试样本上，对 R1/R2 计算相同正文+EOS 掩码的 NLL；与预训练文本流的 NLL 分表展示，避免混用统计口径。
- [x] EVAL-05 实现第 10.2 节的解析和指标。只评模型新生成的内容；额外说明、数字、英文字母或控制标记记为不合规。原始指标不使用补标点、裁切、重试或从多首里挑选的结果。
- [x] EVAL-06 对格式合格输出，查预训练与 SFT 的所有训练正文：标记整诗精确重复、完整四句连续出现在长诗、和训练集五言四句的近重复（规则同 DATA-06）。记录命中的 `poem_id`、原文位置与判定理由；此检查只覆盖已知训练材料。
- [ ] EVAL-07 从主测试按词数分层，用固定种子预抽至少 30 条诗题，每题使用相同预定采样种子的 R1/R2 输出配对。两名评阅者独立评分，看诗题与匿名原始输出；打乱两首呈现顺序，不筛掉错格或漏词作品。按第 10.4 节汇总。[分发包与回件校验](report/blind_review/README.md)已验证，实际回件 0/2。
- [x] EVAL-08 实测每首诗耗时、生成 tokens/s 和显存峰值；统一 prompt/输出预算，GPU 计时同步并区分预热与正式测量。CPU 推理另外记录。
- [x] EVAL-09 对比例指标同时保存分子、分母；多个 SFT 训练种子先各自算指标，再汇总均值与标准差。若给区间，按 prompt 分组重采样，不能把同 prompt 的多次采样当成完全独立输入。
- [x] EVAL-10 将机器可读指标写入 `metrics.json`，逐条样例写入 JSONL；结果带 checkpoint、数据/tokenizer 哈希、prompt 集版本与生成配置。
- [x] EVAL-11 汇总成功与失败案例：格式错误、漏关键词、语义牵强、重复、照抄、OOV、过早 EOS 等，解释可能原因，不只展示最好样例。
- [x] EVAL-12 为主测试中有可替换条件的诗题跑 R2-K：固定选一组词数相同、尽量同长度、与原词不重叠的关键词，其他设置不变。用原关键词算命中率，与 R2 在同一批诗题上配对比较；报告有效配对数。这个对照检验词汇控制，是否自然融入仍由人评分。

### 10.2 自动指标口径

用 tokenizer 保留原始输出和停止原因；解析时只取新生成正文，停止 EOS 不计作正文。统一 NFC 和 CRLF，去掉整体首尾空白。每句只允许 `[\p{Unified_Ideograph}〇]+`，句间分隔允许一个 `，。！？；,.!?;` 中的标点（可紧跟一个换行）或一个换行，末句可以省略标点。先验完整串是否合规，再分句；句内空白、`□`、空句、其它字符报错。四句各五字另计；标点是否严格符合训练模板、是否生成 EOS，可作诊断。

每条诗题的每个采样种子算一次请求。格式、关键词和联合成功的分母都是全部请求，解析失败或拒绝输入计为不成功。BLEU、ROUGE 对单首参考诗的相似程度不作为创作成败依据。自动打分用于可核对的规则，人工评分用于诗意和用词；大模型裁判如使用，只作附录，参考[Ma 等，ACL Findings 2026](https://aclanthology.org/2026.findings-acl.836/)。

| 指标 | 定义 | 解读与限制 |
|---|---|---|
| 四句各五字合规率 | 解析成功且恰好四个五字句的请求数 / 全部请求数 | 只反映外形；与 [CharPoet](https://aclanthology.org/2024.acl-demos.30/) 的评估思路相近 |
| 关键词全命中率 | 每个指定词都在至少一个句子内逐字连续出现的请求数 / 全部请求数 | 保留统一字形，不跨句匹配；可另报每题命中的词数，区分漏词和全漏 |
| 格式与关键词联合成功率 | 同时满足前两项的请求数 / 全部请求数 | 主要自动指标；高分仍可能来自生硬堆词 |
| 训练正文匹配率 | 整诗精确匹配数 / 四句各五字输出数；长诗片段命中、近重复分别报 | 附命中的原文与样例，报告解析合格数；也要检查同一首生成诗在多个题目中复现 |
| 正文 NLL / PPL | 对固定参考正文及 EOS 计算平均 token 负对数似然及其指数 | 对照时 tokenizer、样本、条件和掩码必须相同；用于补充测试 |
| 成本 | 每首延迟、tokens/s、显存、训练 GPU 小时 | 固定测量协议，注明输出长度和测量设备 |

### 10.3 韵脚诊断：平水韵偶句同韵

采用[《平水韵》](https://zh.wikisource.org/zh-hant/%E5%B9%B3%E6%B0%B4%E9%9F%BB)作本项目的韵部口径。已读[固定版本的繁体韵表 JSON](https://github.com/rbnyng/pingshui_rhyme/blob/0d6b05a932ff1828d8403e7a0de67dd1f6bd81d0/pingshui_rhyme/data/organized_ping_ze_rhyme_dict.json)，核对上平、下平、上声、去声、入声各有 `15/15/29/30/17` 部。原文件 SHA-256 为 `42bcfcd06ea040883fb8403f9ebdd4ff0db7a6183e03d0a5e6bf9f0e7bb7386c`。导入时逐组只取汉字，去掉“入聲十七洽”末尾混入的英文标记；每字保留全部可能韵部。

- [x] EVAL-13 对四句各五字的输出取第 2、4 句末字。两字所有可能韵部搭配都落在同一平声韵部，判为**合**；韵部均可查、但没有共同平声韵部，判为**不合**；有缺字或不同读法会改变结论，判为**待判**。同一韵字重复做额外标记。复核例：“流/樓”同属下平十一尤，“流/春”不同韵，“行/庚”需按上下文定读音。
- [x] EVAL-14 分模型报告合、不合、待判的数量；给出 `合 / (合 + 不合)` 和 `(合 + 不合) / 四句各五字输出数`，分母为零时标记不适用。抽查通过和待判样例。首句是否入韵、整首平仄和拗救留给严格格律实验；此处只判断偶句是否同平声韵。

把这条规则试用于第 4.2 节的 3,791 条候选：**1,824 条合、828 条不合、1,139 条待判**，其中缺字形映射 161 条、不同读法影响结论 978 条。这里按源文本字形直接查表，未进行人工释读；这是语料诊断，既不是五言律绝计数，也不是模型成绩。第一版保留五言四句候选供 SFT 筛选，报告中单列偶句韵脚诊断。

### 10.4 人工盲评：诗写得怎么样

参照[Zhang 与 Lapata，EMNLP 2014](https://aclanthology.org/D14-1074/)的多维评分设计。评阅者各自打 1～5 分，2 和 4 为相邻描述之间；切题时允许语义相关但字面没命中的意象，字面命中由机器另计。

| 维度 | 1 分 | 3 分 | 5 分 |
|---|---|---|---|
| 切题 | 与诗题无关，或只堆指定词 | 大体能看出在回应诗题，但有遗漏或牵强处 | 指定意象自然融入诗中，主题清楚 |
| 通顺 | 多数句读不通或词语乱搭 | 大意可读，有个别用词或句式不自然 | 句义清楚，用字自然、凝练 |
| 连贯 | 各句彼此无关或互相矛盾 | 共享场景，但转折或收尾突兀 | 四句铺陈、转折或收束有联系，能读成完整一首 |
| 意象与意境 | 抽象套话或生硬拼接，看不出场景和情绪 | 有可感的画面或情绪，但平常、发展较少 | 画面具体，情绪或情境由意象自然传达，有余味 |

先用不进入最终测试的样例校准。保留单人原始分、备注及相差 ≥ 2 分的条目，再报告各维均值、分布和配对差值。空输出或明确拒绝按预先约定各维记 1，并单列数量。报告里展示切题但漏词、命中却生硬、形式正确但内容不佳的例子，注明人工评价的样本量与评阅人数。

**产物：**评估程序、冻结的 prompt 集、逐条输出、指标、人工评分表和对照图。**验收：**同一协议能重跑；有完整分母与失败样本；模型效果和解码规则收益分开报告。

## 11. 本地推理与可复现交付

已验证封装权重在新进程中使用 CPU 与 CUDA 推理。固定 2 道预热、10 道计时的[实测](report/evidence/infer-benchmark-14m.json)另注明 GPU 同时有训练进程，不用来定性判断独占加速比。

- [x] INFER-01 实现独立 `generate.py`，参数包括检查点、关键词、体裁、device、种子、temperature、top-k/top-p 与最大新 token 数；加载模型所对应的 tokenizer/config。
- [x] INFER-02 实现自回归采样：只取最后位置 logits，按参数筛选并重归一化，采样后追加 token；`temperature=0` 明确走 greedy 分支，避免除零。
- [x] INFER-03 检查关键词个数、字数、字形规范、未登录字和上下文预算，给出拒绝原因；区分正常 EOS 和达到生成上限，不能无声截掉条件。
- [x] INFER-04 使用 `model.eval()` 与 `torch.inference_mode()`，关闭所有 dropout；移除显示用控制标记时不修改模型生成的正文。
- [x] INFER-05 支持 `--device cpu` 的 FP32 推理与 GPU 推理，输出加载耗时和生成耗时；若 GPU 不可用，给出明确提示或记录回退，不能误报 GPU 测速。
- [x] INFER-06 如果增加强制四句、强制标点或押韵约束，增加显式开关，保留原始解码模式并单独评估；不能把强制格式当成模型自行学会格式。
- [x] INFER-07 打包最小推理目录：模型权重、config、tokenizer、版本说明、运行命令和校验值，在新进程中加载测试。
- [x] INFER-08 在 README 中给出环境检查、数据处理、预训练、续训、SFT、评估和推理命令；每条命令需实际跑过，并注明预期产物。

入口命令详见 [README](README.md)。以下是 14M 基线的调用：

```bash
python scripts/prepare_data.py --config configs/data.yaml
python scripts/build_tokenizer.py --config configs/data.yaml
python scripts/pretrain.py --config configs/pretrain.yaml --run-id pretrain-tang-v1 --stop-after 87
python scripts/pretrain.py --config configs/pretrain.yaml --resume checkpoints/pretrain/pretrain-tang-v1/last.pt
python scripts/build_sft_data.py --config configs/sft.yaml
python scripts/finetune.py --config configs/sft.yaml --init checkpoints/pretrain/pretrain-tang-v1/best.pt --seed 2027 --run-id sft-tang-2027
python scripts/evaluate.py --config configs/eval.yaml --device cpu
python scripts/generate.py --checkpoint checkpoints/sft/sft-tang-2027/best.pt --keywords 春 雨 --device cuda
```

多个 SFT 种子的最佳模型由验证集选择。重复实验请换新的运行 ID，并相应更新评估配置中的 checkpoint 路径。

## 12. 必做完成后再考虑的增强任务

单卡规模、长训练和扩充语料见[实测对照](report/scaling.md)，保守平仄见[格律诊断](report/evaluation_baseline.md#平仄诊断的适用范围)。未主张稳定提升的项目不做确定性断言。

| 增强项 | 要额外完成的工作 | 主要风险 |
|---|---|---|
| 七言绝句 | 七字句数据筛选、体裁条件、分别评估 | 样本分布混淆与体裁控制退化 |
| 藏头诗 | 构建首字约束数据或解码规则；检查四句首字 | 解码强制满足不等于模型理解 |
| 严格五言律绝 | 按王力《诗律》选平起/仄起、首句入韵/不入韵的常格；逐字核对平仄、检查粘对及用韵。若把律绝设为训练目标，按同一规则重筛 SFT 语料，并人工抽审 | 普通话读音不能直接替代古韵；人工复核多音字、拗救和首句借邻韵，报告规则未覆盖的例外 |
| tokenizer 对照 | 在同一划分、相近参数/训练预算下比较字符级与训练集学习的 BPE | token 数、词表和 NLL 口径改变，比较需重新设计 |
| LoRA 对照 | 基于同一预训练检查点，记录目标层、rank、可训练参数量、预算和合并方式 | 不能用别人的大模型 LoRA 结果替代本项目预训练证据 |
| 推理优化 | KV cache、SDPA、编译或量化分别验证质量和速度 | 位置索引、mask、精度差异可能导致错误 |
| 交互界面 | 使用已验证的推理入口，显示关键词和采样设置；只增加实际需要的依赖 | 界面工作挤占训练与报告时间 |

- [ ] OPT-01 完成必做闭环后，按剩余时间只选择少量增强；每项记录假设、对照、指标和预算，不同时堆叠多个变量。[已做增强的实验台账](docs/scale_protocol.md#增强范围与实际预算opt-01)已补齐；后续增强等必做盲评完成再定。
- [ ] OPT-02 格律实验分别报告基本字数、韵部和平仄指标，记录规则未覆盖或有争议的样例；请有相关知识的人抽查。[自动诊断与文献复核](report/prosody_source_audit.md)已完成，仍待独立读者[抽查](report/blind_review/meter_review.md)。
- [x] OPT-03 报告同语料等训练量的规模对照、唐宋三种子微调和同模型的解码对照；如实记录扩容版未稳定超过短训版的条件命中。

## 13. 课程报告、图表和提交材料

已编写[可编辑课程报告](report/experiment_report.md)并导出 [PDF](report/Orchid课程报告.pdf)。[独立环境验收](docs/local_delivery.md)已通过；扩容成绩继续记录。

- [x] REPORT-01 按评分要求写知识梳理：下一个 token 目标、字符级分词、嵌入、位置编码、因果注意力、残差与归一化、优化、SFT 掩码、自回归采样；每项联系本项目代码和实际例子。
- [x] REPORT-02 给出实际模型配置、参数量、训练硬件、精度、数据数量/划分、预训练和微调超参数、种子、停止条件及实际时间。
- [x] REPORT-03 至少绘制：模型结构图、预训练训练/验证曲线、微调验证曲线、微调前后任务指标图；数据来自运行日志，保留 CSV/JSON 和绘图脚本。
- [x] REPORT-04 展示固定条件下 R1/R2 对照、阶段预训练样例、正常结果及失败结果，清楚标注采样参数和解码约束。
- [x] REPORT-05 分析限制：小语料、小参数、弱监督关键词、繁简/生僻字、记忆、格律覆盖、人工评分规模；结论不超出实际完成的测试。
- [x] REPORT-06 写个人体会与分工：哪些机制真正实现、借用了什么、遇到什么错误、如何定位、哪些改动有效或无效。
- [x] REPORT-07 列出所用资源的网址、作者/项目、访问日期和代码版本，遵循所借用文件的许可证与署名要求；核对报告引用与实际使用一致。
- [x] REPORT-08 整理源码、配置、复现说明、必要模型文件和下载入口；公开原始数据可提供固定版本与下载地址，按课程要求处理大文件。
- [x] REPORT-09 在独立目录做最终验收：按 README 安装、加载权重、生成一首、运行评估；短程训练验证入口有效，不把再次完整训练设为交付前提。

## 14. 执行顺序、预算与最终验收

### 14.1 建议按里程碑推进

| 顺序 | 完成内容 | 进入下一阶段的条件 |
|---|---|---|
| M0 | 环境固定、代码和开源登记 | 版本与 GPU 检查通过 |
| M1 | 数据清洗、去重、全局划分、tokenizer | 数据清单完整，集合边界与编码检查通过 |
| M2 | GPT、损失、训练循环、续训 | 因果性通过，调试小集合可学习，续训有效 |
| M3 | 正式预训练 | 有自训练的最佳检查点、曲线、原始生成样例 |
| M4 | 监督数据、掩码、SFT | 条件与正文对应，微调初始化与损失位置正确 |
| M5 | 固定测试、本地推理 | 指标与样例可追溯，CPU/GPU 入口可使用 |
| M6 | 报告和打包 | 五类评分证据齐全，按文档可复现 |

建议先分配：环境/工程与数据约 1～2 个工作日，模型与训练调试约 1～2 日，预训练/SFT 与评估约 1～2 日，报告与交付约 1～2 日。这是工作量估计，需按课程截止时间和人员熟悉程度调整。

初始 GPU 预算上限可设为：正式预训练 6 小时、全部 SFT 2 小时、自动评估 1 小时、可选对照 3 小时，共 12 GPU 小时。**这些是计划上限，未实测，不代表训练耗时或生成质量承诺。** 应先用 TRAIN-10 的吞吐测量估算；超预算时缩小语料/模型或训练轮次，并保留真实配置和停止原因。人工评分和报告时间另计。

### 14.2 结束前逐项确认

- [x] FINAL-01 自建 GPT 的结构、配置和实际参数量明确，核心机制能解释并有检查证据。
- [x] FINAL-02 预训练确实从随机初始化开始，有真实训练日志和本项目检查点。
- [x] FINAL-03 微调确实加载本项目预训练权重，监督数据、prompt 掩码和有效标签已核对。
- [x] FINAL-04 全局测试诗作未进入任何训练阶段，tokenizer 未使用验证/测试内容学习字表。
- [ ] FINAL-05 有相同协议下的微调前后对照：独立诗题的联合成功率、偶句同韵诊断、人工分项评分及训练正文匹配；报告失败案例，约束解码收益单列。自动项和案例已完成，等待 EVAL-07 两份真人评分。
- [x] FINAL-06 推理包可在新进程加载，README 的命令、路径、依赖和下载入口已逐项验证。
- [x] FINAL-07 报告覆盖 35%/20%/20%/20%/5% 各项，全部图表数字能追溯到实际运行。
- [x] FINAL-08 开源来源、固定版本、许可证和资源网址齐全；未完成的增强和未验证结论明确标注。

## 15. 本次联网核对的主要资料

工程与环境资料核对日期为 2026-11-17；诗歌语料、韵表和评估章节复核日期为 2026-11-18。代码与数据快照见第 3、4、10 节。论文用于选择方法，未复现论文结果。

| 资料 | 在本清单中的用途 |
|---|---|
| [LLMs-from-scratch：GPT 实现](https://github.com/rasbt/LLMs-from-scratch/blob/0311973e2c974b9e0223933103dc400c7a536179/ch04/01_main-chapter-code/gpt.py) | 核对嵌入、注意力、Pre-LN、残差与输出结构 |
| [LLMs-from-scratch：监督微调实现](https://github.com/rasbt/LLMs-from-scratch/blob/0311973e2c974b9e0223933103dc400c7a536179/ch07/01_main-chapter-code/gpt_instruction_finetuning.py) | 核对序列右移、padding 和本项目需要额外实现的 prompt 掩码 |
| [nanoGPT README](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/README.md) | 字符级小 GPT、可配置训练流程及维护状态 |
| [chinese-poetry 数据说明与样本](https://github.com/chinese-poetry/chinese-poetry/tree/b8594f81a89752241442f2ce267d6f66f96704ee/%E5%85%A8%E5%94%90%E8%AF%97) | 数据路径、繁体字形、字段和句读方式 |
| [同版本《唐诗三百首》JSON](https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/%E5%85%A8%E5%94%90%E8%AF%97/%E5%94%90%E8%AF%97%E4%B8%89%E7%99%BE%E9%A6%96.json) | 核对与主语料的重叠，作为校准样例来源 |
| [PyTorch 官方版本安装表](https://pytorch.org/get-started/previous-versions/) | 确认 CUDA 12.6 构建的安装源 |
| [NVIDIA CUDA 兼容说明](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html) | 区分驱动兼容性与 Python 依赖 |
| [PyTorch 2.14 CrossEntropyLoss](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html) | 原始 logits、标签索引和 `ignore_index` |
| [PyTorch 2.14 SDPA](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html) | 因果掩码、布尔掩码语义、推理 dropout |
| [PyTorch 2.14 AMP](https://docs.pytorch.org/docs/2.14/amp.html) | BF16/FP16 的混合精度训练接口 |
| [PyTorch 2.14 可复现性说明](https://docs.pytorch.org/docs/2.14/notes/randomness.html) | 随机性控制和跨平台复现限制 |
| [PyTorch 检查点保存/加载教程](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html) | 区分权重加载与完整训练恢复；教程当前站点版本较新，具体接口以本机 2.14.1 为准 |
| [Hu、Sun，LREC 2020：Generating Major Types of Chinese Classical Poetry in a Uniformed Framework](https://aclanthology.org/2020.lrec-1.573/) | GPT-2 类结构用于诗歌生成与形式控制的相关工作 |
| [Guo 等，ACL 2019：Jiuge](https://aclanthology.org/P19-3005/) | 关键词驱动的古诗创作及关键词预处理参考 |
| [Zhang 与 Lapata，EMNLP 2014：Chinese Poetry Generation with Recurrent Neural Networks](https://aclanthology.org/D14-1074/) | 参考第 4 节实验中的 Human Evaluation 部分，设计多维人工评分 |
| [Yu 等，ACL 2024：CharPoet](https://aclanthology.org/2024.acl-demos.30/) | §5.3–5.4 分开评估字数格式与内容质量；其模型继承已有预训练能力，论文成绩不直接作为本项目验收线 |
| [Ma 等，ACL Findings 2026：Capabilities and Evaluation Biases…](https://aclanthology.org/2026.findings-acl.836/) | 诗歌评估维度及大模型裁判偏差，支持保留人工评价 |
| [王力《诗律》](https://www.aisixiang.com/data/98039.html) | 核对古绝与律绝的区别、五言律绝常格及边界 |
| [繁体《平水韻》原表](https://zh.wikisource.org/zh-hant/%E5%B9%B3%E6%B0%B4%E9%9F%BB)；[繁体 JSON 的来源说明](https://github.com/rbnyng/pingshui_rhyme/blob/0d6b05a932ff1828d8403e7a0de67dd1f6bd81d0/README.md) | 核对偶句同韵口径；采用 JSON 时登记版本、清洗和来源 |

这些资料支持方案的可行性和实现选择，不能作为 Orchid 已达到相同效果的证据。实际训练完成后，报告应使用本项目自己的数据统计、指标和样例。
