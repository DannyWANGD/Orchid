# Orchid 发布整理与复现

## 本次删改清单

| 动作 | 文件或目录 | 理由 |
|---|---|---|
| 取消 Git 跟踪，保留本地文件 | `report/blind_review/reviewer_1.csv`、`reviewer_2.csv` 及 `expanded/` 中的同名文件 | 这些文件是组织者材料。同一文件名曾被用于空表和已填写回件，重建表改用 `runs/`。已提交的历史版本检查了评分和备注列，四份均为空表 |
| 删除过期快照 | `report/evidence/blind-review-readiness.json` | 记录发放状态和当时零回件，实际评阅证据见现行汇总 |
| 删除三份旧演示输出 | `report/evidence/demo-92m/{selected-cuda,selected-poem-keywords,selected-poem-raw}.json` | 对应换版前的同一组演示词。旧版正式对照、原课程包指纹和新版本逐题测试足够复查 |
| 精简依赖 | `requirements.txt` 从 182 项减为 44 项，新增 13 项的 `requirements-cpu.txt` | 去掉项目未调用的 TensorFlow、Keras、Jupyter 等，保留 PyTorch CUDA 12.6、训练和报告所需的依赖闭包。CPU 推理用同版本 PyTorch 的 CPU 构建 |
| 改打包和复评入口 | `scripts/create_delivery.py`、`scripts/evaluate_prosody.py`、`tests/test_prosody.py` | 移除搜寻本机空白评分表的代码。包内附量表和已发布的汇总。冻结测试可读取导出的权重包，校验其沿袭的 checkpoint 哈希 |
| 改环境检查及速度测试 | `scripts/check_env.py`、`scripts/benchmark_generation.py`、`orchid/evaluation.py` | 指定所用的依赖文件，测自由或默认模式，并记录中位数、p95 和显存峰值 |
| 更新复现文档与结果 | `README.md`、`docs/local_delivery.md`、`report/blind_review/README.md`、`report/prosody_optimization.md`、`report/release_evaluation.md` | 对齐 CPU 快速运行、最终训练、盲评重建与固定测试的实际命令 |

本地回件 `report/blind_review/reviewer_2.csv` 与 `reviewer_2_completed.csv` 已逐字节核验，两份都留在原处。`.gitignore` 覆盖组织者表、回件和映射。盲评分发与汇总见[组织者说明](../report/blind_review/README.md)。

### 本地移出的过程材料

以下项目移至同级恢复目录 `../Orchid-cleanup-recovery-20261121/`，其中 `cleanup-manifest.json` 记录相对路径与字节数，共 1,130,240,915 字节。恢复单项可按清单移回原路径。

- 四个未被 Git 使用的旧临时包 `.git/objects/pack/tmp_pack_{4eQvJH,U8nDgH,lL2lXb,g3yHaM}`。
- 初稿匿名包 `runs/rubric-v2-draft.pTOD4p`。
- 早期短测 `runs/fixed-eval-smoke-cpu`、`runs/fixed-eval-smoke-keyword`、对应的 `fixed-eval-smoke-keyword.log` 和 `runs/delivery-smoke`。
- 无当前复现入口引用的额外学习率试跑 `runs/sft-lr3e4-2026` 和对应的 `checkpoints/sft/sft-lr3e4-2026`。

训练或复核需要的 `.venv`、唐宋源语料、固定处理语料、正式 checkpoints、评阅回件和旧课程包继续供本机使用。VS Code 正在读的 C++ 索引由编辑器管理。以上运行资源均在发布包外。

## 发布文件

- **`Orchid-source.zip`** 含模型、训练、推理及评估代码，固定配置和语料指纹，以及可读报告与封存证据。适合阅读和从头训练。
- **`Orchid-model-kit.zip`** 含 92M 最终推理权重、字表、韵表、命令入口和复测所需源码，可以在独立目录用 CPU 或 CUDA 推理。
- **`Orchid-release-verification.json`** 记录两包 SHA-256、依赖安装、解压推理和重建测试结果。源语料从固定公开版本取得。发布时一并保留[第三方登记](../THIRD_PARTY_NOTICES.md)和[韵表许可](../data/lexicon/pingshui-LICENSE.txt)。

原始项目源码的许可证与署名、模型权重的公开授权范围由项目作者确认。确认后把授权文本纳入相应发布包。[GitHub 许可证说明](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)

## 复现路径

在有已导出最终权重的源码工作目录中构建两种包。Git 源码包以已提交版本为准。

```bash
mkdir -p dist
git archive --format=zip --prefix=Orchid/ --output=dist/Orchid-source.zip HEAD
python -E scripts/create_delivery.py \
    --bundle dist/orchid-92m-prosody --output dist/Orchid-model-kit.zip
```

按[快速开始](../README.md#试用-92m-生成模型)安装并验证 CPU 或 CUDA 推理。无需启动训练即可复测生成速度。完整规则复测先按 README 的唐诗预处理、字表、唐诗监督集、宋诗快照、唐宋处理与监督集步骤准备数据，然后用[评估命令](../report/release_evaluation.md#复跑命令)重放封存测试。新训权重按[训练与选模](../report/prosody_optimization.md#本地复现)生成自己的验证和测试证据。

## 展示与报告更新

新版源码加入 `demo/`、回环服务、隔位重字排序、36 道补测题、92M AI 盲评汇总和 LaTeX 工程。离线网页使用冻结样例，实时页与命令行共用原推理代码。新版导出为 `Orchid-source-demo.zip` 和 `Orchid-model-kit-demo.zip`，模型包同时附带演示与报告源码。署名封面使用独立的 `cover.local.tex`，已由 Git 和模型包的文件筛选规则排除。编译、补测和图表记录见[报告构建说明](../report/latex/README.md)。
