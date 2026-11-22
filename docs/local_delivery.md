# 本地交付和验收

这里记录原课程结项。清单指纹、旧 PDF 和构建入口对应 Git 提交 `fd27236`。当前课程报告与模型包使用现行发布入口。[结项时的源码与文档](https://github.com/DannyWANGD/Orchid/tree/fd27236156654c8038c64db1e6fc8cfbe36961d8) · [当前发布与复现](open_source_cleanup.md)

独立临时目录用 `git archive` 导出源码和固定清单，使用新装的 Python 3.11.16 虚拟环境执行 182 个 pin 校验、`pip check` 与 CUDA 前向/反向。随后从固定上游分片重建 Tang train/validation/test、训练字表和监督配对；从导出的独立推理包在 CPU 生成一首；使用复制的已登记检查点运行三题版评估。最后按 README 的续训方式，用真实训练数据在 CPU FP32 上从优化步 1 恢复到步 2。没有用模拟数据替代这些验收。[机器可读记录](../report/evidence/independent-acceptance.json)。

GitHub 提供源码、配置、报告和图表。模型 ZIP 包含必要权重、配置、字表和校验清单，用于在独立目录推理。完整唐诗和宋诗的下载路径见固定数据清单。[数据约定](data_contract.md)

14M 基线包的本机路径为 `dist/Orchid-course-baseline.zip`。其中权重、字表、配置与归档内容都已逐项校验。在独立目录解压后可用 `--checkpoint model` 运行推理。[归档校验](../report/evidence/delivery-baseline.json) · [解压后生成记录](../report/evidence/independent-archive-generation.json)

结项课程包 `dist/Orchid-course-prosody.zip` 使用 92M 韵律续训权重、离线韵表和默认四句五字押韵解码。解压至独立目录后，用 `--checkpoint model` 即可生成。结项课程包使用空白评分表。[原解压验收](../report/evidence/delivery-92m-prosody.json)

结项验收核对了清单的 119 个条目。实测包括 182 个依赖固定项、CUDA 前后向和 18 项测试。语料的源文件哈希、两套数据划分、训练字表、检查点哈希和封存评估已核对。主评阅使用确认的真人 v1 回件与同版 AI 参照，格律采用八首独立 AI 复核。每组条目的证据索引在[结项记录](../report/evidence/final-acceptance.json)。

原 92M 唐宋版对应旧评阅与对照，归档路径为 `dist/Orchid-course.zip`。在独立环境，归档可在 CPU 加载原始模式，也可在 CUDA 跑原版字数和关键词约束。[原归档校验](../report/evidence/delivery-92m.json) · [原解压验收](../report/evidence/delivery-92m-extraction.json)

真人评阅者只使用各自的[独立评分包](../report/blind_review/README.md)。课程 ZIP 含解盲后的自动评估和可复现源码，用于课程交付，不用作盲评发放材料。
