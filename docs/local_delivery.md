# 本地交付和验收

独立临时目录用 `git archive` 导出源码和固定清单，使用新装的 Python 3.11.16 虚拟环境执行 182 个 pin 校验、`pip check` 与 CUDA 前向/反向。随后从固定上游分片重建 Tang train/validation/test、训练字表和监督配对；从导出的独立推理包在 CPU 生成一首；使用复制的已登记检查点运行三题版评估。最后按 README 的续训方式，用真实训练数据在 CPU FP32 上从优化步 1 恢复到步 2。没有用模拟数据替代这些验收。[机器可读记录](../report/evidence/independent-acceptance.json)。

课程的源码、配置、可编辑报告和图表在 Git；本机下载入口是 `dist/` 下的 ZIP。它包含必要权重、配置、字表和校验清单，不依赖该工作目录的训练数据做推理。完整唐诗和宋诗用数据清单中的公开固定 URL 取得。没有把大 checkpoint 写入 Git，也没有将项目上传到外部服务；如课程提交需要同学可访问的外部链接，提交者可把 ZIP 放到自己的私有网盘后填写。

14M 基线包的本机入口：[Orchid-course-baseline.zip](../dist/Orchid-course-baseline.zip)。其中权重、字表、配置与归档内容都已逐项校验；在独立目录解压后可用 `--checkpoint model`，无需训练数据。[归档校验](../report/evidence/delivery-baseline.json)与[解压后生成记录](../report/evidence/independent-archive-generation.json)可复查。

新版课程包使用 92M 韵律续训权重、离线韵表和默认四句五字押韵解码。解压至独立目录后，用 `--checkpoint model` 即可生成。课程包使用空白评分表。[新版课程包](../dist/Orchid-course-prosody.zip) · [新版解压验收](../report/evidence/delivery-92m-prosody.json)

原 92M 唐宋版作为旧评阅和实验对应的留档。在独立环境，归档可在 CPU 加载原始模式，也可在 CUDA 跑原版字数和关键词约束。[原课程包](../dist/Orchid-course.zip) · [原归档校验](../report/evidence/delivery-92m.json) · [原解压验收](../report/evidence/delivery-92m-extraction.json)

真人评阅者只使用各自的[独立评分包](../report/blind_review/README.md)。课程 ZIP 含解盲后的自动评估和可复现源码，用于课程交付，不用作盲评发放材料。
