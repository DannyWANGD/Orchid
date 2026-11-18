# 第三方资料与数据

Orchid 的训练与推理代码独立实现。以下登记区分思路参考与实际导入的数据；项目不加载这些来源提供的预训练权重。

| 来源及版本 | 使用位置和方式 | 许可与说明 |
|---|---|---|
| [Sebastian Raschka：LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch/tree/0311973e2c974b9e0223933103dc400c7a536179) | 第 2～5、7 章：自回归目标、GPT 模块及监督微调的教学参考。Orchid 使用自己的字表和监督掩码 | [Apache-2.0](https://github.com/rasbt/LLMs-from-scratch/blob/0311973e2c974b9e0223933103dc400c7a536179/LICENSE.txt)；未复制源文件 |
| [Andrej Karpathy：nanoGPT](https://github.com/karpathy/nanoGPT/tree/3adf61e154c3fe3fca428ad6bc3818b27a3b8291) | 字符级实验、训练配置、日志与采样的思路参考 | [MIT](https://github.com/karpathy/nanoGPT/blob/3adf61e154c3fe3fca428ad6bc3818b27a3b8291/LICENSE)；未复制源文件 |
| [chinese-poetry：全唐诗 JSON](https://github.com/chinese-poetry/chinese-poetry/tree/b8594f81a89752241442f2ce267d6f66f96704ee/%E5%85%A8%E5%94%90%E8%AF%97) | `poet.tang.*.json` 的古诗正文用于预训练与监督标签派生；本项目清洗、分组，保留来源索引 | [仓库标示 MIT](https://github.com/chinese-poetry/chinese-poetry/blob/b8594f81a89752241442f2ce267d6f66f96704ee/LICENSE)。不纳入现代注释及译文，源文件以固定链接提供 |
| [rbnyng：繁体平水韵 JSON](https://github.com/rbnyng/pingshui_rhyme/blob/0d6b05a932ff1828d8403e7a0de67dd1f6bd81d0/pingshui_rhyme/data/organized_ping_ze_rhyme_dict.json) | 只用于评估偶句韵脚，保留每字全部可能韵部；不复制分类器 | [MIT 项目及来源说明](https://github.com/rbnyng/pingshui_rhyme/tree/0d6b05a932ff1828d8403e7a0de67dd1f6bd81d0)，数据整理自[维基文库《平水韻》](https://zh.wikisource.org/zh-hant/%E5%B9%B3%E6%B0%B4%E9%9F%BB) |

第 15 节的诗歌生成论文用于方法与指标设计，固定来源和数据处理范围见 [数据约定](docs/data_contract.md)。

唐宋扩容版另用同一 chinese-poetry 提交的 `poet.song.*.json`，实际取得日期 2026-11-18。宋诗只用于训练，并应用唐诗留出集的泄漏筛查；具体用途、规则与版本在 [数据约定](docs/data_contract.md)。
