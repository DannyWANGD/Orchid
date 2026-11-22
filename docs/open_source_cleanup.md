# 发布与复现

GitHub 源码包含训练与推理入口、离线演示、固定数据清单、图表和公开项目报告。已导出的模型权重用模型包交付。复现训练时按固定清单获取唐宋语料。[项目首页](../README.md) · [公开报告](../report/Orchid项目报告.pdf) · [数据约定](data_contract.md)

## 本地构建交付包

以下命令在有已导出权重的仓库根目录运行。源码 ZIP 使用当前提交，模型包使用指定的权重与字表，构建时校验模型文件。

```bash
mkdir -p dist
git archive --format=zip --prefix=Orchid/ --output=dist/Orchid-source-demo.zip HEAD
python -E scripts/create_delivery.py \
    --bundle dist/orchid-92m-prosody --output dist/Orchid-model-kit-demo.zip
```

- `Orchid-source-demo.zip` 用于阅读实现和从语料重建。
- `Orchid-model-kit-demo.zip` 带 92M 推理权重。解压后安装相应依赖即可试用。旁边的 `Orchid-model-kit-demo.json` 记录 ZIP 指纹与模型文件清单。
- `report/Orchid项目报告.pdf` 是从公开 LaTeX 源码构建的阅读版。个人署名从本地封面文件读取，生成文件放到本地构建目录。[报告构建说明](../report/latex/README.md)

完整规则复测先按 README 构建唐宋语料和监督集，再重放封存测试。推理计时可以直接使用模型包。第三方数据和韵表的来源及许可见登记。[评估命令](../report/release_evaluation.md#复跑命令) · [第三方登记](../THIRD_PARTY_NOTICES.md)

## 历史材料

课程结项时的 Markdown 报告、原 PDF 和旧构建器保存在固定 Git 提交。训练结果以各运行的 `run.json`、指标文件和样本记录为准，训练配置可在 `run.json` 的 `resolved` 字段查阅。集成评估保留独立的配置快照和逐条风险检查记录。[结项留档](local_delivery.md) · [冻结实验数据](../report/evidence/)
