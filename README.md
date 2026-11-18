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

如果是新建的空环境，激活后先运行 `python -m pip install -r requirements.txt`。`check_env.py` 会写出版本、显卡、BF16 能力和 CUDA 前向/反向检查结果；只使用 CPU 时去掉 `--require-cuda`。VS Code 工作区默认选择 `.venv/bin/python`；若编辑器已记住其他解释器，在「Python: Select Interpreter」中切换。
