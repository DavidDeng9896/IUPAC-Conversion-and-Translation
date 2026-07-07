# IUPAC Conversion and Translation

本仓库整合了化学命名相关的两类开源工具：

| 目录 | 功能 | 来源论文 |
|------|------|----------|
| [`en-zh-iupac-translation/`](en-zh-iupac-translation/) | 英文 IUPAC 化合物名称 → 中文名称 | Xu et al., *J. Cheminform.* 2020 |
| [`stout-2.0/`](stout-2.0/) | SMILES → IUPAC 名称（双向） | Rajan et al., *J. Cheminform.* 2024 |

## 典型工作流

```
SMILES  ──(STOUT 2.0)──►  英文 IUPAC  ──(EN-ZH NMT)──►  中文化合物名称
```

## 快速开始

### STOUT 2.0（SMILES → IUPAC）

```bash
cd stout-2.0
pip install -e .
```

```python
from STOUT import translate_forward, translate_reverse

smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
print(translate_forward(smiles))  # 1,3,7-trimethylpurine-2,6-dione
```

### 英中 IUPAC 翻译

训练数据与 CNN/LSTM 模型代码见 [`en-zh-iupac-translation/`](en-zh-iupac-translation/)。完整训练集位于 `en-zh-iupac-translation/data/training_dataset.csv`；原始脚本依赖 Keras 2.3 与 SQL Server 读取训练集，使用前需改写数据加载逻辑以从本地 CSV 读取。

## 许可证

各子项目保留其原始许可证，详见各目录内的 `LICENSE` 或论文补充材料说明。
