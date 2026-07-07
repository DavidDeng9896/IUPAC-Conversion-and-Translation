# English–Chinese IUPAC Name Translation

英文化合物 IUPAC 名称与中文名称之间的神经机器翻译模型。

## 论文信息

- **标题**: Neural machine translation of chemical nomenclature between English and Chinese
- **作者**: Tingjun Xu, Weiming Chen, Junhong Zhou, Jingfang Dai, Yingyong Li, Yingli Zhao
- **期刊**: *Journal of Cheminformatics*, 12, 50 (2020)
- **DOI**: [10.1186/s13321-020-00457-0](https://doi.org/10.1186/s13321-020-00457-0)
- **PMC**: [PMC7460765](https://pmc.ncbi.nlm.nih.gov/articles/PMC7460765/)

## 目录结构

```
en-zh-iupac-translation/
├── src/
│   ├── cnn_model.py      # 基于 CNN 的字符级 seq2seq 模型（原 Additional file 2）
│   └── lstm_model.py     # 基于 LSTM 的字符级 seq2seq 模型（原 Additional file 3）
└── data/
    ├── training_dataset.csv     # 完整训练集（CSV，约 100 万行，不纳入 Git）
    ├── training_dataset.xlsx    # 训练/验证数据集（原 Additional file 1）
    ├── validation_results.xlsx
    ├── en2ch_evaluation.xlsx    # 英→中评估结果
    └── ch2en_evaluation.xlsx    # 中→英评估结果
```

## 模型说明

- **CNN 模型** (`src/cnn_model.py`): 三层一维卷积编码器-解码器 + 注意力机制
- **LSTM 模型** (`src/lstm_model.py`): LSTM 编码器-解码器 + teacher forcing

数据集规模：
- En2Ch（英→中）: 30,394 条
- Ch2En（中→英）: 37,207 条

## 依赖

- Python 3.7+
- Keras 2.3 / TensorFlow
- pymssql（原始脚本从 SQL Server 读取训练数据）
- numpy, matplotlib

## 使用注意

原始补充材料中的训练脚本通过 `pymssql` 连接 SQL Server 数据库读取 `TrainDataSet`。本地复现时，建议：

1. 使用 `data/training_dataset.csv`（完整训练集）或 `data/training_dataset.xlsx`（论文补充材料）
2. 修改 `src/cnn_model.py` 和 `src/lstm_model.py` 中的数据加载部分，改为从本地文件读取

在线规则翻译工具（论文对比基线）: [SIOC 化学命名翻译](https://www.organchem.csdb.cn/translate)

## 引用

```bibtex
@article{xu2020neural,
  title={Neural machine translation of chemical nomenclature between English and Chinese},
  author={Xu, Tingjun and Chen, Weiming and Zhou, Junhong and Dai, Jingfang and Li, Yingyong and Zhao, Yingli},
  journal={Journal of Cheminformatics},
  volume={12},
  pages={50},
  year={2020},
  doi={10.1186/s13321-020-00457-0}
}
```
