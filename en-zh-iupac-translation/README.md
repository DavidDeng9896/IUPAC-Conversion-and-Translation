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
├── scripts/
│   └── prepare_training_data.py  # 整理原始 CSV 为训练/验证集
├── src/
│   ├── data_loader.py            # 从本地 CSV 加载训练数据
│   ├── cnn_model.py              # 基于 CNN 的字符级 seq2seq 模型（原 Additional file 2）
│   └── lstm_model.py             # 基于 LSTM 的字符级 seq2seq 模型（原 Additional file 3）
└── data/
    ├── training_dataset.csv      # 原始完整训练集（不纳入 Git）
    ├── processed/                # 整理后的训练/验证 CSV（不纳入 Git）
    │   ├── ch2en_train.csv
    │   ├── ch2en_val.csv
    │   ├── en2ch_train.csv
    │   ├── en2ch_val.csv
    │   └── dataset_stats.json
    ├── training_dataset.xlsx     # 论文补充材料子集
    ├── validation_results.xlsx
    ├── en2ch_evaluation.xlsx     # 英→中评估结果
    └── ch2en_evaluation.xlsx     # 中→英评估结果
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
- numpy, matplotlib

## 训练数据整理

原始 `data/training_dataset.csv` 需先整理为训练/验证集：

```bash
python scripts/prepare_training_data.py
```

脚本会生成 `data/processed/` 下的 CSV 文件，列名为 `SourceName`、`TargetName`（与原始 SQL 表字段一致）：

| 文件 | 说明 |
|------|------|
| `ch2en_train.csv` / `ch2en_val.csv` | 中→英，训练集 / 验证集（8:2 划分） |
| `en2ch_train.csv` / `en2ch_val.csv` | 英→中（由中→英反向生成） |

整理规则：去除空值与重复项，保留「源含中文、目标为英文」的有效样本。

## 模型训练

在 `src/cnn_model.py` 或 `src/lstm_model.py` 中设置 `direction = 'ch2en'` 或 `'en2ch'`，然后：

```bash
cd en-zh-iupac-translation/src
python cnn_model.py
```

训练脚本从 `data/processed/` 读取对应方向的 CSV，不再依赖 SQL Server。

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
