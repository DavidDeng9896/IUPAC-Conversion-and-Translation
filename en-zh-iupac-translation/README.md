# English–Chinese IUPAC Name Translation

英文化合物 IUPAC 名称与中文名称之间的神经机器翻译模型。

## 论文信息

- **标题**: Neural machine translation of chemical nomenclature between English and Chinese
- **作者**: Tingjun Xu, Weiming Chen, Junhong Zhou, Jingfang Dai, Yingyong Li, Yingli Zhao
- **期刊**: *Journal of Cheminformatics*, 12, 50 (2020)
- **DOI**: [10.1186/s13321-020-00457-0](https://doi.org/10.1186/s13321-020-00457-0)

## 目录结构

```
en-zh-iupac-translation/
├── requirements.txt
├── data/
│   ├── training_dataset.xlsx    # 训练数据（En2Ch / Ch2En 两个 sheet）
│   └── ...
└── src/
    ├── data_loader.py           # 从 xlsx 加载数据
    ├── train_lstm.py            # 训练 LSTM 模型（推荐，支持推理）
    ├── train_cnn.py             # 训练 CNN 模型
    ├── translate_lstm.py        # 使用训练好的 LSTM 模型翻译
    ├── cnn_model.py             # 论文原始 CNN 脚本（依赖 SQL Server）
    └── lstm_model.py            # 论文原始 LSTM 脚本（依赖 SQL Server）
```

## 环境准备

```bash
cd en-zh-iupac-translation
pip install -r requirements.txt
```

依赖：Python 3.9+、TensorFlow 2.x、pandas、openpyxl。

> 论文补充材料**未提供预训练权重**，需要先训练模型再翻译。

## 快速验证（小样本试跑）

```bash
cd src
python train_lstm.py --direction en2ch --epochs 1 --max-samples 500 --output ../models/lstm_en2ch_demo
python translate_lstm.py --model-dir ../models/lstm_en2ch_demo --text "benzene"
```

## 正式训练

### LSTM（英→中，论文中该方向效果更好）

```bash
cd src
python train_lstm.py \
  --direction en2ch \
  --epochs 100 \
  --batch-size 64 \
  --output ../models/lstm_en2ch
```

### LSTM（中→英）

```bash
python train_lstm.py \
  --direction ch2en \
  --epochs 100 \
  --output ../models/lstm_ch2en
```

### CNN（仅训练，暂不提供 CLI 推理）

```bash
python train_cnn.py --direction en2ch --epochs 100 --output ../models/cnn_en2ch
```

训练完成后，模型目录包含：

- `model.keras` — 模型权重
- `metadata.json` — 字符表、序列长度等推理所需元数据

## 翻译推理

```bash
cd src

# 单条翻译
python translate_lstm.py --model-dir ../models/lstm_en2ch --text "1,3,7-trimethylpurine-2,6-dione"

# 批量翻译
python translate_lstm.py --model-dir ../models/lstm_en2ch --input-file names.txt
```

## 数据说明

`data/training_dataset.xlsx` 含两个 sheet：

| Sheet | 方向 | 样本数 |
|-------|------|--------|
| `Data_EN2CH` | 英文 → 中文 | ~31,000 |
| `Data_CH2EN` | 中文 → 英文 | ~37,800 |

## 原始脚本说明

`cnn_model.py` 和 `lstm_model.py` 是论文作者发布的原始训练脚本，通过 `pymssql` 连接 SQL Server 读取数据，**无法直接在本仓库运行**。请使用上面的 `train_lstm.py` / `train_cnn.py`，它们会从本地 `data/training_dataset.xlsx` 读取数据。

在线规则翻译基线：[SIOC 化学命名翻译](https://www.organchem.csdb.cn/translate)

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
