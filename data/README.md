# 训练数据

本目录用于存放英中 IUPAC 名称翻译训练集，不纳入 Git 版本控制。

## 文件

| 文件 | 说明 |
|------|------|
| `training_dataset.csv` | 训练/验证数据集，列：`Source Name`（中文）、`Target Name`（英文） |

## 获取方式

从内部文档管理系统下载 `training_dataset.csv`，保存到本目录。

下载后可用以下命令校验：

```bash
wc -l data/training_dataset.csv
head -5 data/training_dataset.csv
```

预期约 100 万行（含表头），文件大小约 130 MB。
