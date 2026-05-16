# Osu-2-4K

一个用于 **osu!mania 4K** 的谱面生成/实验项目：把歌曲切片并提取特征，使用模型预测音符序列，输出 `predictions.csv`（以及可选生成 `.osu/.osz`）。

## 目录结构

- `input_songs/`：歌曲特征输入（`*.npy`，通常是每个时间片的 mel/log-mel 特征）
- `input_charts/`：对应谱面标签（`*.npy`）
- `model_sigmoid_pro.py`：训练脚本（tflearn/tensorflow）
- `output/output_sigmoid_multi.py`：推理/生成脚本（对 mp3 分段、提特征、跑模型、写 `predictions.csv`，并包含生成 `.osu/.osz` 的函数）
- `output/predictions.csv`：示例/输出结果
- `*.ipynb`：数据处理与实验用笔记本

## 运行方式（概览）

### 训练

训练脚本会从 `input_songs/` + `input_charts/` 读取数据并训练模型：

```bash
python model_sigmoid_pro.py
```

训练完成后会尝试保存到 `model_sigmoid_multi/model.tfl`（如果目录不存在需要先创建，或自行调整保存路径）。

### 生成/推理

对一首 `mp3` 做分段与特征分析，然后加载模型进行预测：

```bash
python output/output_sigmoid_multi.py your_song.mp3
```

默认会在运行目录写入/追加 `predictions.csv`。

## 依赖（不完整）

该项目使用到的主要依赖包括：

- `tensorflow` / `tflearn`
- `numpy`
- `essentia`（音频特征）
- `pydub`（切歌）

不同平台下音频依赖安装方式差异较大，建议按你当前环境的报错逐项安装补齐。

