#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# 用下方代码进行CPU训练，注释后会使用GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import random
from typing import Callable, List, Optional, Tuple, Union
import tflearn
import tensorflow as tf
import numpy as np

# 这个脚本是一个“音乐特征 -> 未来若干帧谱面”的训练脚本，面向 osu!mania 4K。
# 核心思路：
# - 输入：过去 16 个音乐片段的梅尔频带特征（每帧 90 维），以及过去 12 帧的谱面（每帧 8 维）
# - 输出：预测“接下来 4 帧”的谱面（4×8）
# - 这里的 8 维通常表示 4K 下每个轨道的若干状态（例如：按下/长按等），具体语义取决于你数据集的编码方式
#
# 输入张量的具体拼接方式（最终喂给网络的是一条 1560 维向量）：
# - 音乐窗口：16 帧 × 90 维 = 1440
# - 谱面窗口：15 帧 × 8 维  = 120
# - 总计：1560
#
# 注意：tflearn 基于 TF1 图模式（graph mode），在 TF2 默认 eager 下会报错或行为异常，
# 所以这里显式关掉 eager，保证 tflearn 能正常工作。

try:
    tf.compat.v1.disable_eager_execution()
except Exception:
    pass


def set_seed(seed: int) -> None:
    """
    设定随机种子，尽量让训练可复现。

    这里同时设置：
    - Python random
    - NumPy
    - TensorFlow（TF2 / TF1 兼容写法）
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        tf.random.set_seed(seed)
    except Exception:
        try:
            tf.compat.v1.set_random_seed(seed)
        except Exception:
            pass


def multilabel_categorical_crossentropy(y_pred, y_true):
    """
    旧版（保留未使用）：多标签交叉熵。

    说明：
    - 这个实现实际上调用了 softmax_cross_entropy_with_logits，
      更适合“互斥分类”（单标签）而不是“多标签”（独立 Bernoulli）。
    - 当前训练使用的是 sigmoid + BCE / focal，因此这里仅作为历史遗留保留。
    """
    with tf.name_scope("MultilabelCategoricalCrossentropy"):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    

def _make_multilabel_binary_accuracy(threshold: float = 0.5):
    """
    构造一个多标签二值准确率指标。

    用途：
    - 输出层是 sigmoid（每个维度独立概率）
    - 先按 threshold 把 y_pred 二值化，再和 y_true 做逐元素相等率

    注意：
    - 这是“逐元素准确率”，在类别极不平衡时（大量 0）会虚高；
      更严谨的评估可以看 precision/recall/F1，但 tflearn 集成起来更麻烦，
      这里提供一个最直观/最低成本的监控指标。
    """
    def metric(y_pred, y_true):
        y_pred = tf.cast(y_pred >= threshold, tf.float32)
        y_true = tf.cast(y_true >= 0.5, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    return metric


def _compute_pos_weights(y: np.ndarray, eps: float = 1e-6, max_weight: float = 20.0) -> np.ndarray:
    """
    根据训练标签 y 统计每个 label 的正负比例，生成正样本权重 pos_weight。

    背景：
    - 对于谱面预测，“有 note”往往是极少数，“无 note”占绝大多数
    - 直接用普通 BCE 会促使模型学会“全预测 0”也能拿到很低 loss
    - 通过对正样本加权，可以让“错过一个 note”变得更贵

    返回：
    - shape = (32,) 的权重，32 = 4×8（未来 4 帧，每帧 8 维）
    """
    y_flat = y.reshape((-1, 32)).astype(np.float32, copy=False)
    pos = np.sum(y_flat, axis=0)
    total = float(y_flat.shape[0])
    neg = total - pos
    w = neg / (pos + eps)
    w = np.clip(w, 1.0, max_weight)
    return w.astype(np.float32)


def _make_weighted_bce_loss(pos_weights: Optional[np.ndarray], label_smoothing: float = 0.0):
    """
    构造（可选正样本加权的）BCE loss。

    参数：
    - pos_weights：shape=(32,) 或 None
      - None 表示普通 BCE
      - 非 None 表示对正样本项 y*log(p) 乘以权重（让正样本更重要）
    - label_smoothing：标签平滑系数，避免过度自信（对极端不平衡时有时能稳定训练）

    说明：
    - 这里 y_pred 期望是 sigmoid 后的概率（0~1）
    - 因为 tflearn 的 regression 接口会把 loss 当函数调用，所以我们返回一个闭包
    """
    w = None
    if pos_weights is not None:
        w = tf.constant(pos_weights.reshape((1, 32)), dtype=tf.float32)

    def loss(y_pred, y_true):
        # 数值稳定：避免 log(0) 或 log(1) 导致 NaN
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing and label_smoothing > 0.0:
            # 将 0/1 轻微推向 0.5，减少过拟合/过度自信
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        # 把 (batch,4,8) 展平成 (batch,32)，统一按 label 维度计算
        y_pred = tf.reshape(y_pred, [-1, 32])
        y_true = tf.reshape(y_true, [-1, 32])
        if w is None:
            per_label = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        else:
            # 只对正样本项加权：w*y*log(p)
            per_label = -(w * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        # 先对 label 求平均，再对 batch 求平均
        return tf.reduce_mean(tf.reduce_mean(per_label, axis=1))

    return loss


def _make_weighted_focal_loss(pos_weights: Optional[np.ndarray], gamma: float = 2.0, alpha: float = 0.25, label_smoothing: float = 0.0):
    """
    构造（可选正样本加权的）Focal Loss（多标签版本）。

    背景：
    - Focal Loss 会降低“易分类样本”的贡献，把优化重心放在“难样本”上
    - 在极不平衡任务（少数 note）里通常比 BCE 更有效

    参数：
    - gamma：聚焦强度，越大越聚焦难样本（常用 2）
    - alpha：类别平衡系数（常用 0.25），这里以 y_true 为 1 的部分乘 alpha
    - pos_weights：额外的正样本权重（可选），与 focal 项相乘
    """
    w = None
    if pos_weights is not None:
        w = tf.constant(pos_weights.reshape((1, 32)), dtype=tf.float32)

    def loss(y_pred, y_true):
        # 数值稳定
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing and label_smoothing > 0.0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        y_pred = tf.reshape(y_pred, [-1, 32])
        y_true = tf.reshape(y_true, [-1, 32])

        # p_t：预测为真类的概率（对正类是 p，对负类是 1-p）
        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        # a_t：对正类/负类分别赋予 alpha / (1-alpha)
        a_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        # focal：a_t * (1-p_t)^gamma * (-log(p_t))
        focal = a_t * tf.pow(1.0 - p_t, gamma) * (-tf.math.log(p_t))
        if w is not None:
            # 额外对正样本乘 pos_weight；负样本保持 1
            focal = focal * (y_true * w + (1.0 - y_true))
        return tf.reduce_mean(tf.reduce_mean(focal, axis=1))

    return loss


def _load_song(song_path: str) -> np.ndarray:
    """
    读取一首歌的特征数据（npy）。

    期望格式：
    - shape = (T, 90)
      - T：片段数（时间步数）
      - 90：每个时间步的梅尔频带特征维度
    """
    song_mm = np.load(song_path, mmap_mode="r")
    song_data = np.asarray(song_mm, dtype=np.float32)
    if song_data.ndim != 2 or song_data.shape[1] != 90:
        raise ValueError(f"unexpected song shape: {song_data.shape} from {song_path}")
    return song_data


def _load_chart(chart_path: str) -> np.ndarray:
    """
    读取谱面标签数据（npy）。

    期望格式：
    - shape = (T, 8)
      - 8：每个时间步的谱面标签维度（通常对应 4K 的 4 轨与其状态编码）

    兼容：
    - 如果保存时是 1 维数组，则尝试按 8 列 reshape 成 (T, 8)
    """
    note_mm = np.load(chart_path, mmap_mode="r")
    note_data = np.asarray(note_mm)
    if note_data.ndim == 1:
        if note_data.size % 8 != 0:
            raise ValueError(f"chart length not divisible by 8: {note_data.size} from {chart_path}")
        note_data = note_data.reshape(-1, 8)
    if note_data.ndim != 2 or note_data.shape[1] != 8:
        raise ValueError(f"unexpected chart shape: {note_data.shape} from {chart_path}")
    return note_data.astype(np.float32, copy=False)


def _sliding_window_view_axis0(a: np.ndarray, window_shape: int) -> np.ndarray:
    """
    在 axis=0 上做 sliding window view（滑动窗口视图）。

    说明：
    - 优先使用 numpy 自带的 sliding_window_view（版本较新的 numpy 才有）
    - 如果没有，则退回到 as_strided 手工实现（更危险，但这里可控）

    返回：
    - 对于 a.shape = (T, ...)，返回 shape = (T-window+1, window, ...)
    """
    try:
        return np.lib.stride_tricks.sliding_window_view(a, window_shape=window_shape, axis=0)
    except Exception:
        if window_shape <= 0:
            raise ValueError("window_shape must be positive")
        if a.shape[0] < window_shape:
            raise ValueError("window larger than axis length")
        out_shape = (a.shape[0] - window_shape + 1, window_shape) + a.shape[1:]
        out_strides = (a.strides[0],) + a.strides
        return np.lib.stride_tricks.as_strided(a, shape=out_shape, strides=out_strides)


def _build_samples(song_data: np.ndarray, note_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将一首歌的数据构造成训练样本 X/Y。

    输出：
    - X: (N, 1560)
    - Y: (N, 4, 8)

    细节（对齐原始脚本逻辑）：
    - 每个时间步 j 生成 1 条样本：
      - 音乐输入：过去 16 帧（j, j-1, ..., j-15），不足用 0 填充
      - 谱面输入：过去 12 帧真实 note + 3 帧常量 1（对应你原来的 k=12..14 的 ones）
      - 输出：未来/接下来 4 帧（对应你原来 k=12..15 的 output_chunk）

    注意：
    - 这里“过去/未来”的语义以你的原代码为准：它使用倒序拼接（最新帧在前）
    - note_data 会在末尾补 16 帧 0，避免最后几帧取 label 越界
    """
    steps = int(song_data.shape[0])
    # 保证 note_data 至少覆盖到 steps，并额外多 16 帧 0 以支持最后窗口的输出取值
    if note_data.shape[0] < steps:
        pad_len = (steps - note_data.shape[0]) + 16
        note_data = np.concatenate([note_data, np.zeros((pad_len, 8), dtype=np.float32)], axis=0)
    else:
        note_data = np.concatenate([note_data, np.zeros((16, 8), dtype=np.float32)], axis=0)

    # 前 16 帧需要处理“左侧不足”的 padding（j-i < 0），用小循环更直观
    head_steps = min(16, steps)
    x_head = np.zeros((head_steps, 1560), dtype=np.float32)
    y_head = np.zeros((head_steps, 4, 8), dtype=np.float32)

    one_note = np.ones((8,), dtype=np.float32)
    for h in range(head_steps):
        # song_window: (16,90)；note_window: (15,8)；out_window: (4,8)
        # 注意这里窗口内部按“从当前往过去”倒序存放，和原脚本一致
        song_window = np.zeros((16, 90), dtype=np.float32)
        note_window = np.zeros((15, 8), dtype=np.float32)
        # k=12..14 的 note 输入被固定为 1（你原脚本用 ones 占位）
        note_window[12:15] = one_note
        out_window = np.zeros((4, 8), dtype=np.float32)

        for k in range(16):
            idx = h - k
            if idx < 0:
                continue
            song_window[k] = song_data[idx]
            if k < 12:
                note_window[k] = note_data[idx]
            if k > 11:
                # k=12..15 -> out_window[0..3]
                out_window[k - 12] = note_data[idx]

        # 拼成模型的扁平输入
        x_head[h] = np.concatenate([song_window.reshape(-1), note_window.reshape(-1)], axis=0)
        y_head[h] = out_window

    if steps <= 16:
        return x_head, y_head

    # 从第 16 帧开始就没有“左侧不足”问题了，可以用滑窗一次性批量生成
    # song_windows: (steps-16, 16, 90)，每个样本对应 16 帧音乐
    song_windows = _sliding_window_view_axis0(song_data, window_shape=16)
    # 对齐原逻辑：j=16 对应窗口 [1]（song[16..1]），并且需要倒序
    song_windows = song_windows[1:, ::-1, :]

    # note 输入部分只取过去 12 帧真实标签，然后拼 3 帧常量 1
    note12_windows = _sliding_window_view_axis0(note_data, window_shape=12)
    # 对齐原逻辑：j=16 时 note 需要 [16..5]（共 12 帧），同样倒序
    note12_windows = note12_windows[5: steps - 11, ::-1, :]
    ones_tail = np.ones((steps - 16, 3, 8), dtype=np.float32)
    note_windows = np.concatenate([note12_windows, ones_tail], axis=1)

    # 输出部分取 4 帧标签（对应 k=12..15），同样需要与 j 对齐并倒序
    out_windows = _sliding_window_view_axis0(note_data, window_shape=4)
    out_windows = out_windows[1: steps - 15, ::-1, :]

    # 拼 X：音乐(16×90) + note(15×8)
    x_main = np.concatenate(
        [song_windows.reshape(steps - 16, -1), note_windows.reshape(steps - 16, -1)], axis=1
    ).astype(np.float32, copy=False)
    y_main = out_windows.astype(np.float32, copy=False)

    # 合并 head（前 16 帧）和 main（剩余帧）
    x = np.concatenate([x_head, x_main], axis=0)
    y = np.concatenate([y_head, y_main], axis=0)
    return x, y


def preprocess(charts_dir: str = "input_charts", songs_dir: str = "input_songs", start_at: int = 41, test_data: Tuple[int, ...] = (1, 11, 21, 31, 45)):
    '''
    将要处理的文件加载，处理成模型的输入，并且分组为训练数据和测试数据
    '''

    # 重要：用 sorted() 固定顺序，避免 os.listdir() 返回顺序不稳定导致训练/测试划分漂移
    charts = sorted([f for f in os.listdir(path=charts_dir) if f.endswith(".npy")])
    songs = sorted([f for f in os.listdir(path=songs_dir) if f.endswith(".npy")])

    # 建立 song_id -> 文件名 的映射
    # 约定：歌曲特征文件名以“id 开头，空格分隔”存放，例如： "1397714 tokiwa - Orthodox Input.npy"
    song_by_id = {}
    for s in songs:
        song_id = s.split()[0]
        if song_id not in song_by_id:
            song_by_id[song_id] = s

    # 用 list 先收集每首歌生成的样本块，最后再 concatenate
    # 这样比逐条 append Python list 再转 numpy 更快、更省内存碎片
    train_x_chunks: List[np.ndarray] = []
    train_y_chunks: List[np.ndarray] = []
    test_x_chunks: List[np.ndarray] = []
    test_y_chunks: List[np.ndarray] = []

    test_set = set(test_data)
    for num, chart in enumerate(charts, start=1):
        # 兼容你早期脚本的“从第 start_at 首开始训练”的做法（可能用于跳过质量差的早期数据）
        if num < start_at:
            continue
        training = num not in test_set

        # 约定：谱面文件名以“id_...”开头，例如： "1397714_Orthodox_tokiwa_IN.npy"
        id_number = chart.split("_")[0]
        song = song_by_id.get(id_number)
        if song is None:
            raise FileNotFoundError(f"cannot find song for chart {chart} (id {id_number})")

        print(song, chart)

        # 加载特征与标签
        song_data = _load_song(os.path.join(songs_dir, song))
        note_data = _load_chart(os.path.join(charts_dir, chart))

        # 单曲构造样本
        x, y = _build_samples(song_data, note_data)
        if training:
            train_x_chunks.append(x)
            train_y_chunks.append(y)
        else:
            print("test data")
            test_x_chunks.append(x)
            test_y_chunks.append(y)

    # 合并所有歌曲的样本
    trainX = np.concatenate(train_x_chunks, axis=0) if train_x_chunks else np.empty((0, 1560), dtype=np.float32)
    trainY = np.concatenate(train_y_chunks, axis=0) if train_y_chunks else np.empty((0, 4, 8), dtype=np.float32)
    testX = np.concatenate(test_x_chunks, axis=0) if test_x_chunks else np.empty((0, 1560), dtype=np.float32)
    testY = np.concatenate(test_y_chunks, axis=0) if test_y_chunks else np.empty((0, 4, 8), dtype=np.float32)

    # 打印 shape 便于确认数据是否正确
    print(trainX.shape, "trainX")
    print(trainY.shape, "trainY")
    print(testX.shape, "testX")
    print(testY.shape, "testY")

    return trainX, trainY, testX, testY


def build_network(
    legacy_arch: bool = True,
    learning_rate: float = 1e-4,
    loss: Union[str, Callable] = "binary_crossentropy",
    metric: Optional[Callable] = None,
    lstm_activation: str = "relu",
):
    """
    构建 tflearn 网络（训练/推理共用，避免结构漂移）。

    参数：
    - legacy_arch：
      - True：保留你早期脚本中的两个“非典型连接”以兼容旧模型/旧行为：
        1) 第二层 conv 仍然从原始 song_trans 接入（而不是接在第一层 conv 后）
        2) 第二层 LSTM 从 song_encoder 接入（而不是接在第一层 LSTM 后）
      - False：使用更常见的串联方式（conv2 接 conv1，lstm2 接 lstm1）
    - loss / metric：
      - 允许传入字符串（例如 "binary_crossentropy"）或自定义函数（闭包）
      - 训练时可用加权 BCE / focal；推理时一般保持默认即可
    - lstm_activation：
      - 默认 relu（保持旧行为）
      - 更常见的是 tanh（更稳定），可用参数切换
    """
    # 输入是扁平向量 (1560,)
    # 然后通过 slice/reshape 拆成音乐窗口与谱面窗口
    net = tflearn.input_data([None, 1560])
    song = tf.slice(net, [0, 0], [-1, 1440])
    song = tf.reshape(song, [-1, 90, 16])
    prev_notes = tf.slice(net, [0, 1440], [-1, 120])
    prev_notes = tf.reshape(prev_notes, [-1, 8, 15])

    # 转置后变成：
    # - song_trans: (batch, 16, 90)  -> 适配 conv_1d 的输入 [batch, steps, channels]
    # - prev_notes_trans: (batch, 15, 8)
    song_trans = tf.transpose(song, perm=[0, 2, 1])
    prev_notes_trans = tf.transpose(prev_notes, perm=[0, 2, 1])
    print(song_trans.shape, "song_trans shape before input")
    print(prev_notes_trans.shape, "prev_notes_trans shape before input")

    # 1D CNN：在时间维度上做卷积，提取局部节奏/频谱模式
    song_encoder = tflearn.conv_1d(song_trans, nb_filter=16, filter_size=3, activation="relu")
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=3)
    song_encoder = tflearn.dropout(song_encoder, keep_prob=0.8)
    print(song_encoder.shape, "song_encoder shape after conv 1")

    # 第二层 CNN：
    # - legacy_arch=True 时仍从 song_trans 接入（保留旧结构）
    # - legacy_arch=False 时从上一层的 song_encoder 接入（更常见）
    song_encoder = tflearn.conv_1d(song_trans if legacy_arch else song_encoder, nb_filter=32, filter_size=3, activation="relu")
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=1)
    print(song_encoder.shape, "song_encoder shape after conv 2")

    # 这里用 fully_connected 把每个时间步的特征投影到 128 维
    song_encoder = tflearn.fully_connected(song_encoder, n_units=128, activation="relu")
    print(song_encoder.shape, "song_encoder shape after fc")
    # 再 reshape 成 (batch, 16, 8)，方便后续与 note 维度（8）对齐
    song_encoder = tf.reshape(song_encoder, [-1, 16, 8])

    # 切分过去 15 帧（用于与 prev_notes 相乘）和当前帧（用于预测未来）
    past_chunks = tf.slice(song_encoder, [0, 0, 0], [-1, 15, 8])
    curr_chunk = tf.slice(song_encoder, [0, 15, 0], [-1, 1, 8])

    # 将音乐特征与历史谱面做逐元素相乘（相当于用“是否有 note”的信息去 gate 音乐特征）
    lstm_input = tf.math.multiply(past_chunks, prev_notes_trans)
    # 当前帧没有历史谱面信息，按原逻辑用全 1 作为占位（不抑制特征）
    curr_chunk = tf.math.multiply(curr_chunk, tf.ones([1, 8]))
    # 拼回 16 帧：前 15 帧 gated + 1 帧 ungated
    lstm_input = tf.concat([lstm_input, curr_chunk], 1)

    print(lstm_input.shape, "shape of lstm input")

    # LSTM：建模时间序列依赖（节奏型、pattern）
    lstm_input = tflearn.lstm(lstm_input, 64, dropout=0.6, activation=lstm_activation)
    # 原脚本中做了 reshape 到 (batch, 8, 8)，保持一致（是否最合理取决于你的想法/实验）
    lstm_input = tf.reshape(lstm_input, [-1, 8, 8])
    print(lstm_input.shape, "lstm_input shape after lstm 1 + reshape to 8 by 8 (64)")

    # 第二层 LSTM：
    # - legacy_arch=True 时从 song_encoder 进入（保留旧结构）
    # - legacy_arch=False 时从上一层 lstm_input 进入（更常见）
    lstm_input = tflearn.lstm(song_encoder if legacy_arch else lstm_input, 64, dropout=0.6, activation=lstm_activation)
    print(lstm_input.shape, "lstm_input shape after lstm 2")

    # 输出层：32 = 4×8，激活 sigmoid -> 每个维度独立概率
    lstm_input = tflearn.fully_connected(lstm_input, n_units=32, activation="sigmoid")
    lstm_input = tflearn.reshape(lstm_input, [-1, 4, 8])
    print(lstm_input.shape, "lstm_input shape after final fc sigmoid layer")

    # regression：指定优化器、loss、学习率，以及可选 metric
    return tflearn.regression(lstm_input, optimizer="adam", loss=loss, learning_rate=learning_rate, metric=metric)



def main():
    '''
    本模型结构：
    1.预处理 concat flatten
    2.CNN 特征提取
    3.LSTM 序列处理
    4.fully connect输出结果
    '''
    # 命令行参数：让脚本更容易复用/对比实验
    parser = argparse.ArgumentParser()
    parser.add_argument("--charts-dir", default="input_charts")
    parser.add_argument("--songs-dir", default="input_songs")
    parser.add_argument("--model-dir", default="model_sigmoid_multi")
    parser.add_argument("--start-at", type=int, default=41)
    # 训练/测试划分：以“第几首 chart（从 1 开始）”为索引
    parser.add_argument("--test-indices", type=int, nargs="*")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    # loss 相关：用于应对极度不平衡（noNote 太多）
    parser.add_argument("--loss", choices=["bce", "weighted_bce", "focal", "weighted_focal"], default="bce")
    parser.add_argument("--pos-weight-mode", choices=["none", "auto"], default="none")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    # LSTM 激活：默认保持旧行为（relu），更常见的选择是 tanh
    parser.add_argument("--lstm-activation", choices=["relu", "tanh"], default="relu")
    # 网络结构开关：legacy 保留旧连接方式，modern 使用更常见串联
    parser.add_argument("--legacy-arch", dest="legacy_arch", action="store_true")
    parser.add_argument("--modern-arch", dest="legacy_arch", action="store_false")
    parser.set_defaults(legacy_arch=True)
    args = parser.parse_args()

    # 设置随机种子（尽量复现）
    set_seed(args.seed)

    # 预处理参数整理：保持 preprocess 接口清晰
    preprocess_kwargs = {
        "charts_dir": args.charts_dir,
        "songs_dir": args.songs_dir,
        "start_at": args.start_at,
    }
    if args.test_indices is not None:
        preprocess_kwargs["test_data"] = tuple(args.test_indices)

    # 读取数据并构造样本
    trainX, trainY, testX, testY = preprocess(
        **preprocess_kwargs,
    )
    if trainX.size == 0:
        raise RuntimeError("no training data produced")

    # 训练输出目录（保存模型文件）
    os.makedirs(args.model_dir, exist_ok=True)

    # 可选：自动计算每个 label 的正样本权重
    pos_weights = None
    if args.pos_weight_mode == "auto" and args.loss in {"weighted_bce", "weighted_focal"}:
        pos_weights = _compute_pos_weights(trainY)

    # 选择 loss 函数（普通/加权 BCE 或 focal）
    if args.loss == "bce":
        loss_fn = _make_weighted_bce_loss(None, label_smoothing=args.label_smoothing)
    elif args.loss == "weighted_bce":
        loss_fn = _make_weighted_bce_loss(pos_weights, label_smoothing=args.label_smoothing)
    elif args.loss == "focal":
        loss_fn = _make_weighted_focal_loss(None, gamma=args.focal_gamma, alpha=args.focal_alpha, label_smoothing=args.label_smoothing)
    else:
        loss_fn = _make_weighted_focal_loss(pos_weights, gamma=args.focal_gamma, alpha=args.focal_alpha, label_smoothing=args.label_smoothing)

    # 构建网络（训练/推理共用）
    network = build_network(
        legacy_arch=args.legacy_arch,
        learning_rate=args.learning_rate,
        loss=loss_fn,
        metric=_make_multilabel_binary_accuracy(0.5),
        lstm_activation=args.lstm_activation,
    )
    model = tflearn.DNN(
        network,
        # checkpoint_path：训练过程中会自动保存中间 checkpoint
        checkpoint_path=os.path.join(args.model_dir, "model_rt.tfl"),
        tensorboard_verbose=1,
    )

    # 开始训练
    # 注意：tflearn 的 validation_set 支持 (X, Y) 或者一个 float（比例）
    model.fit(
        trainX,
        trainY,
        validation_set=(testX, testY) if testX.size else 0.0,
        show_metric=True,
        batch_size=args.batch_size,
        n_epoch=args.epochs,
    )
    # 保存最终模型
    model.save(os.path.join(args.model_dir, "model.tfl"))


if __name__ == "__main__":
    # 作为脚本运行时才执行训练，避免被 import 时直接开训
    main()


    
