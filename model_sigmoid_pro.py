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

try:
    tf.compat.v1.disable_eager_execution()
except Exception:
    pass


def set_seed(seed: int) -> None:
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
    with tf.name_scope("MultilabelCategoricalCrossentropy"):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    

def _make_multilabel_binary_accuracy(threshold: float = 0.5):
    def metric(y_pred, y_true):
        y_pred = tf.cast(y_pred >= threshold, tf.float32)
        y_true = tf.cast(y_true >= 0.5, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))
    return metric


def _compute_pos_weights(y: np.ndarray, eps: float = 1e-6, max_weight: float = 20.0) -> np.ndarray:
    y_flat = y.reshape((-1, 32)).astype(np.float32, copy=False)
    pos = np.sum(y_flat, axis=0)
    total = float(y_flat.shape[0])
    neg = total - pos
    w = neg / (pos + eps)
    w = np.clip(w, 1.0, max_weight)
    return w.astype(np.float32)


def _make_weighted_bce_loss(pos_weights: Optional[np.ndarray], label_smoothing: float = 0.0):
    w = None
    if pos_weights is not None:
        w = tf.constant(pos_weights.reshape((1, 32)), dtype=tf.float32)

    def loss(y_pred, y_true):
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing and label_smoothing > 0.0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        y_pred = tf.reshape(y_pred, [-1, 32])
        y_true = tf.reshape(y_true, [-1, 32])
        if w is None:
            per_label = -(y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        else:
            per_label = -(w * y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
        return tf.reduce_mean(tf.reduce_mean(per_label, axis=1))

    return loss


def _make_weighted_focal_loss(pos_weights: Optional[np.ndarray], gamma: float = 2.0, alpha: float = 0.25, label_smoothing: float = 0.0):
    w = None
    if pos_weights is not None:
        w = tf.constant(pos_weights.reshape((1, 32)), dtype=tf.float32)

    def loss(y_pred, y_true):
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        y_true = tf.cast(y_true, tf.float32)
        if label_smoothing and label_smoothing > 0.0:
            y_true = y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing
        y_pred = tf.reshape(y_pred, [-1, 32])
        y_true = tf.reshape(y_true, [-1, 32])

        p_t = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
        a_t = y_true * alpha + (1.0 - y_true) * (1.0 - alpha)
        focal = a_t * tf.pow(1.0 - p_t, gamma) * (-tf.math.log(p_t))
        if w is not None:
            focal = focal * (y_true * w + (1.0 - y_true))
        return tf.reduce_mean(tf.reduce_mean(focal, axis=1))

    return loss


def _load_song(song_path: str) -> np.ndarray:
    song_mm = np.load(song_path, mmap_mode="r")
    song_data = np.asarray(song_mm, dtype=np.float32)
    if song_data.ndim != 2 or song_data.shape[1] != 90:
        raise ValueError(f"unexpected song shape: {song_data.shape} from {song_path}")
    return song_data


def _load_chart(chart_path: str) -> np.ndarray:
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
    steps = int(song_data.shape[0])
    if note_data.shape[0] < steps:
        pad_len = (steps - note_data.shape[0]) + 16
        note_data = np.concatenate([note_data, np.zeros((pad_len, 8), dtype=np.float32)], axis=0)
    else:
        note_data = np.concatenate([note_data, np.zeros((16, 8), dtype=np.float32)], axis=0)

    head_steps = min(16, steps)
    x_head = np.zeros((head_steps, 1560), dtype=np.float32)
    y_head = np.zeros((head_steps, 4, 8), dtype=np.float32)

    one_note = np.ones((8,), dtype=np.float32)
    for h in range(head_steps):
        song_window = np.zeros((16, 90), dtype=np.float32)
        note_window = np.zeros((15, 8), dtype=np.float32)
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
                out_window[k - 12] = note_data[idx]

        x_head[h] = np.concatenate([song_window.reshape(-1), note_window.reshape(-1)], axis=0)
        y_head[h] = out_window

    if steps <= 16:
        return x_head, y_head

    song_windows = _sliding_window_view_axis0(song_data, window_shape=16)
    song_windows = song_windows[1:, ::-1, :]

    note12_windows = _sliding_window_view_axis0(note_data, window_shape=12)
    note12_windows = note12_windows[5: steps - 11, ::-1, :]
    ones_tail = np.ones((steps - 16, 3, 8), dtype=np.float32)
    note_windows = np.concatenate([note12_windows, ones_tail], axis=1)

    out_windows = _sliding_window_view_axis0(note_data, window_shape=4)
    out_windows = out_windows[1: steps - 15, ::-1, :]

    x_main = np.concatenate(
        [song_windows.reshape(steps - 16, -1), note_windows.reshape(steps - 16, -1)], axis=1
    ).astype(np.float32, copy=False)
    y_main = out_windows.astype(np.float32, copy=False)

    x = np.concatenate([x_head, x_main], axis=0)
    y = np.concatenate([y_head, y_main], axis=0)
    return x, y


def preprocess(charts_dir: str = "input_charts", songs_dir: str = "input_songs", start_at: int = 41, test_data: Tuple[int, ...] = (1, 11, 21, 31, 45)):
    '''
    将要处理的文件加载，处理成模型的输入，并且分组为训练数据和测试数据
    '''

    charts = sorted([f for f in os.listdir(path=charts_dir) if f.endswith(".npy")])
    songs = sorted([f for f in os.listdir(path=songs_dir) if f.endswith(".npy")])

    song_by_id = {}
    for s in songs:
        song_id = s.split()[0]
        if song_id not in song_by_id:
            song_by_id[song_id] = s

    train_x_chunks: List[np.ndarray] = []
    train_y_chunks: List[np.ndarray] = []
    test_x_chunks: List[np.ndarray] = []
    test_y_chunks: List[np.ndarray] = []

    test_set = set(test_data)
    for num, chart in enumerate(charts, start=1):
        if num < start_at:
            continue
        training = num not in test_set

        id_number = chart.split("_")[0]
        song = song_by_id.get(id_number)
        if song is None:
            raise FileNotFoundError(f"cannot find song for chart {chart} (id {id_number})")

        print(song, chart)

        song_data = _load_song(os.path.join(songs_dir, song))
        note_data = _load_chart(os.path.join(charts_dir, chart))

        x, y = _build_samples(song_data, note_data)
        if training:
            train_x_chunks.append(x)
            train_y_chunks.append(y)
        else:
            print("test data")
            test_x_chunks.append(x)
            test_y_chunks.append(y)

    trainX = np.concatenate(train_x_chunks, axis=0) if train_x_chunks else np.empty((0, 1560), dtype=np.float32)
    trainY = np.concatenate(train_y_chunks, axis=0) if train_y_chunks else np.empty((0, 4, 8), dtype=np.float32)
    testX = np.concatenate(test_x_chunks, axis=0) if test_x_chunks else np.empty((0, 1560), dtype=np.float32)
    testY = np.concatenate(test_y_chunks, axis=0) if test_y_chunks else np.empty((0, 4, 8), dtype=np.float32)

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
    net = tflearn.input_data([None, 1560])
    song = tf.slice(net, [0, 0], [-1, 1440])
    song = tf.reshape(song, [-1, 90, 16])
    prev_notes = tf.slice(net, [0, 1440], [-1, 120])
    prev_notes = tf.reshape(prev_notes, [-1, 8, 15])

    song_trans = tf.transpose(song, perm=[0, 2, 1])
    prev_notes_trans = tf.transpose(prev_notes, perm=[0, 2, 1])
    print(song_trans.shape, "song_trans shape before input")
    print(prev_notes_trans.shape, "prev_notes_trans shape before input")

    song_encoder = tflearn.conv_1d(song_trans, nb_filter=16, filter_size=3, activation="relu")
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=3)
    song_encoder = tflearn.dropout(song_encoder, keep_prob=0.8)
    print(song_encoder.shape, "song_encoder shape after conv 1")

    song_encoder = tflearn.conv_1d(song_trans if legacy_arch else song_encoder, nb_filter=32, filter_size=3, activation="relu")
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=1)
    print(song_encoder.shape, "song_encoder shape after conv 2")

    song_encoder = tflearn.fully_connected(song_encoder, n_units=128, activation="relu")
    print(song_encoder.shape, "song_encoder shape after fc")
    song_encoder = tf.reshape(song_encoder, [-1, 16, 8])

    past_chunks = tf.slice(song_encoder, [0, 0, 0], [-1, 15, 8])
    curr_chunk = tf.slice(song_encoder, [0, 15, 0], [-1, 1, 8])

    lstm_input = tf.math.multiply(past_chunks, prev_notes_trans)
    curr_chunk = tf.math.multiply(curr_chunk, tf.ones([1, 8]))
    lstm_input = tf.concat([lstm_input, curr_chunk], 1)

    print(lstm_input.shape, "shape of lstm input")

    lstm_input = tflearn.lstm(lstm_input, 64, dropout=0.6, activation=lstm_activation)
    lstm_input = tf.reshape(lstm_input, [-1, 8, 8])
    print(lstm_input.shape, "lstm_input shape after lstm 1 + reshape to 8 by 8 (64)")

    lstm_input = tflearn.lstm(song_encoder if legacy_arch else lstm_input, 64, dropout=0.6, activation=lstm_activation)
    print(lstm_input.shape, "lstm_input shape after lstm 2")

    lstm_input = tflearn.fully_connected(lstm_input, n_units=32, activation="sigmoid")
    lstm_input = tflearn.reshape(lstm_input, [-1, 4, 8])
    print(lstm_input.shape, "lstm_input shape after final fc sigmoid layer")

    return tflearn.regression(lstm_input, optimizer="adam", loss=loss, learning_rate=learning_rate, metric=metric)



def main():
    '''
    本模型结构：
    1.预处理 concat flatten
    2.CNN 特征提取
    3.LSTM 序列处理
    4.fully connect输出结果
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--charts-dir", default="input_charts")
    parser.add_argument("--songs-dir", default="input_songs")
    parser.add_argument("--model-dir", default="model_sigmoid_multi")
    parser.add_argument("--start-at", type=int, default=41)
    parser.add_argument("--test-indices", type=int, nargs="*")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--loss", choices=["bce", "weighted_bce", "focal", "weighted_focal"], default="bce")
    parser.add_argument("--pos-weight-mode", choices=["none", "auto"], default="none")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--lstm-activation", choices=["relu", "tanh"], default="relu")
    parser.add_argument("--legacy-arch", dest="legacy_arch", action="store_true")
    parser.add_argument("--modern-arch", dest="legacy_arch", action="store_false")
    parser.set_defaults(legacy_arch=True)
    args = parser.parse_args()

    set_seed(args.seed)

    preprocess_kwargs = {
        "charts_dir": args.charts_dir,
        "songs_dir": args.songs_dir,
        "start_at": args.start_at,
    }
    if args.test_indices is not None:
        preprocess_kwargs["test_data"] = tuple(args.test_indices)

    trainX, trainY, testX, testY = preprocess(
        **preprocess_kwargs,
    )
    if trainX.size == 0:
        raise RuntimeError("no training data produced")

    os.makedirs(args.model_dir, exist_ok=True)

    pos_weights = None
    if args.pos_weight_mode == "auto" and args.loss in {"weighted_bce", "weighted_focal"}:
        pos_weights = _compute_pos_weights(trainY)

    if args.loss == "bce":
        loss_fn = _make_weighted_bce_loss(None, label_smoothing=args.label_smoothing)
    elif args.loss == "weighted_bce":
        loss_fn = _make_weighted_bce_loss(pos_weights, label_smoothing=args.label_smoothing)
    elif args.loss == "focal":
        loss_fn = _make_weighted_focal_loss(None, gamma=args.focal_gamma, alpha=args.focal_alpha, label_smoothing=args.label_smoothing)
    else:
        loss_fn = _make_weighted_focal_loss(pos_weights, gamma=args.focal_gamma, alpha=args.focal_alpha, label_smoothing=args.label_smoothing)

    network = build_network(
        legacy_arch=args.legacy_arch,
        learning_rate=args.learning_rate,
        loss=loss_fn,
        metric=_make_multilabel_binary_accuracy(0.5),
        lstm_activation=args.lstm_activation,
    )
    model = tflearn.DNN(
        network,
        checkpoint_path=os.path.join(args.model_dir, "model_rt.tfl"),
        tensorboard_verbose=1,
    )

    model.fit(
        trainX,
        trainY,
        validation_set=(testX, testY) if testX.size else 0.0,
        show_metric=True,
        batch_size=args.batch_size,
        n_epoch=args.epochs,
    )
    model.save(os.path.join(args.model_dir, "model.tfl"))


if __name__ == "__main__":
    main()


    
