# Osu-2-4K

An experimental **osu!mania 4K** beatmap generation project: slice songs, extract audio features, use a model to predict note sequences, and output `predictions.csv` (optionally generate `.osu/.osz`).

## Project Layout

- `input_songs/`: input song features (`*.npy`, typically mel/log-mel features per time slice)
- `input_charts/`: chart/label data (`*.npy`)
- `model_sigmoid_pro.py`: training script (tflearn/tensorflow)
- `output/output_sigmoid_multi.py`: inference/generation script (splits mp3, extracts features, runs the model, writes `predictions.csv`, and contains `.osu/.osz` helpers)
- `output/predictions.csv`: sample/output predictions
- `*.ipynb`: notebooks for data processing and experiments

## Usage (Quick Overview)

### Training

The training script loads data from `input_songs/` + `input_charts/` and trains the model:

```bash
python model_sigmoid_pro.py
```

After training, it attempts to save to `model_sigmoid_multi/model.tfl` (create the directory first if it does not exist, or adjust the save path).

### Inference / Generation

Run segmentation + feature extraction on an `mp3`, then load the model to make predictions:

```bash
python output/output_sigmoid_multi.py your_song.mp3
```

By default it writes/appends `predictions.csv` in the current working directory.

## Dependencies (Partial)

Main dependencies used by this project:

- `tensorflow` / `tflearn`
- `numpy`
- `essentia` (audio feature extraction)
- `pydub` (audio slicing)

Audio dependencies differ a lot across platforms; install missing packages based on the errors you see in your environment.

