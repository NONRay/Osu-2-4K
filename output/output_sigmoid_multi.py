#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import tflearn
import tensorflow as tf
import numpy as np
import sys
import os
import random
import csv
from essentia.standard import MonoLoader, Windowing, Spectrum, MelBands
from pydub import AudioSegment
from collections import deque
from zipfile import ZipFile

try:
    tf.compat.v1.disable_eager_execution()
except Exception:
    pass

def main():
    
    '''
    本函数主要包含三个模块：
    1.对需要生成的音乐进行分析，按照模型需要的数据格式量化
    2.根据模型，预测生成的音符
    3.将音符数据转换成OSU对应的文件格式，打包为osz文件
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("song_file")
    parser.add_argument("--model-dir", default="model_sigmoid_multi")
    parser.add_argument("--legacy-arch", dest="legacy_arch", action="store_true")
    parser.add_argument("--modern-arch", dest="legacy_arch", action="store_false")
    parser.set_defaults(legacy_arch=True)
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--lstm-activation", choices=["relu", "tanh"], default="relu")
    args = parser.parse_args()

    dir_name = args.song_file + " chunks"

    songfile = args.song_file
    song = args.song_file[:len(args.song_file) - 4]
    npy_file = song + " Input.npy"
    outfile = song + " Archetype.osu"
    package = song + " Archetype.osz"
    dir_list = os.listdir()

    if dir_name not in dir_list:
        print("Chopping up your song..")
        process_song(sys.argv[1])
    else:
        print("Song already chopped, moving on..")

    print("Analyzing your song..")
    analyze_song(npy_file, dir_name)

    print("Making predictions..")
    make_predictions(
        npy_file,
        outfile,
        model_dir=args.model_dir,
        legacy_arch=args.legacy_arch,
        threshold=args.threshold,
        lstm_activation=args.lstm_activation,
    )

    print("Packing up your beatmap..")
    #create_osz(songfile, outfile, package)
    print("All done!", package, "created in current directory.")
        
    return

def process_song(song_name):
    no_prefix = song_name[:len(song_name) - 4]
    print(no_prefix)
    song = AudioSegment.from_mp3(song_name)
    seg_length = 23
    current_segment = song[:seg_length]
    directory_name = song_name + " chunks"
    print(len(current_segment))

    try:
        os.mkdir(directory_name)
    except:
        print("directory exists, skipping this one")
        return
    else:
        print("directory", directory_name, "created")

    i = 1
    current_segment.export(directory_name + "/" + no_prefix + ' ' + str(i) + ".wav", format = "wav", bitrate = "192k")
    i += 1
    
    while len(current_segment) == 23:
        current_segment = song[seg_length*i : seg_length*i + seg_length]
        current_segment.export(directory_name + "/" + no_prefix + ' ' + str(i) + ".wav", format = "wav", bitrate = "192k")
        i += 1

    # cap it off
    current_segment.export(directory_name + "/" + no_prefix + ' ' + str(i) + ".wav", format = "wav", bitrate = "192k")
    print("success! number of segments:", str(i))

    return


def create_analyzers(fs=44100.0,
                     nhop=1024,
                     nffts=[1024, 2048, 4096],
                     mel_nband=90,
                     mel_freqlo=27.5,
                     mel_freqhi=16000.0):
    '''
    create analyzer from DDC, adapted to ManiaModule
    https://arxiv.org/abs/1703.06891
    '''
    analyzers = []
    for nfft in nffts:
        window = Windowing(size=nfft, type='blackmanharris62')
        spectrum = Spectrum(size=nfft)
        mel = MelBands(inputSize=(nfft // 2) + 1,
                       numberBands=mel_nband,
                       lowFrequencyBound=mel_freqlo,
                       highFrequencyBound=mel_freqhi,
                       sampleRate=fs)
        analyzers.append((window, spectrum, mel))
    return analyzers[0][0], analyzers[0][1], analyzers[0][2]

def analyze_song(file_name = None, dir_name = None):
        '''
        write something here
        '''
        file_list = os.listdir()
        for f in file_list:
            if f == file_name:
                print("Song already has been processed, exiting processing..")
                #os.chdir(cwd)
                return

        cwd = os.getcwd()
        if dir_name != None:
            new_dir = cwd + "/" + dir_name
            os.chdir(new_dir)

        file_list = os.listdir()
        window, spectrum, mel = create_analyzers()
        feats_list = []
        i = 0
        
        for fn in file_list:
            if fn[len(fn) - 1] != 'v':
                continue
            try:
                loader = MonoLoader(filename=fn, sampleRate=44100.0)
                samples = loader()
                feats = window(samples)
                if len(feats) % 2 != 0:
                    feats = np.delete(feats, random.randint(0, len(feats) - 1))
                feats = spectrum(feats)
                feats = mel(feats)
                feats_list.append(feats)
                i+=1
            except Exception as e:
                feats_list.append(np.zeros(90, dtype=np.float32))
                i += 1

        # Apply numerically-stable log-scaling
        feats_list = np.array(feats_list)
        feats_list = np.log(feats_list + 1e-16)
        print(len(feats_list), "length of feats list")
        print(type(feats_list[0][0]))
        if dir_name != None:
            os.chdir(cwd)
        np.save(file_name, feats_list)
        return
        
def make_predictions(npy_file=None, outfile=None, model_dir="model_sigmoid_multi", legacy_arch=True, threshold=None, lstm_activation="relu"):
    '''
    Makes note predictions using the given song data (npy_file), then calls create_chart to... create the chart!
    '''

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from model_sigmoid_pro import build_network

    network = build_network(legacy_arch=legacy_arch, learning_rate=1e-4, lstm_activation=lstm_activation)
    model = tflearn.DNN(network, checkpoint_path=os.path.join(model_dir, "model_rt.tfl"))
    model.load(os.path.join(model_dir, "model.tfl"))

    # load the song data from memory map and reshape appropriately        
    song_mm = np.load(npy_file, mmap_mode="r")
    song_data = np.asarray(song_mm, dtype=np.float32)
    
    # create the given song chunk
    # predict for the current song chunk
    # feed this prediction information back into the model

    note_queue = deque([])
    for i in range(16):
        note_queue.append(np.zeros(8))
    
    predictions = []

    while len(note_queue) != 16 and len(note_queue) < 16:
        note_queue.append(np.zeros(8))
    for j in range(len(song_data)):
        input_chunk = []
        song = []
        note_data = []

        for i in range(16):
            if j - i < 0:
                song.append(np.zeros(90))
            else:
                song.append(song_data[j-i])
            if i < 12:
                note_data.append(note_queue[i])
            elif i != 15:
                note_data.append(np.ones(8))

        song = np.array(song).flatten()
        note_data = np.array(note_data).flatten()
        input_chunk = np.concatenate([song, note_data])
        input_chunk = np.expand_dims(input_chunk, axis=0)
        p = model.predict(input_chunk)
        if threshold is not None:
            p = (np.asarray(p) >= threshold).astype(np.float32)
        note_queue.popleft()
        predictions.append(p[0])
        note_queue.append(p[0][0])

    note_selections = []
    '''
    这里原方案是通过类型被选择的次数给出最终确定是该类型的概率
    由于新方案可以选择多轨道，不同轨道需要分开计算
    '''
    f = open("predictions.csv", "a+", newline='')
    writer = csv.writer(f)
    for k in range(len(predictions)):
        for n in range(4):
            try:
                selection = np.array(predictions[k+n][3-n])
                writer.writerow(selection)
            except IndexError:
                # we have reached the end
                break
    # create_chart(note_selections, outfile)
    return

def create_chart(note_selections, file_name="outfile.osu"):
    '''
    Create the .osu file based on the note selections
    '''
    # template for beginning of file
    osu_file = """osu file format v14

[General]
AudioFilename: audio.mp3
AudioLeadIn: 0
PreviewTime: 1022
Countdown: 0
SampleSet: Normal
StackLeniency: 0.7
Mode: 3
LetterboxInBreaks: 0
SpecialStyle: 0
WidescreenStoryboard: 0
SamplesMatchPlaybackRate: 1

[Editor]
DistanceSpacing: 0.9
BeatDivisor: 4
GridSize: 8
TimelineZoom: 1


[Metadata]
Title:SongTitle
TitleUnicode:SongTitle
Artist:ArtistName
ArtistUnicode:ArtistName
Creator:ManiaArchetype
Version:ManiaArchetype v1
Source:
Tags:
BeatmapID:-1
BeatmapSetID:-1

[Difficulty]
HPDrainRate:6
CircleSize:4
OverallDifficulty:8
ApproachRate:5
SliderMultiplier:1.4
SliderTickRate:1

[TimingPoints]
0,368,4,1,0,40,1,0


[HitObjects]
"""
    current_ms = 0
    last_rail_active = np.array([False, False, False, False], dtype = bool)
    is_reading_holds = np.array([False, False, False, False], dtype = bool)
    k = np.array([0, 0, 0, 0]) # 记录一下滑条结束的i值
    outfile = open(file_name, "w+")
    readCount_a = np.array([0, 0, 0, 0])
    readCount_b = np.array([0, 0, 0, 0])
    # 同理这里需要对每个轨道的选择进行识别，note是一个数组，记录了四个轨道的预期选择
    for i in range(len(note_selections)):
        # 这里因为osu文件每个音符都有一列所以读一个轨道就加一个
        for j in range(len(note_selections[i])):
            # 还原轨道
            railX = int((j + 0.5) * 128)
            railY = 192 # y一般默认192
            
            '''
            还原音符，这里实际上有几种情况：
            1.1.滑条开始
            1.2.正在滑条，之前处理过了不用管，跳到空拍
            2.1.有一个滑条之后紧接的单击，处理成滑条
            2.2.有一个单击之后紧接的单击，跳到空拍
            2.3.普通单击
            3.1.空拍
            '''
            if is_reading_holds[j] == False and last_rail_active[j] == False:
                # 不在任意状态，直接读取
                if note_selections[i][j] == 1:
                    if readCount_a[j] == 4:
                        # 开启了一个滑条
                        # 计算滑条结束时间
                        readHolds = True
                        endTime = current_ms
                        endPos = 1 # 滑条结尾距离开头还有几个时间戳                        
                        while readHolds == True:
                            try:
                                if note_selections[i + endPos][j] == 1 and endPos > 4:
                                    # 到结尾了
                                    readHolds = False
                                else:
                                    endPos += 1
                            except IndexError:
                                # 也到结尾了，不过是以预测被读完的形式
                                break
                        endTime += (endPos * 23)
                        osu_file += (str(railX) + "," + str(railY) + "," + str(current_ms) + ",128,0," + str(endTime) + ":0:0:0:0:\n")
                        k[j] = endPos + i
                        readCount_a[j] = 0
                        is_reading_holds[j] = True
                        last_rail_active[j] = True
                    else:
                        readCount_a[j] += 1

                elif note_selections[i][j] == 2:
                    # 读一个单音符
                    if readCount_b[j] == 4:
                        osu_file += (str(railX) + "," + str(railY) + "," + str(current_ms) + ",1,0,0:0:0:0:\n")
                        # print("i:",i,", k[j]:",k[j])
                        readCount_b[j] = 0
                    else:
                        readCount_b[j] += 1
                    is_reading_holds[j] = False
                    last_rail_active[j] = True

                else:
                    # 空拍
                    last_rail_active[j] = False
                    
            elif is_reading_holds[j] == False and last_rail_active[j] == True:
                # 刚读过一个单击，这个不起效
                    is_reading_holds[j] = False
                    last_rail_active[j] = False
            
            else:
                # 本轨道正在连续滑条
                if i > (k[j] + 1):
                    # 抵达记录的滑条结尾了
                    is_reading_holds[j] = False
                    last_rail_active[j] = True
                else:
                    # print("still reading holds...")
                    is_reading_holds[j] = True
                    last_rail_active[j] = True
            j += 1
        i += 1
        current_ms += 23
    outfile.write(osu_file)
    return

def create_osz(songfile, outfile, package):
    '''
    Package the song .mp3 (songfile) and .osu chart data (outfile) into a single .osz which can be dragged into osu for instant use (package)
    '''
    # set up
    temp_name = songfile
    os.rename(songfile, "audio.mp3")
    # zip up
    with ZipFile(package, mode="w") as oszf:
        oszf.write("audio.mp3")
        oszf.write(outfile)
    # clean up
    os.rename("audio.mp3", temp_name)
    os.remove(outfile)
    return


main()
