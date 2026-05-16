#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
# 用下方代码进行CPU训练，注释后会使用GPU
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tflearn
import tensorflow as tf
import numpy as np

def multilabel_categorical_crossentropy(y_pred, y_true):
    with tf.name_scope("MultilabelCategoricalCrossentropy"):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
    

def preprocess():
    '''
    将要处理的文件加载，处理成模型的输入，并且分组为训练数据和测试数据
    '''

    charts = os.listdir(path="input_charts")
    songs = os.listdir(path="input_songs")
    
    trainX = []
    trainY = []
    testX = []
    testY = []
    
    fail_count = 0
    
    test_data = [1, 11, 21, 31, 45]
    
    num = 0
    for chart in charts:
        num += 1
        if num < 41:
            continue
        training = True
        # 判断一下是不是测试曲目，是的话标记
        for t in test_data:
            if num == t:
                print("test data")
                training = False
                
        # 通过歌曲id找到这个谱面对应歌曲
        id_number = chart.split("_")[0]
        song = None
        for s in songs:
            if s.split()[0] == id_number:
                song = s
                break
                
        if song == None:
            print(chart)
            raise FileNotFoundError
        else:
            print(song, chart)
            
        # 读歌曲
        song_mm = np.load("input_songs/" + song, mmap_mode="r")
        song_data = np.frombuffer(buffer=song_mm, dtype=np.float32, count=-1)
        song_data = song_data[0:song_mm.shape[0]*song_mm.shape[1]]
        song_data = np.reshape(song_data, song_mm.shape)
        
        
        if song_mm[0][0] != song_data[0][0]:
            fail_count += 1
            
        # 读谱面
        note_mm = np.load("input_charts/" + chart, mmap_mode="r")
        note_data = np.frombuffer(note_mm, dtype=np.int32, count=-1)
        note_data = np.reshape(note_data, [len(note_mm), 8])
        
        # 将末尾没有填充的note也填上
        diff = len(song_data) - len(note_data)
        padding = []
        for i in range(diff + 16):
            padding.append(np.zeros(8))

        note_data = np.append(note_data, padding, axis=0)
        
        # 处理一下最后16个歌曲片段
        for h in range(16):
            song_input = []
            note_input = []
            output_chunk = []
            for k in range(16):
                if h - k < 0:
                    song_input.append(np.zeros([90]))
                    if k < 12:
                        note_input.append(np.zeros([8]))
                    elif k != 15:
                        note_input.append(np.ones([8]))
                    if k > 11:
                        output_chunk.append(np.zeros([8]))
                else:
                    song_input.append(song_data[h-k])
                    if k < 12:
                        note_input.append(note_data[h-k])
                    elif k != 15:
                        note_input.append(np.ones([8]))
                    if k > 11: 
                        output_chunk.append(note_data[h-k])
            song_input = np.array(song_input).flatten()
            note_input = np.array(note_input).flatten()
            input_chunk = np.concatenate([song_input, note_input])
            output_chunk = np.concatenate(output_chunk)
            # 预测接下来4个音符片段
            output_chunk = np.reshape(output_chunk, [4, 8])        

            if training:
                trainX.append(input_chunk)
                trainY.append(output_chunk)
            else:
                testX.append(input_chunk)
                testY.append(output_chunk)

        # 正式开始处理
        for j in range(16, len(song_data)):
            song_input = []
            note_input = []
            output_chunk = []
            for k in range(16):
                song_input.append(song_data[j-k])
                if k < 12:
                    note_input.append(note_data[j-k])
                elif k != 15:
                    note_input.append(np.ones([8]))
                if k > 11:
                    output_chunk.append(note_data[j-k])
    
            song_input = np.array(song_input).flatten()
            note_input = np.array(note_input).flatten()
            input_chunk = np.concatenate([song_input, note_input])
            output_chunk = np.concatenate(output_chunk)
            output_chunk = np.reshape(output_chunk, [4, 8])
            if training:
                trainX.append(input_chunk)
                trainY.append(output_chunk)
            else:
                testX.append(input_chunk)
                testY.append(output_chunk)
        
        #if num > 30:
         #   break

        
    # 确认数据情况
    print(len(trainX), "train X", len(trainY), "train Y")
    print(len(trainX[0]), "trainX 0", len(testX[0]), "testX 0")
    print(len(testX), "test X", len(testY), "test Y")
    print(len(trainY[0]), "trainY 0", len(testY[0]), "testY 0")
        
    return trainX,trainY,testX,testY



def main():
    '''
    本模型结构：
    1.预处理 concat flatten
    2.CNN 特征提取
    3.LSTM 序列处理
    4.fully connect输出结果
    '''
    
    # 将数据处理一下，把网做好
    trainX, trainY, testX, testY = preprocess()

    # 将输入数据从长条变为矩阵
    net = tflearn.input_data([None, 1560])
    # net = tflearn.reshape(net, [-1, 1, 1560])
    song = tf.slice(net, [0, 0], [-1, 1440])
    song = tf.reshape(song, [-1, 90, 16])
    prev_notes = tf.slice(net, [0, 1440], [-1, 120])
    prev_notes = tf.reshape(prev_notes, [-1, 8, 15])
    
    # 转置一下保证输入格式正确（交换2轴和3轴）
    song_trans = tf.transpose(song, perm=[0, 2, 1])
    prev_notes_trans = tf.transpose(prev_notes, perm=[0, 2, 1])
    print(song_trans.shape, "song_trans shape before input")
    print(prev_notes_trans.shape, "prev_notes_trans shape before input")
    
    # 去掉80%的noNote数据
    song_encoder = tflearn.conv_1d(song_trans, nb_filter=16, filter_size=3, activation="relu")
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=3)
    song_encoder = tflearn.dropout(song_encoder, keep_prob=0.8)
    print(song_encoder.shape, "song_encoder shape after conv 1")

    song_encoder = tflearn.conv_1d(song_trans, nb_filter=32, filter_size=3, activation="relu")
    song_encoder = tflearn.max_pool_1d(song_encoder, kernel_size=1)
    print(song_encoder.shape, "song_encoder shape after conv 2")

    song_encoder = tflearn.fully_connected(song_encoder, n_units=128, activation="relu")
    print(song_encoder.shape, "song_encoder shape after fc")
    song_encoder = tf.reshape(song_encoder, [-1, 16, 8])
    
    # 将前面的和当前音乐数据区分开
    past_chunks = tf.slice(song_encoder, [0, 0, 0], [-1, 15, 8])
    curr_chunk = tf.slice(song_encoder, [0, 15, 0], [-1, 1, 8])
    
    # 将前面的15段音符数据和音乐数据对应
    # lstm_input = tf.unstack(past_chunks, axis=0)
    lstm_input = tf.math.multiply(past_chunks, prev_notes_trans)
    # lstm_input = tf.reshape(lstm_input, [-1]) # flatten this to add on the current chunk
    
    curr_chunk = tf.math.multiply(curr_chunk, tf.ones([1, 8]))
    # curr_chunk = tf.reshape(curr_chunk, [-1])
    lstm_input = tf.concat([lstm_input, curr_chunk], 1)
    
    # 定义lstm网络形状
    # lstm_input = tf.reshape(lstm_input, [-1, 16, 8]) # reshape to desired shape
    print(lstm_input.shape, "shape of lstm input")

    lstm_input = tflearn.lstm(lstm_input, 64, dropout=0.6, activation="relu")
    lstm_input = tf.reshape(lstm_input, [-1, 8, 8])
    print(lstm_input.shape, "lstm_input shape after lstm 1 + reshape to 8 by 8 (64)")
    
    lstm_input = tflearn.lstm(song_encoder, 64, dropout=0.6, activation="relu")
    print(lstm_input.shape, "lstm_input shape after lstm 2")

    lstm_input = tflearn.fully_connected(lstm_input, n_units=32, activation="sigmoid")
    lstm_input = tflearn.reshape(lstm_input, [-1, 4, 8])
    print(lstm_input.shape, "lstm_input shape after final fc sigmoid layer")
    
    network = tflearn.regression(lstm_input, optimizer = "adam", loss="binary_crossentropy", learning_rate=0.0000001, batch_size=128)
    model = tflearn.DNN(network, checkpoint_path="model_sigmoid_multi/model_rt.tfl", tensorboard_verbose=1)
    #model.load("model_sigmoid_pro/model.tfl")
    model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=16, n_epoch=5) # currently set for retraining
    model.save("model_sigmoid_multi/model.tfl")

main()


    