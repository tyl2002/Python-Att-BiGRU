import tensorflow

from numpy.random import seed

seed(5)
import time
import numpy as np

np.random.seed(5)
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import r2_score
from sklearn import preprocessing
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from math import sqrt
import pandas as pd
from tensorflow.python.keras.layers import merge
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.models import *
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Input, Dropout, Flatten, MaxPooling1D, TimeDistributed, GRU, SimpleRNN, \
    Bidirectional, RNN
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
import numpy as np
from matplotlib import pyplot as plt
# import opfunu
import os

accuracy = tf.keras.metrics.Accuracy()

china_v = pd.read_csv('china_data.csv')
china_data = china_v['value']
# china_v1=china_data[:1610]
china_v1 = china_data[1615:3220]
# china_v1=china_data[3227:4836]
# china_v1=china_data[4839:6447]
data = np.array(china_v1)
data = data.reshape(-1, 1)


def pre_fanchafen(m, n):
    arr = []
    for i in range(0, len(m)):
        a = n[i] + m[i]
        arr.append(a)
    return np.array(arr).astype("int32")


# chafen_low1= chafen_low1.reshape(-1,1)
# a = chafen_low1
look_back = 13
input_dim = 1

# 归一化
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data = min_max_scaler.fit_transform(data)


# 数据集划分
def create_dataset(data, look_back):
    dataset_x, dataset_y = [], []
    for i in range(len(data) - look_back):
        _x = data[i:(i + look_back)]
        dataset_x.append(_x)
        dataset_y.append(data[i + look_back])
    return (np.array(dataset_x), np.array(dataset_y))


# 划分训练集和测试集,90%作为训练集,10%作为测试集

dataset_x, dataset_y = create_dataset(data, look_back)
train_size = int(len(dataset_x) * 0.9)
X_train = dataset_x[:train_size]
y_train = dataset_y[:train_size]
X_test = dataset_x[train_size:]
y_test = dataset_y[train_size:]

X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)
X_test = X_test.astype(np.float32)
y_test = y_test.astype(np.float32)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


def get_data_recurrent(time_steps, input_dim, x, y, attention_column):
    y = y.reshape(y.shape[0], 1)
    x[:, attention_column, :] = np.tile(y[:], (1, 1))
    return x, y


def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(look_back, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = merge.multiply([inputs, a_probs])
    return output_attention_mul


def get_attention_model(time_steps, input_dim, units, dropout, learning_rate, attention_column):
    inputs1 = Input(shape=(time_steps, input_dim))
    lstm_out = LSTM(units, return_sequences=True)(inputs1)
    lstm_out = Dropout(dropout)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)

    attention_mul = Flatten()(attention_mul)

    output = Dense(1)(attention_mul)
    model = Model(inputs=[inputs1], outputs=output)
    #     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    optimizer = tf.keras.optimizers.Adagrad(learning_rate)
    model.compile(optimizer=optimizer, loss="mae")
    return model


def f13(x):
    units = int(x[0])
    dropout = x[1]
    batch_size = int(x[2])
    learning_rate = int(x[3])
    attention_column = int(x[4])
    model = get_attention_model(look_back, input_dim, units, dropout, learning_rate, attention_column)
    #     model = create_model(units, dropout)
    model.fit(X, Y, batch_size=batch_size, validation_split=0.1, epochs=10, shuffle=False, verbose=1)
    #     y_pred = model.predict(X_test)
    #     y_pred_int = np.argmax(y_pred, axis=1)
    model.evaluate(X_test, y_test)


X, Y = get_data_recurrent(look_back, input_dim, X_train, y_train, attention_column=6)

fobj = f13
pop = 10
dim = 5
MaxIter = 10
ub = [128,0.5,128,0.1,11]
lb = [32,0.2,32,0.01,1]

import AOA
# SSAGbestScore,SSAGbestPositon,SSACurve = SSA.SSA(pop=10,dim=4,lb=low_params,ub=up_params,MaxIter=1,fun=fobj) 
GbestScore_4, GbestPositon_4, Curve_4 = AOA.AOA(pop, dim, lb, ub, MaxIter, fobj)
# GbestScore_4,GbestPositon_4,Curve_4 = SSA.SSA(pop,dim,lb,ub,MaxIter,fobj)
# GbestScore_4,GbestPositon_4,Curve_4 = PSO.PSO(pop,dim,lb,ub,MaxIter,fobj) 
# GbestScore_4,GbestPositon_4,Curve_4 = DWO.DWO(pop,dim,lb,ub,MaxIter,fobj) 
# GbestScore_4,GbestPositon_4,Curve_4 = DBO.DBO(pop,dim,lb,ub,MaxIter,fobj) 
# GbestScore_4,GbestPositon_4,Curve_4 = BWO.BWO(pop,dim,lb,ub,MaxIter,fobj) 
# GbestScore_4,GbestPositon_4,Curve_4 = SSA.SSA(pop,dim,lb,ub,MaxIter,fobj) 
# GbestScore_4,GbestPositon_4,Curve_4 = CGAOA.CGAOA(pop,dim,lb,ub,MaxIter,fobj) 
print('最优适应度值：', GbestScore_4)
print('最优解：', GbestPositon_4)
