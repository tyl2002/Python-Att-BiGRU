#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.python.keras.layers import merge
from tensorflow.python.keras.layers.core import *
from tensorflow.python.keras.layers.recurrent import GRU
from tensorflow.python.keras.models import *
from tensorflow.keras.layers import Input,LSTM
import tensorflow as tf
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.layers import Bidirectional
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn import preprocessing
import random
from tensorflow.keras import Input, Model,Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import LeakyReLU


low_tec=np.load(r"C:\Users\86156\Desktop\tensorflow实战\TEC代码与数据集\北纬22.5东经150low_2.npy").astype("int32") 


low_tec_diff1=np.diff(low_tec)

chafen_low_tec=(low_tec_diff1).astype("int32")

#m是y_pre,n是test
def pre_fanchafen(m,n):
    arr = []
    print("m",len(m))
    print("n",len(n))
    for i in range(0,len(m)):
        a = n[i]+m[i]
        # print("test",n[i])
        # print("pre",a)
        # if a>n[i]:
        #     print("1")
        arr.append(a)
    
    return np.array(arr).astype("int32")


chafen_low_tec = chafen_low_tec.reshape(-1,1)
a = chafen_low_tec

look_back = 12
input_dim = 1


min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
data = min_max_scaler.fit_transform(chafen_low_tec)


#相对误差
def custom_rel_err(x, y):
    return tf.reduce_mean((tf.abs(x-y)/tf.maximum(1e-8, tf.abs(x) + tf.abs(y))))


#1 数据集划分
def create_dataset(data, look_back) :

    dataset_x, dataset_y = [], []

    for i in range(len(data) - look_back):

        _x = data[i:(i + look_back)]

        dataset_x.append(_x)

        dataset_y.append(data[i + look_back])

    return (np.array(dataset_x), np.array(dataset_y))


dataset_x, dataset_y = create_dataset(data,look_back)


# 2 划分训练集和测试集,90%作为训练集,10%作为测试集

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


def lstm1():
    model = tf.keras.models.Sequential()
  
    model.add(LSTM(128, input_shape=(12, 1), return_sequences=True))
    model.add(Dropout(0.5))
  
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.5))
    
    
    model.add(Flatten())
  
    model.add(Dense(32,activation='tanh'))    
    model.add(Dropout(0.5))
  
    model.add(Dense(1,activation='sigmoid'))
    return model


model = lstm1()
model.compile(optimizer="adagrad", loss="mae") 
model.summary()


history = model.fit(X_train, y_train, epochs=20, batch_size=128, validation_split=0.1) #原本50次
model.evaluate(X_test,y_test)


#反归一化
a = a.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler()
min_max_scaler.fit_transform(a)

_pre = model.predict(X_test)
_pre = _pre.reshape(-1, 1)
_pre = min_max_scaler.inverse_transform(_pre).astype("int32")


y_test = low_tec[72974:]
yy = low_tec


ypre = pre_fanchafen(_pre,y_test).flatten()


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']

x = np.arange(72974,73074)
#y = np.arange(72000,72100)


#只画测试和预测的
plt.plot(x,y_test[0:100]/100,label='True',color="gray",linestyle="-")
plt.plot(x,ypre[0:100]/100,label="Pre",linestyle="--") 


plt.ylim(0,5.5)
plt.xlabel("Time/h",size=12,weight="bold") #xlabel、ylabel：分别设置X、Y轴的标题文字。
plt.ylabel("TEC/100*1e-1*TECU",size=12,weight="bold")
# plt.grid(axis='y')
plt.legend(loc='best',prop={'size': 12})


plt.show()


relative_error = 0.
for i in range(100):
    relative_error += (abs(ypre[i] - y_test[i]) / y_test[i]) ** 2
acc = 1- np.sqrt(relative_error / 100)
print(f'模型的测试准确率为：{acc*100:.2f}%')

loss=history.history['loss']
val_loss=history.history['val_loss']
plt.plot(loss,label='Training Loss')
plt.plot(val_loss,'o--',label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()