import tensorflow

tensorflow.random.set_random_seed(5)
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
from tensorflow.keras.utils import plot_model

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
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

china_v = pd.read_csv('D:\project\pythonProject2\Japan.csv')
china_data = china_v['value']
china_v1 = china_data[:1610]
# # china_v1 = china_data[1614:3224]
# china_v1 = china_data[3226:4836]
data = np.array(china_v1)
data = data.reshape(-1, 1)


def first_diff(data):
    """
    计算给定数据列表的相邻元素之间的差值。

    参数:
    data: 一个包含数字元素的列表或数组，可以是浮点数或整数。

    返回值:
    一个numpy数组，包含输入数据中相邻元素之间的差值，数据类型为float64。
    """
    # 初始化用于存储差值的列表
    data_diff = []
    # 计算相邻元素之间的差值，并添加到data_diff列表中
    for i in range(len(data) - 1):
        data_diff.append(data[i + 1] - data[i])
    # 将data_diff列表转换为numpy数组
    data_diff = np.array(data_diff)
    # 将数组的数据类型转换为float64并返回
    return data_diff.astype(np.float64)


def anti_first_diff(m, n):
    data = []
    for i in range(0, len(m)):
        a = n[i] + m[i]
        data.append(a)
    return np.array(data).astype("float32")


def split_sequence(sequence, look_back, forecast_horizon):
    X, y = list(), list()
    for i in range(len(sequence)):
        lag_end = i + look_back
        forecast_end = lag_end + forecast_horizon
        if forecast_end > len(sequence):
            break
        seq_x, seq_y = sequence[i:lag_end], sequence[lag_end:forecast_end]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def inverse_transform(y_test, yhat):
    y_test_reshaped = y_test.reshape(-1, y_test.shape[-1])
    yhat_reshaped = yhat.reshape(-1, yhat.shape[-1])
    yhat_inverse = scaler.inverse_transform(yhat_reshaped)
    y_test_inverse = scaler.inverse_transform(y_test_reshaped)
    return yhat_inverse, y_test_inverse


def rmse(x, y):
    return tf.sqrt(tf.reduce_mean(tf.square((x - y))))


# 让模型侧重于输入维度的第几个维度，attention_column，目前是10,第11维
def get_data_recurrent(x, y, attention_column=7):
    y = y.reshape(y.shape[0], 1)
    x[:, attention_column, :] = np.tile(y[:], (1, 1))
    return x, y


def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(LOOK_BACK, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    # 相当于获得每一个step中，每个特征的权重
    output_attention_mul = merge.multiply([inputs, a_probs])  # 新版本 keras
    return output_attention_mul


def get_attention_model(time_steps, input_dim, lstm_units=64):
    inputs1 = Input(shape=(time_steps, input_dim))
    lstm_out = GRU(lstm_units, return_sequences=True)(inputs1)
    # lstm_out = Dropout(0.8)(lstm_out)  # 添加Dropout层
    # attention_mul = attention_3d_block(lstm_out)
    # attention_mul = Flatten()(attention_mul)
    attention_mul = Flatten()(lstm_out)
    output = Dense(1)(attention_mul)
    model = Model(inputs=[inputs1], outputs=output)
    return model


LOOK_BACK = 12
FORECAST_RANGE = 1
tec_diff = first_diff(np.array(data))  # 进行差分进行平稳化
split_size = int(len(tec_diff) * 0.7)  # 前70%为训练集后30%为测试集
train, test = tec_diff[:split_size], tec_diff[split_size:]
scaler = MinMaxScaler()  # 归一化：归一化的目的就是使得预处理的数据被限定在一定的范围内（比如[0,1]或者[-1,1]），从而消除奇异样本数据导致的不良影响。
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)  # fit_transform()的作用就是先拟合数据，然后转化它将其转化为标准形式
x_train, y_train = split_sequence(scaled_train, look_back=LOOK_BACK,
                                  forecast_horizon=FORECAST_RANGE)  # fit_transform后使用前12个值预测第13个值
x_test, y_test = split_sequence(scaled_test, look_back=LOOK_BACK,
                                forecast_horizon=FORECAST_RANGE)  # fit_transform后使用前12个值预测第13个值
xa, ya = split_sequence(np.array(data[int(len(tec_diff) * 0.7) + 1:]), look_back=LOOK_BACK,
                        forecast_horizon=FORECAST_RANGE)  # 不进行归一化
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=20)  # 早停回调
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

INPUT_DIMS = 1
TIME_STEPS = 12
start = time.perf_counter()
tf.random.set_random_seed(5)
model = get_attention_model(LOOK_BACK, INPUT_DIMS)
X, Y = get_data_recurrent(x_train, y_train)  # 将数据拼接
optimizer = tf.keras.optimizers.SGD(learning_rate=0.005)  # 优化器 SGD
model.compile(optimizer='SGD', loss='mae')  # 实际优化器
history = model.fit(X, Y, epochs=100, callbacks=[early_stop], batch_size=64, verbose=2, shuffle=False,
                    validation_split=0.1)
model.summary()  # 输出模型各层的参数状况
# plot_model(model,to_file='model.png',show_shapes=True)
using_time = time.perf_counter() - start
yhat = model.predict(x_test, verbose=0)  # 测试集的预测试
yhat_inverse, y_test_inverse = inverse_transform(y_test, yhat)  # 预测值反归一化
y_pre = []
for i in range(yhat_inverse.shape[0]):
    for j in range(yhat_inverse.shape[1]):
        y_pre.append(yhat_inverse[i][j])
        y = np.array(y_test_inverse)

y_pre = np.array(y_pre)  # 将y_pre转化为numpy数组
y = anti_first_diff(ya, y)
y_pre = anti_first_diff(ya, y_pre)  # y是实际值，y_pre是预测值

y_r, y_p = [], []  # y_r 真实值，y_p 预测值
for i in range(0, y.shape[0]):
    for j in range(0, y.shape[1]):
        y_r.append(y[i][j])
        y_p.append(y_pre[i][j])
y_r1 = np.array(y_r)
y_p1 = np.array(y_p)
y_r2 = data[1139:]
x = np.arange(0, 1610, 1)
MAE = mean_absolute_error(y_r2, y_p1)
MSE = mean_squared_error(y_r2, y_p1)
RMSE = sqrt(mean_squared_error(y_r2, y_p1))
R2_score = r2_score(y_r2, y_p1)
output_value = []
output_value.append(MAE)
output_value.append(MSE)
output_value.append(RMSE)
output_value.append(R2_score)
output_value.append(int(using_time))
output_value = np.array(output_value)
output_data = pd.DataFrame(output_value)

output_data.index = ['MAE', 'MSE', 'RMSE', 'R2 Score', 'Time-consuming(s)']
print(output_data)
