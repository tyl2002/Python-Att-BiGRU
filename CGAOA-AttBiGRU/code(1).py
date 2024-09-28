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

accuracy = tf.keras.metrics.Accuracy()

china_v = pd.read_csv('china_data.csv')
china_data = china_v['value']
china_v1 = china_data[:1610]
# china_v1=china_data[1615:3220]
# china_v1=china_data[3227:4836]
# china_v1=china_data[4839:6447]
data = np.array(china_v1)
data = data.reshape(-1, 1)


def first_diff(data):
    data_diff = []
    for i in range(len(data) - 1):
        data_diff.append(data[i + 1] - data[i])
    data_diff = np.array(data_diff)
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


def MAPE(y_true, y_pred):
    return K.mean(K.abs((y_true - y_pred) / y_true)) * 100


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # 避免除以零，可加入一个小数值
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def attention_3d_block(inputs):
    a = Permute((2, 1))(inputs)
    a = Dense(LOOK_BACK, activation='softmax')(a)
    a_probs = Permute((2, 1))(a)
    # 相当于获得每一个step中，每个特征的权重
    output_attention_mul = merge.multiply([inputs, a_probs])  # 新版本 keras
    return output_attention_mul


def get_attention_model(time_steps, input_dim, lstm_units=40):
    inputs1 = Input(shape=(time_steps, input_dim))
    #     lstm_out = Bidirectional(GRU(lstm_units, activity_regularizer=regularizers.l1(0.6),return_sequences=True,name="Bidirectional_1"))(inputs1)
    lstm_out = Bidirectional(GRU(lstm_units, return_sequences=True))(inputs1)
    lstm_out = Dropout((0.2))(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)
    output = Dense(1, activation='relu')(attention_mul)
    model = Model(inputs=[inputs1], outputs=output)

    #     inputs1 = Input(shape=(time_steps, input_dim))
    #     attention_mul = attention_3d_block(inputs1)
    #     attention_mul=Bidirectional(GRU(lstm_units, return_sequences=True))(attention_mul)
    #     output = Dense(1,activation='relu')(attention_mul)
    #     model = Model(inputs=[inputs1], outputs=output)

    return model


LOOK_BACK = 12
FORECAST_RANGE = 1
tec_diff = first_diff(np.array(data))
split_size = int(len(tec_diff) * 0.7)
train, test = tec_diff[:split_size], tec_diff[split_size:]
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train)
scaled_test = scaler.transform(test)
x_train, y_train = split_sequence(scaled_train, look_back=LOOK_BACK, forecast_horizon=FORECAST_RANGE)
x_test, y_test = split_sequence(scaled_test, look_back=LOOK_BACK, forecast_horizon=FORECAST_RANGE)
xa, ya = split_sequence(np.array(data[int(len(tec_diff) * 0.7) + 1:]), look_back=LOOK_BACK,
                        forecast_horizon=FORECAST_RANGE)
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

INPUT_DIMS = 1
TIME_STEPS = 12
start = time.perf_counter()
model = get_attention_model(LOOK_BACK, INPUT_DIMS)
X, Y = get_data_recurrent(x_train, y_train)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.045)
# optimizer =tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mae')
# model.fit(X, Y, epochs=50, batch_size=64, validation_split=0.1)
history = model.fit(X, Y, epochs=50, callbacks=[early_stop], batch_size=64, verbose=2, shuffle=False,
                    validation_split=0.1)
model.summary()

using_time = time.perf_counter() - start
yhat = model.predict(x_test, verbose=0)
yhat_inverse, y_test_inverse = inverse_transform(y_test, yhat)
y_pre = []
for i in range(yhat_inverse.shape[0]):
    for j in range(yhat_inverse.shape[1]):
        y_pre.append(yhat_inverse[i][j])
        y = np.array(y_test_inverse)

y_pre = np.array(y_pre)
y = anti_first_diff(ya, y)
y_pre = anti_first_diff(ya, y_pre)

y_r, y_p = [], []
for i in range(0, y.shape[0]):
    for j in range(0, y.shape[1]):
        y_r.append(y[i][j])
        y_p.append(y_pre[i][j])
y_r1 = np.array(y_r)
y_p1 = np.array(y_p)
y_r2 = data[1139:]  # power

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
