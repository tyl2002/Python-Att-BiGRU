import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

filename = 'D:\project\pythonProject2\Japan.csv'
df = pd.read_csv(r'D:\project\pythonProject2\Japan.csv')
data = df[df['sector'] == 'power']
data.Timestamp = pd.to_datetime(data.date, format='%d-%m-%Y %H:%M')
data.index = data.date
train = data['01/01/2019':'01/01/2022']
test = data['01/01/2022':'28/05/2023']
train.tail()
# 可视化
train.Count.plot(figsize=(15, 8), title='daily passages', fontsize=14)
test.Count.plot(figsize=(15, 8), title='daily passages', fontsize=14)
plt.show()
# print(type(df[1:2]['date']))
# print(type(df.loc[1, 'date']))
#
# # for i in pd.unique(data['sector']):
# plt.rcParams["font.sans-serif"] = "KaiTi"
# plt.rcParams["axes.unicode_minus"] = False
# plt.plot(df.loc[:30, ].date, df.loc[:30, ].value)
# plt.title('japan')
# plt.xlabel('date')
# plt.ylabel('value')
# plt.show()
