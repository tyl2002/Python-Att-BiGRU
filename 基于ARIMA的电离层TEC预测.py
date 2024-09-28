#!/usr/bin/env python
# coding: utf-8

#1.导入数据
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf  #自相关图、偏自相关图
from statsmodels.tsa.stattools import adfuller as ADF #平稳性检验
from statsmodels.stats.diagnostic import acorr_ljungbox #白噪声检验
import statsmodels.api as sm #D-W检验,一阶自相关检验
from statsmodels.graphics.api import qqplot #画QQ图,检验一组数据是否服从正态分布
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults

dta=[10930,10318,10595,10972,7706,6756,9092,10551,9722,10913,11151,8186,6422, 
6337,11649,11652,10310,12043,7937,6476,9662,9570,9981,9331,9449,6773,6304,9355, 
10477,10148,10395,11261,8713,7299,10424,10795,11069,11602,11427,9095,7707,10767, 
12136,12812,12006,12528,10329,7818,11719,11683,12603,11495,13670,11337,10232, 
13261,13230,15535,16837,19598,14823,11622,19391,18177,19994,14723,15694,13248, 
9543,12872,13101,15053,12619,13749,10228,9725,14729,12518,14564,15085,14722, 
11999,9390,13481,14795,15845,15271,14686,11054,10395]
dta=np.array(dta,dtype=np.float) #这里要转一下数据类型，不然运行会报错
#numpy转csv
#dta_new=pd.DataFrame(dta).to_csv(r"C:\Users\86156\Desktop\dta_new.csv")


dta=pd.Series(dta)
dta.index=pd.Index(sm.tsa.datetools.dates_from_range('1901','1990'))
dta.plot(figsize=(12,8))


#3.ADF单位根检验
from statsmodels.tsa.stattools import adfuller
result = adfuller(dta)
print('原始序列的ADF平稳性检验结果为：',result)


# 2.时间序列的差分
# ARIMA模型对时间序列的要求是平稳型，当得到一个非平稳的时间序列时
# 要进行时间序列的差分，如果对时间序列做d次差分才能得到一个平稳序列，可以使用ARIMA(p,d,q)模型
# 其中d是差分次数

fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(111)
diff1=dta.diff(1)
diff1.plot(figsize=(12,8))

diff2=diff1.diff(1)
diff2.plot(figsize=(12,8))


diff1[np.isnan(diff1)] = 0
diff1[np.isinf(diff1)] = 0

#3.ADF单位根检验
#观察一阶差分图像判断数据平稳性的方式有一定主观性，因此进一步采用.ADF单位根检验；确定d值（差分阶数）
result1 = adfuller(diff1)
print('一阶差分序列的ADF平稳性检验结果为：',result1)


diff2[np.isnan(diff2)] =0
diff2[np.isinf(diff2)] = 0


#3.ADF单位根检验
#观察二阶差分图像判断数据平稳性的方式有一定主观性，因此进一步采用.ADF单位根检验；确定d值（差分阶数）
result2 = adfuller(diff2)
print('二阶差分序列的ADF平稳性检验结果为：',result2)


#4.白噪声检验
#对白噪声的平稳性进行检验
from statsmodels.stats.diagnostic import acorr_ljungbox
print('数据白噪声检验的结果')
print(acorr_ljungbox(diff1,lags=[6,12,24],return_df=True))


# 各阶滞后项均小于0.05，拒绝为白噪声的原假设


fig=plt.figure(figsize=(12,8))
ax1=fig.add_subplot(211)
fig=sm.graphics.tsa.plot_acf(diff1,lags=30,ax=ax1)
ax2=fig.add_subplot(212)
fig=sm.graphics.tsa.plot_pacf(diff1,lags=30,ax=ax2)


# 如果一个序列中有较多自相关系数的值在边界之外，那么该序列很可能不是白噪声序列。

# 其中lags表示滞后的阶数，以上分别得到acf图和pacf图

# 自相关图显示滞后有三个阶超出了置信边界
# 偏自相关图显示滞后1至7阶(lags1,2,....,7)时的偏自相关系数超出了置信边界，从lag7滞后偏自相关系数缩小至0

# 模型选择，可以选择ARMA(7,0)模型，ARMA(7,1)模型，ARMA（8,0）,ARMA(8,1)模型
# 采用AIC准则，增加自由参数的目的提高了拟合的优良性，AIC鼓励数据拟合的优良性但是尽量避免出现过度拟合的情况


#选择合适的ARMA模型
arma_mod01=sm.tsa.ARMA(diff1,(0,1)).fit()
print(arma_mod01.aic,arma_mod01.bic,arma_mod01.hqic)
arma_mod02=sm.tsa.ARMA(diff1,(0,2)).fit()
print(arma_mod02.aic,arma_mod02.bic,arma_mod02.hqic)
arma_mod22=sm.tsa.ARMA(diff1,(2,2)).fit()
print(arma_mod22.aic,arma_mod22.bic,arma_mod22.hqic)
arma_mod03=sm.tsa.ARMA(diff1,(0,3)).fit()
print(arma_mod03.aic,arma_mod03.bic,arma_mod03.hqic)


from pmdarima.arima import auto_arima      
model1=auto_arima(diff1,start_p=0,start_q=0,max_p=9,max_q=9,m=12,start_P=0,trace = True,error_action ='ignore',suppress_warnings = True,stepwise=True)
model1.fit(diff1)


model=ARIMA(diff1,order=(1,0,1)).fit()
#残差检验
resid=model.resid
#1
#自相关图
plot_acf(resid,lags=50).show()
#解读：有短期相关性，但趋向于零。
#偏自相关图
plot_pacf(resid,lags=50).show()
#偏自相关图
plot_pacf(resid,lags=10).show()

#2 qq图
qqplot(resid, line='q', fit=True).show()

#3 DW检验
print('D-W检验的结果为：',sm.stats.durbin_watson(resid.values))
#解读：不存在一阶自相关

#4 LB检验
print('残差序列的白噪声检验结果为：',acorr_ljungbox(resid,lags=[6,12,24]))#返回统计量、P值
#解读：残差是白噪声 p>0.05
# confint,qstat,pvalues = sm.tsa.acf(resid.values, qstat=True)


from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(diff1,order=(1,0,1)).fit(disp=-1)
print(model.summary())


fig,ax=plt.subplots(figsize=(12,10))
ax=diff1.loc['1901':].plot(ax=ax)
fig=model.plot_predict(5,100)
plt.show()