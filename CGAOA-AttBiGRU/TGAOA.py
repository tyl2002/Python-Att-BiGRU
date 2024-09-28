
import numpy as np
import random
import copy


'''一种引入过渡阶段和高斯变异的改进算术优化算法(TGAOA) '''
''' '''
'''

'''
''' 种群初始化函数 '''


def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]

    return X, lb, ub


'''边界检查函数'''


def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''计算适应度函数'''


def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness


'''适应度排序'''


def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index


'''根据适应度对位置进行排序'''


def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


'''改进算术优化算法'''


def TGAOA(pop, dim, lb, ub, MaxIter, fun):

    MOP_Max = 1  # 加速度最大值
    MOP_Min = 0.2  # 加速度最小值
    Alpha = 5  # 敏感参数
    Mu = 0.4999  # 控制参数
    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    minIndex = np.argmin(fitness)
    GbestScore = copy.copy(fitness[minIndex])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[minIndex, :])
    Curve = np.zeros([MaxIter, 1])
    Xnew = copy.copy(X)
    fitnessNew = copy.copy(fitness)
    for t in range(MaxIter):
#         print("第"+str(t)+"次迭代")
        MOP = 1 - (t**(1/Alpha)/(MaxIter)**(1/Alpha))  # 数学优化器概率
        # 改进点：重构数学加速优化器
        if t<=2*MaxIter/3:
            MOA=0.5*np.cos(3*np.pi*t/(2*MaxIter))+0.5
        else:
            MOA=0
        if t<=MaxIter/3:
            MOB=0
        else:
            MOB=0.5*np.cos(3*np.pi*t/(2*MaxIter))+0.5
        for i in range(pop):
            Xmean=np.sum(X,axis=0)/pop #求平均
            for j in range(dim):
                r1 = np.random.random()
                if r1 < MOA:  # 乘除算子
                    r2 = np.random.random()
                    if r2 > 0.5:
                        Xnew[i, j] = GbestPositon[0, j] / \
                            (MOP+np.spacing(1))*((ub[j]-lb[j])*Mu+lb[j])
                    else:
                        Xnew[i, j] = GbestPositon[0, j] * \
                            MOP*((ub[j]-lb[j])*Mu+lb[j])
                # 改进点：新策略的引入
                elif MOA<=r1 and r1<MOB: #过渡阶段
                    Xnew[i,j]=GbestPositon[0, j]+2*MOP*(X[i,j]-Xmean[j])
                else:
                    r3 = np.random.random()
                    if r3 > 0.5:
                        Xnew[i, j] = GbestPositon[0, j] - \
                            MOP*((ub[j]-lb[j])*Mu+lb[j])
                    else:
                        Xnew[i, j] = GbestPositon[0, j] + \
                            MOP*((ub[j]-lb[j])*Mu+lb[j])
        Xnew = BorderCheck(Xnew, ub, lb, pop, dim)
        fitnessNew = CaculateFitness(Xnew, fun)  # 计算适应度值
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                X[i, :] = copy.copy(Xnew[i, :])
                fitness[i] = copy.copy(fitnessNew[i])
        #改进点：具有一致性的高斯变异策略
        for i in range(pop):
            r4=np.random.random()
            if r4<=0.5:
                Xnew[i,:]=2*MOP*np.random.randn()*X[i,:]
            else:
                Xnew[i,:]=2*MOP*np.random.randn()+X[i,:]
        Xnew = BorderCheck(Xnew, ub, lb, pop, dim)
        fitnessNew = CaculateFitness(Xnew, fun)  # 计算适应度值
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                X[i, :] = copy.copy(Xnew[i, :])
                fitness[i] = copy.copy(fitnessNew[i])
        #改进点：具有一致性的边界函数变异策略
        for i in range(pop):
            r4=np.random.random()
            if r4<=0.5:
                Xnew[i,:]=(ub.T-lb.T)*np.random.random()*X[i,:]
            else:
                Xnew[i,:]=(ub.T-lb.T)*np.random.random()+X[i,:]
        for i in range(pop):
            if fitnessNew[i] < fitness[i]:
                X[i, :] = copy.copy(Xnew[i, :])
                fitness[i] = copy.copy(fitnessNew[i])
        minIndex = np.argmin(fitness)
        if fitness[minIndex] < GbestScore:
            GbestScore = copy.copy(fitness[minIndex])
            GbestPositon[0, :] = copy.copy(X[minIndex, :])
        # 改进点：自适应t分布改进最优位置
        Temp = np.zeros([1, dim])
        Temp[0, :] = GbestPositon[0, :] + \
            GbestPositon[0, :]*np.random.standard_t(t+1)
        for j in range(dim):
            if Temp[0, j] > ub[j]:
                Temp[0, j] = ub[j]
            if Temp[0, j] > lb[j]:
                Temp[0, j] = lb[j]
        fTemp = fun(Temp[0, :])
        if fTemp < GbestScore:
            GbestScore = fTemp
            GbestPositon[0, :] = copy.copy(Temp[0, :])
            X[minIndex, :] = copy.copy(Temp[0, :])
            fitness[minIndex] = fTemp

        Curve[t] = GbestScore
        

    return GbestScore, GbestPositon, Curve
