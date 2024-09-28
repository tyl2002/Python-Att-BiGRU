import numpy as np
import random
import math
import copy

''' 种群初始化函数 '''


def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

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

''' Levy飞行'''


def Levy(d):
    beta = 3/2
    sigma = (math.gamma(1 + beta)*np.sin(math.pi*beta/2)) / \
        (math.gamma((1 + beta)/2)*beta*2**((beta-1)/2))**(1/beta)
    u = np.random.randn(1, d)*sigma
    v = np.random.randn(1, d)
    step = u/np.abs(v)**(1/beta)
    L = 0.05*step
    return L

'''白鲸优化算法'''
'''白鲸优化算法，write byJack旭:https://mbd.pub/o/JackYM'''
'''如需其他代码请访问：链接：https://pan.baidu.com/s/1QIHWRh0bNfZRA8KCQGU8mg 提取码：1234'''

def BWO(pop, dim, lb, ub, MaxIter, fun):

    X, lb, ub = initial(pop, dim, ub, lb)  # 初始化种群
    fitness = CaculateFitness(X, fun)  # 计算适应度值
    indexBest = np.argmin(fitness)
    GbestScore = copy.copy(fitness[indexBest])
    GbestPositon = np.zeros([1, dim])
    GbestPositon[0, :] = copy.copy(X[indexBest, :])
    Curve = np.zeros([MaxIter, 1])
    Xnew = copy.deepcopy(X)
    fitNew = copy.deepcopy(fitness)
    for t in range(MaxIter):
        WF = 0.1-0.05*(t/MaxIter) # 鲸落的概率
        # 第一阶段
        for i in range(pop):
            k=(1-0.5*t/MaxIter)*np.random.random() # 开发与探索切换概率
            if k>0.5: # 探索
                for j in range(dim):
                    r1=np.random.random()
                    r2=np.random.random()
                    Index=np.random.randint(pop)
                    if j%2: #j为奇数
                        Xnew[i,j]=X[i,j]+(X[Index,j]-X[i,j])*(r1+1)*np.sin(r2*2*np.pi)
                    else:   # j 为偶数
                        Xnew[i,j]=X[i,j]+(X[Index,j]-X[i,j])*(r1+1)*np.cos(r2*2*np.pi)
            else: #开发
                r3=np.random.random()
                r4=np.random.random()
                C1=2*r4*(1-t/MaxIter)
                Index=np.random.randint(pop)
                Xnew[i,:]=r3*GbestPositon-r4*X[i,:]+C1*Levy(dim)*(X[Index,:]-X[i,:])
            for j in range(dim):
                if Xnew[i,j]>ub[j]:
                    Xnew[i,j]=ub[j]
                if Xnew[i,j]<lb[j]:
                    Xnew[i,j]=lb[j]
            fitNew[i]=fun(Xnew[i,:])
            if fitNew[i]<fitness[i]:
                X[i,:]=copy.copy(Xnew[i,:])
                fitness[i]=copy.copy(fitNew[i])
        # 鲸落
        for i in range(pop):
            k=(1-0.5*t/MaxIter)*np.random.random()
            if k<=WF:
                r5=np.random.random()
                r6=np.random.random()
                r7=np.random.random()
                C2=2*pop*WF
                Index=np.random.randint(pop)
                stepsize=r7*(ub.T-lb.T)*np.exp(-C2*t/MaxIter)
                Xnew[i,:]=(r5*X[i,:]-r6*X[Index,:])+stepsize
                for j in range(dim):
                    if Xnew[i,j]>ub[j]:
                        Xnew[i,j]=ub[j]
                    if Xnew[i,j]<lb[j]:
                        Xnew[i,j]=lb[j]
                fitNew[i]=fun(Xnew[i,:])
                if fitNew[i]<fitness[i]:
                    X[i,:]=copy.copy(Xnew[i,:])
                    fitness[i]=copy.copy(fitNew[i])
        indexBest = np.argmin(fitness)
        if fitness[indexBest] <= GbestScore:  # 更新全局最优
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0, :] = copy.copy(X[indexBest, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon, Curve
