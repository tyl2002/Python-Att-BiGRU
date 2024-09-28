'''算术优化算法'''
import numpy as np
import random
import copy


''' 种群初始化函数 '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random()*(ub[j] - lb[j]) + lb[j]
    
    return X,lb,ub
            
'''边界检查函数'''
def BorderCheck(X,ub,lb,pop,dim):
    for i in range(pop):
        for j in range(dim):
            if X[i,j]>ub[j]:
                X[i,j] = ub[j]
            elif X[i,j]<lb[j]:
                X[i,j] = lb[j]
    return X
    
    
'''计算适应度函数'''
def CaculateFitness(X,fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness

'''适应度排序'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness,index


'''根据适应度对位置进行排序'''
def SortPosition(X,index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i,:] = X[index[i],:]
    return Xnew


'''算术优化算法'''
def AOA(pop,dim,lb,ub,MaxIter,fun):
    
    MOP_Max= 1 #加速度最大值
    MOP_Min = 0.2 #加速度最小值
    Alpha = 5   #敏感参数
    Mu = 0.4999 #控制参数
    X,lb,ub = initial(pop, dim, ub, lb) #初始化种群
    fitness = CaculateFitness(X,fun) #计算适应度值
    minIndex = np.argmin(fitness)
    GbestScore = copy.copy(fitness[minIndex])
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[minIndex,:])
    Curve = np.zeros([MaxIter,1])
    Xnew = copy.copy(X)
    fitnessNew = copy.copy(fitness)
    for t in range(MaxIter):
#         print("第"+str(t)+"次迭代")
        MOP =  1 - (t**(1/Alpha)/(MaxIter)**(1/Alpha)) #数学优化器概率
        MOA = MOP_Min+t*((MOP_Max-MOP_Min)/MaxIter) #加速函数
        for i in range(pop):
            for j in range(dim):
                r1=np.random.random()
                if r1<MOA: #乘除算子
                    r2=np.random.random()
                    if r2>0.5:
                        Xnew[i,j]=GbestPositon[0,j]/(MOP+np.spacing(1))*((ub[j]-lb[j])*Mu+lb[j])
                    else:
                        Xnew[i,j]=GbestPositon[0,j]*MOP*((ub[j]-lb[j])*Mu+lb[j])
                else:#加减算子
                    r3=np.random.random()
                    if r3>0.5:
                        Xnew[i,j]=GbestPositon[0,j]-MOP*((ub[j]-lb[j])*Mu+lb[j])
                    else:
                        Xnew[i,j]=GbestPositon[0,j]+MOP*((ub[j]-lb[j])*Mu+lb[j])

        Xnew=BorderCheck(Xnew,ub,lb,pop,dim)
        fitnessNew = CaculateFitness(Xnew,fun) #计算适应度值
        for i in range(pop):
            if fitnessNew[i]<fitness[i]:
                X[i,:]=copy.copy(Xnew[i,:])
                fitness[i] = copy.copy(fitnessNew[i])
        
       
        minIndex = np.argmin(fitness)
        if fitness[minIndex]<GbestScore:
            GbestScore = copy.copy(fitness[minIndex])
            GbestPositon[0,:] = copy.copy(X[minIndex,:])
    
        Curve[t] = GbestScore
    
    return GbestScore,GbestPositon,Curve









