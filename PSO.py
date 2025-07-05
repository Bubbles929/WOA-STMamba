# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 20:55:34 2021
PSO优化网络的超参数
@author: shenya
"""

'粒子群优化算法'
import numpy
import math
import time
import numpy as np
import matplotlib.pyplot as plt

'表示粒子的类'
class Particle():
    def __init__(self, dim):
        self.dim = dim
        self.V = np.zeros(self.dim)  # 粒子速度
        self.X = np.zeros(self.dim)  # 当前位置,即外层网络给内层的超参数
        self.fit = np.inf  # 当前超参数下,内层网络优化出的适应度值
        self.solution = np.empty  # 当前超参数下,内层网络对优化问题求得的解
        self.pBest_X = np.zeros(self.dim)  # 当前粒子的历史最佳位置,即历史最佳超参数
        self.pBest_fit = np.inf  # 历史最佳超参数下,内层网络优化出的适应度值

class PSO():
    # PSO参数设置
    def __init__(self, pop_num, max_iter, fitFunc, dim, lb, ub):
        self.pop_num = pop_num
        self.max_iter = max_iter
        self.fitFunc = fitFunc
        self.dim = dim
        self.lb = lb  # 变量下界
        self.ub = ub  # 变量上界
        self.X = []  # 列表存储所有的粒子对象
        self.gBest_X = np.zeros(self.dim)  # 外层网络给内层的最佳超参数
        self.gBest_fit = np.inf  # 当前最佳超参数下,内层网络优化出的适应度值
        # PSO算法的内部参数
        self.c1 = 2
        self.c2 = 2
        self.wMax = 0.9
        self.wMin = 0.2
        self.vMax = (self.ub - self.lb) * 0.2  # 粒子速度边界
        self.vMin = -self.vMax

    '''检查超参数边界、取整并取精度为e-8'''
    def check_round(self, X):
        X = np.clip(X, self.lb, self.ub)  # 首先限制边界
        for i in range(self.dim):
            #if i == 0:
            if i == 1:
                X[i] = round(X[i], 6)  # lr取到小数点后6位
            else:
                X[i] = int(X[i])  # epoch; batch_size; cfc_hidden_size应该进行取整, math.trunc()截断取整
        return X

    '初始化种群'
    def init_population(self):
        for i in range(self.pop_num):
            par = Particle(self.dim)
            for j in range(self.dim):
                par.X[j] = np.random.uniform() * (self.ub[j] - self.lb[j]) + self.lb[j]
            self.X.append(par)
            print(self.X[i].X)
        print('------------------------------\n')

        for i in range(self.pop_num):  # 计算初始种群适应度
            self.X[i].X = self.check_round(self.X[i].X)  # 计算适应度前,限制范围,并对整数超参求整
            tmp_fit = self.fitFunc(self.X[i].X)
            # fit = tmp_fit.cpu().numpy() # 将cuda上的数据复制到cpu再转为numpy
            fit = tmp_fit
            self.X[i].fit = fit
            print('The ' + str(i + 1) + 'th particle has be initialized')
            if fit <= self.gBest_fit:  # 可以看出是求最小化问题,更新pBest和gBest
                self.gBest_X = self.X[i].X
                self.gBest_fit = self.X[i].fit
            if fit <= self.X[i].pBest_fit:
                self.X[i].pBest_X = self.X[i].X
                self.X[i].pBest_fit = self.X[i].fit
        print('------------------------------\n')

        print('The initial population as following:')
        for i in range(self.pop_num):
            print(self.X[i].X)
        print('------------------------------\n')

    '''#优化模块'''  
    def optimization(self):
        t = 1
        fitness = []
        start = time.perf_counter()  # 记录优化算法开始时间
        while t <= self.max_iter:
            print('#################### At the ' + str(t) + 'th iteration ####################')
            weight = self.wMax - (self.wMax - self.wMin) * (t / self.max_iter)  # 自适应的惯性权重
            # w = 2*(self.max_iter - t)/self.max_iter
            # w = np.exp(1) - np.exp((t/self.max_iter)**1.5)
            '''#Update the location of PSO'''
            for i in range(self.pop_num):
                for j in range(self.dim):
                    self.X[i].V[j] = weight*self.X[i].V[j] + \
                                     self.c1*np.random.uniform()*(self.X[i].pBest_X[j]-self.X[i].X[j])\
                                     + self.c2*np.random.uniform()*(self.gBest_X[j]-self.X[i].X[j])
                np.clip(self.X[i].V, self.vMin, self.vMax)
                self.X[i].X = self.X[i].X + self.X[i].V

            '''#Calculate the fitness of PSO'''
            for i in range(self.pop_num):
                self.X[i].X = self.check_round(self.X[i].X)  # 计算适应度前需要限制边界，并局部求整
                tmp_fit = self.fitFunc(self.X[i].X)
                fit = tmp_fit
                print('The fitness of '+str(i+1)+'th particle at '+str(t)+'th iteration has be calculated')
                self.X[i].fit = fit
                if fit <= self.gBest_fit:  # 更新gBest
                    self.gBest_X = self.X[i].X
                    self.gBest_fit = self.X[i].fit
                if fit <= self.X[i].pBest_fit:  # 更新pBest
                    self.X[i].pBest_X = self.X[i].X
                    self.X[i].pBest_fit = self.X[i].fit
            print('------------------------------\n')

            fitness.append(self.gBest_fit)
            t += 1
            time_during = time.perf_counter() - start  # 记录优化算法每次迭代结束时间

            print(f"""Current Best hyper-para:\n {self.gBest_X}
                \nCurrent Best fitness:\n {self.gBest_fit} \n""")

        return fitness, time_during, self.gBest_X, self.gBest_fit


def main():
    print('############### PSO starts the optimization process ###############\n')
    from trainpso import main as fitFunc
    # x1:batch_size[1, 100]; x2:epoch[1, 200]; x3:cfc_hidden_size[6,128] 小一些, x4:Lr[0.00001,0.005]
    # lb_array = np.array([1, 1, 6, 0.00001])
    # ub_array = np.array([200, 100, 20, 0.005])
    # x1:cfc_hidden_size[32, 128]; x2:epoch[30, 50]; x3:Lr[0.0001,0.0003] , x4:weight_decay[0.0004,0.003]
    lb_array = np.array([32, 0.0009, 1, 2, 1, 12, 1])
    ub_array = np.array([256, 0.004, 32, 4, 12, 144, 64])
    my_pso = PSO(pop_num=10, max_iter=10, fitFunc=fitFunc, dim=np.size(lb_array), lb=lb_array, ub=ub_array)
    my_pso.init_population()
    fitness, time_during, gBest_X, gBest_fit = my_pso.optimization()
    print('############### Optimization has completed ###############\n')
    print('Time during:\n', time_during)
    print('Global Best_hyper-para:\n', gBest_X)
    print('Global Best_fitness:\n', gBest_fit)

    # 画图
    plt.figure(1)
    plt.title("PSO")
    plt.xlabel("iterations", size=14)
    plt.ylabel("fitness", size=14)
    plt.plot(fitness, color='b', linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    main()



