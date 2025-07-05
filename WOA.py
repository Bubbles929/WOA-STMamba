# -*- coding: utf-8 -*-
"""
Created on Mon May 16 14:19:49 2016

@author: hossam
"""
import random
import numpy
import math

import numpy as np
from matplotlib import pyplot as plt

from SI.genlife.solution import solution
import time


def WOA(objf, lb, ub, dim, SearchAgents_no, Max_iter):

    # dim=30
    # SearchAgents_no=50
    # lb=-100
    # ub=100
    # Max_iter=500
    # if not isinstance(lb, list):
    #     lb = [lb] * dim
    # if not isinstance(ub, list):
    #     ub = [ub] * dim

    # initialize position vector and score for the leader
    Leader_pos = numpy.zeros(dim)
    Leader_score = float("inf")  # change this to -inf for maximization problems

    # Initialize the positions of search agents
    Positions = numpy.zeros((SearchAgents_no, dim))
    for i in range(dim):
        Positions[:, i] = (
            numpy.random.uniform(0, 1, SearchAgents_no) * (ub[i] - lb[i]) + lb[i]
        )

    # Initialize convergence
    convergence_curve = numpy.zeros(Max_iter)

    ############################
    s = solution()

    print('WOA is optimizing  "' + objf.__name__ + '"')

    timerStart = time.time()
    s.startTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    ############################

    t = 0  # Loop counter

    # Main loop
    while t < Max_iter:
        for i in range(0, SearchAgents_no):

            print('The fitness of ' + str(i+1) + 'th particle at ' + str(t+1) + 'th iteration has be calculated')
            # Return back the search agents that go beyond the boundaries of the search space

            # Positions[i,:]=checkBounds(Positions[i,:],lb,ub)
            for j in range(dim):
                Positions[i, j] = numpy.clip(Positions[i, j], lb[j], ub[j])
                # Positions[i, j] = min(max(Positions[i, j], lb[j]), ub[j])
            # Calculate objective function for each search agent
            fitness = objf(Positions[i, :])

            # Update the leader
            if fitness < Leader_score:  # Change this to > for maximization problem
                Leader_score = fitness
                # Update alpha
                Leader_pos = Positions[
                    i, :
                ].copy()  # copy current whale position into the leader position

        a = 2 - t * ((2) / Max_iter)
        # a decreases linearly fron 2 to 0 in Eq. (2.3)

        # a2 linearly decreases from -1 to -2 to calculate t in Eq. (3.12)
        a2 = -1 + t * ((-1) / Max_iter)


        # Update the Position of search agents
        for i in range(0, SearchAgents_no):
            r1 = random.random()  # r1 is a random number in [0,1]
            r2 = random.random()  # r2 is a random number in [0,1]

            A = 2 * a * r1 - a  # Eq. (2.3) in the paper
            C = 2 * r2  # Eq. (2.4) in the paper

            b = 1
            #  parameters in Eq. (2.5)
            l = (a2 - 1) * random.random() + 1  #  parameters in Eq. (2.5)

            p = random.random()  # p in Eq. (2.6)

            for j in range(0, dim):

                if p < 0.5:
                    if abs(A) >= 1:
                        rand_leader_index = math.floor(
                            SearchAgents_no * random.random()
                        )
                        X_rand = Positions[rand_leader_index, :]
                        D_X_rand = abs(C * X_rand[j] - Positions[i, j])
                        Positions[i, j] = X_rand[j] - A * D_X_rand

                    elif abs(A) < 1:
                        D_Leader = abs(C * Leader_pos[j] - Positions[i, j])
                        Positions[i, j] = Leader_pos[j] - A * D_Leader

                elif p >= 0.5:

                    distance2Leader = abs(Leader_pos[j] - Positions[i, j])
                    # Eq. (2.5)
                    Positions[i, j] = (
                        distance2Leader * math.exp(b * l) * math.cos(l * 2 * math.pi)
                        + Leader_pos[j]
                    )

        convergence_curve[t] = Leader_score


        if t % 1 == 0:
            # print(
            #     ["At iteration " + str(t) + " the best fitness is " + str(Leader_score)]
            # )
            print(f"At iteration {t+1} the best fitness is {Leader_score}")
            print(f"Current Best hyper-para: {[round(x,6) if i == 1 else round(x) for i, x in enumerate(Leader_pos)]}\n")


        t = t + 1

    timerEnd = time.time()
    s.endTime = time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime = timerEnd - timerStart
    s.convergence = convergence_curve
    s.optimizer = "WOA"
    s.objfname = objf.__name__
    s.best = Leader_score
    s.bestIndividual = Leader_pos

    return s

def main():
    print('############### WOA starts the optimization process ###############\n')
    from trainwoa import main as objf
    # x1:batch_size[1, 100]; x2:epoch[1, 200]; x3:cfc_hidden_size[6,128] 小一些, x4:Lr[0.00001,0.005]
    # lb_array = np.array([1, 1, 6, 0.00001])
    # ub_array = np.array([200, 100, 20, 0.005])
    # x1:cfc_hidden_size[32, 128]; x2:epoch[30, 50]; x3:Lr[0.0001,0.0003] , x4:weight_decay[0.0004,0.003]

    # lb_array = np.array([32, 0.0009, 1, 1, 1, 12, 1])
    # ub_array = np.array([256, 0.004111, 32, 4, 12, 144, 64]) #beifeng
#YUAN
    lb_array = np.array([32, 0.0009, 1, 2, 1, 12, 1])
    ub_array = np.array([256, 0.00411, 32, 4, 12, 176, 128])
###对第一部分消融
    # lb_array = np.array([32, 0.0009])
    # ub_array = np.array([256, 0.00411])
###对第二部分消融
    # lb_array = np.array([1, 2, 1, 12, 1])
    # ub_array = np.array([32, 4, 12, 176, 128])

    # objf, lb, ub, dim, SearchAgents_no, Max_iter
    my_woa = WOA(objf=objf, lb=lb_array, ub=ub_array, dim=7, SearchAgents_no=10, Max_iter=10)
    result = my_woa

    print('############### Optimization has completed ###############\n')
    print('Time during:\n', result.executionTime)
    print('Global Best_hyper-para:\n', [round(x,6) if i == 1 else round(x) for i, x in enumerate(result.bestIndividual)])
    print('Global Best_fitness:\n', result.best)



    # 画图
    plt.figure(1)
    plt.title("WOA")
    plt.xlabel("iterations", size=14)
    plt.ylabel("fitness", size=14)
    plt.plot(result.convergence, color='b', linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    main()