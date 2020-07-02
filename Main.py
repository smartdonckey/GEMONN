

from public import F_distance,F_mating,P_generator_SL_optimization,F_EnvironmentSelect

from Private_function import *

from public import NDsort
import time
import matplotlib.pyplot as plt
import cupy as np
import numpy






def EA_Run(Generations, PopSize, HiddenNum, M, Run, Problem, Algorithm):

    Dim, train_loader, test_loader = LoadData(batch=128)
    Model = Net(Dim, HiddenNum, Dim).cuda()


    Population, Boundary, Coding = Initialization_Pop(PopSize, Dim, HiddenNum)
    FunctionValue, Weight_Grad = Evaluation(Population, Dim, HiddenNum, train_loader,Model,False)

    FrontValue = NDsort.NDSort(FunctionValue,PopSize)[0]  #
    CrowdDistance = F_distance.F_distance(FunctionValue, FrontValue)



    since = time.time()
    plt.ion()

    for Gene in range(Generations):
        MatingPool, index = F_mating.F_mating(Population, FrontValue, CrowdDistance)
        Weight_Grad_Mating = Weight_Grad[index, :]


        Offspring = P_generator_SL_optimization.P_generator_SL(MatingPool, Weight_Grad_Mating, Boundary, Coding, PopSize)
        FunctionValue_Offspring, Weight_Grad_Offspring = Evaluation(Offspring, Dim, HiddenNum, train_loader,Model,False)

        if Gene == 499:
            _, _, Offspring_Next = Evaluation(Population, Dim, HiddenNum, train_loader, Model,True)
            FunctionValue_Offspring_Next, Weight_Grad_Offspring_Next = Evaluation(Offspring_Next, Dim, HiddenNum, train_loader, Model,False)

            Offspring = np.vstack((Offspring,Offspring_Next))
            FunctionValue_Offspring = numpy.vstack((FunctionValue_Offspring,FunctionValue_Offspring_Next))
            Weight_Grad_Offspring = np.vstack((Weight_Grad_Offspring,Weight_Grad_Offspring_Next))




        Population = np.vstack((Population, Offspring))
        FunctionValue = numpy.vstack((FunctionValue, FunctionValue_Offspring))
        Weight_Grad_Temp = np.vstack((Weight_Grad, Weight_Grad_Offspring))

        Population, FunctionValue, Weight_Grad, FrontValue, CrowdDistance = F_EnvironmentSelect.F_EnvironmentSelect(
            Population, Weight_Grad_Temp, FunctionValue, PopSize)





        print(np.sort(FunctionValue[:, 1])[:2])
        # plt.clf()
        # plt.plot(FunctionValue[:, 0], FunctionValue[:, 1], "*")
        # plt.pause(0.001)


        print(Algorithm, "Run :", Gene, "代，Complete：", 100 * Gene / Generations, "%, time consuming:",
              numpy.round(time.time() - since, 2), "s")




    FunctionValueNon = FunctionValue[(FrontValue == 1)[0], :]
    PopulationNon = Population[(FrontValue == 1)[0],:]

    plt.plot(FunctionValue[:, 0], FunctionValue[:, 1], "*")
    plt.ioff()
    plt.show()






if __name__ == "__main__":


    Generations = 10
    PopSize = 50
    HiddenNum = 500 #1300
    M = 2
    Run = 1
    Algorithm = "NSGA-II"
    Problem = "DTLZ1"
    root = 'result_millon_'

    EA_Run(Generations, PopSize, HiddenNum, M, Run, Problem, Algorithm)
