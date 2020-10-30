

from public import F_distance,F_mating,P_generator_SL_optimization,F_EnvironmentSelect

from Private_function import *

from public import NDsort
import time
import matplotlib.pyplot as plt
import cupy as np
import numpy,argparse






def EA_Run(Generations, PopSize, HiddenNum, Algorithm, args):

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



        if Gene == Generations+1:
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





        if args.plot:
            plt.clf()
            plt.plot(FunctionValue[:, 0], FunctionValue[:, 1], "*")
            plt.pause(0.001)


        print('\r',Algorithm, "Run :", Gene, "代，Complete：", 100 * Gene / (Generations-1), "%, time consuming:",
              numpy.round(time.time() - since, 2), "s, minimal loss:", np.sort(FunctionValue[:, 1])[:1], end='')




    FunctionValueNon = FunctionValue[(FrontValue == 1)[0], :]
    PopulationNon = Population[(FrontValue == 1)[0],:]
    if args.save:
        numpy.savetxt(args.save_dir+'/FunctionValueNon.txt', FunctionValueNon,delimiter=' ')
        numpy.savetxt(args.save_dir+'/PopulationNon.txt', np.asnumpy(PopulationNon),delimiter=' ')

        plt.plot(FunctionValue[:, 0], FunctionValue[:, 1], "*")
        plt.savefig(args.save_dir + '/plot_'+str(Generations-1)+'.png')

    plt.plot(FunctionValue[:, 0], FunctionValue[:, 1], "*")
    plt.savefig(args.save_dir + '/plot_' + str(Generations) + '.png')
    plt.ioff()
    plt.show()







if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='GEMONN setting')

    parser.add_argument('--Generations',type=int,default=500,help='The maximal iteration of the algorithm')
    parser.add_argument('--Popsize',type=int,default=50,help='The population size')

    parser.add_argument('--HiddenNum',type=int,default=500,help='The number of hidden units of an auto-encoder')
    parser.add_argument('--Algorithm',type=str,default="NSGA-II",help='The used framwork of Evoluitonary Algorithms ')

    parser.add_argument('--plot', action='store_true', default=True, help='Plot the function value each generation')
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='./result')



    args = parser.parse_args()
    create_dir(args.save_dir)



    EA_Run(args.Generations, args.Popsize, args.HiddenNum, args.Algorithm,args)
