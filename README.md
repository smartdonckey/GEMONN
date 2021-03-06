# GEMONN

## Introduction
This the source code of GEMONN *A Gradient Guided Evolutionary Approach to Training Deep Neural Networks*

## Requirements and  Dependency
```bash
torch>=1.4.0
torchvision >=0.5.1
cuda 10.0
cupy
numpy
matplotlib
```
## Usage
```bash
python Main.py
python Main.py --Generations 500 --Popsize 50 --HiddenNum 500 --plot  --save --save_dir ./result
```
## Result 
 #### This is the results of Auto-Encoder (AE) with sinlge hidden layer on MNIST results: the x-axis is MSE loss, and the y-axis is the sparsity in term of L0-norm


<img src="./result/plot_initialization.png" width = "500" align=center/>
(Initial result)


<img src="./result/plot_499.png" width = "500" align=center/>
(Result after evolution)

<img src="./result/plot_500.png" width = "500" align=center/>
(Final results)

 #### Next, the comparison of  final solutions obtained by GEMONN, SGD, Adam, and NSGA-II vatiants on MNIST is shown in the MSE-Sparsity axis as follows:

<img src="./result/MSE_Pareto.png" width = "500" align=center/>

#### Then, the convergence spped of GEMONN and these compared approaches in terms of the decline in MSE are presented (MNIST):

<img src="./result/MSE_descent.png" width = "500" align=center/>

#### The following figure shows the test CCR values (accuracy) of the models on ten datasets, which tail a classifier  to the hidden layer of the AE:
<img src="./result/acc_comparison.png" width = "500" align=center/>

 #### In the following,  the comparison of final solution in terms of sparsity is given:
 ![sparsity_comparison](result/sparsity_comparison.png)

 #### The sparsity obtained by GEMONN in different generation(1st, 10th, 100th, and 500th):

 <div align="center">
<img src="./result/1st_weights.png" height="500" alt="1st generation" width="500" >

<img src="./result/10th_weights.png" height="500" alt="10th generationn" width="500" >

 </div>

  (1st generation)                                      (10th generation)


 <div align="center">
<img src="./result/100th_weights.png" height="500" width="500" >

<img src="./result/500th_weights.png" height="500" width="500" >

 </div>
 (100th generation)                                     (500th generation)



#### The reconstruction image of the optimal solution obtained by GEMONN in different generation(1st, 10th, 50th, 100th, 200th, 300th, 500th, orinal_image):

 

 
 <div align="center">
<img src="./result/restruction_0.svg" height="200" width="200" >

<img src="./result/restruction_10.svg" height="250" width="250" >

<img src="./result/restruction_50.svg" height="250" width="250" >

<img src="./result/restruction_100.svg" height="250" width="250" >

 </div>
   (1st generation)                                      (10th generation) (50th generation) (100th generation)
 
 <div align="center">
<img src="./result/restruction_200.svg" height="237" width="237" >

<img src="./result/restruction_300.svg" height="237" width="237" >

<img src="./result/restruction_499.svg" height="237" width="237" >

<img src="./result/original_image.svg" height="237" width="237" >

 </div>
    (200th generation)                                      (300th generation) (500th generation) (origianl image)


## Extension for More Models
For training more models such as LSTM and CNNs, there are guidance  in  `Private_function.py` :

1) Get your model in pytorch
```bash
Model = LeNet()
```
2) Get the weights dictionary of the model :
```bash
 Parameter_dict = Model.state_dict()
```
3) Initialize the population and obtain corresponding size and length inforamtion of weighs in different parts of the model :
```bash
 Population, Boundary, Coding, SizeInform, LengthInform = Initialization_Pop_(PopSize =10, Model = Model)
```
4) Obtain the weights dictionary of each individual in population and compute the inference loss for evaluation:
```bash
 Parameter_dict_i = Pop2weights(Population[0], SizeInform, LengthInform, Parameter_dict)
 Model.load_state_dict(Parameter_dict_i)
```
5) Train this model by GEMONN supported by sparse-SGD or spare-Adam



