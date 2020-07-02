import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision可以帮助我们处理常用数据集，如MNIST，COCO, ImageNET等
import torchvision.datasets as dsets
import torchvision.transforms as transforms


import cupy as np



class Net(nn.Module):
    def __init__(self, InputDim, HiddenNum, OutputDim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(InputDim, HiddenNum)

        self.fc2 = nn.Linear(HiddenNum, OutputDim,bias=False)


    def forward(self, X):
        X = torch.sigmoid(self.fc1(X))
        X1 = X
        self.fc2.weight.data = self.fc1.weight.data.t()
        X = torch.sigmoid(self.fc2(X))
        return X, X1

    def Initialization(self, weights_P):

        weights = weights_P[:,:-1]
        bias = weights_P[:,-1]
        self.fc1.weight.data = weights
        self.fc1.bias.data = bias
        self.fc2.weight.data = weights.t()

    def get_weights(self):
        weights_bias = torch.cat((self.fc1.weight.data.t(), self.fc1.bias.data.reshape(1, -1)))
        return weights_bias



def LoadData(batch = 64):
    # MNIST dataset
    train_dataset = dsets.MNIST(root='Data/',  # 选择数据的根目录
                                train=True,  # 选择训练集
                                transform=transforms.ToTensor(),  # 转换成tensor变量
                                download=True)  # 不从网络上download图片
    test_dataset = dsets.MNIST(root='Data/',  # 选择数据的根目录
                               train=False,  # 选择训练集
                               transform=transforms.ToTensor(),  # 转换成tensor变量
                               download=True)  # 不从网络上download图片
    T_Dim = np.array(train_dataset.train_data.shape)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch,
                                               shuffle=True)  # 将数据打乱
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch,
                                              shuffle=False)  # 将数据打乱
    Dim = T_Dim[1] * T_Dim[2]



    return int(Dim), train_loader, test_loader



def Initialization_Pop(PopSize, Dim, HiddenNum):
    Dim += 1
    Dim = int(Dim)
    # Population = (np.random.random((PopSize, Dim * HiddenNum)) - 0.5) * 2 * ((np.power(6 / (Dim + HiddenNum), 1 / 2)))
    Population = (np.random.random((PopSize, Dim * HiddenNum)) - 0.5) * 2 * ((6/np.power((Dim + HiddenNum), 1 / 2)))
    for i in range(PopSize):
        Population[i] = Population[i]*(np.random.rand( Dim * HiddenNum,) < ((i+1)/PopSize))

    Boundary = np.hstack(( np.tile([[10], [-10]], [1, (Dim-1) * HiddenNum]), np.tile([[20], [-20]], [1,  HiddenNum])))
    # Boundary = np.tile([[20], [-20]], [1, Dim * HiddenNum])
    Coding = 'Real'
    return Population, Boundary, Coding


def Evaluation(Population, Dim, HiddenNum, Data, Model,flag):
    import numpy

    # Here Dim is  28*28  not 28
    pop_size = Population.shape[0]
    Weight_Grad = np.zeros(Population.shape)

    Update_weights = np.empty((0,Population.shape[1]))


    FunctionValue = np.zeros((pop_size, 2))
    # # 计算 sparsity
    FunctionValue[:, 0] = np.sum(Population != 0, axis=1) / ((Dim+1) * HiddenNum)

    FunctionValue = np.asnumpy(FunctionValue)

    if flag:
        Update_step = numpy.zeros((pop_size,))
        Update_step = numpy.int64( ( FunctionValue[:, 0]/numpy.max(FunctionValue[:, 0]) )*1000 )
        Update_step[FunctionValue[:, 0]<0.05] = 20


    # Load Train Data
    data_iter = iter(Data)
    images, labels = data_iter.next()
    images = images.view(-1, Dim).cuda()  # Dim is 28*28 not 28
    labels = labels.cuda()



    # Define Loss funcion , here MSE is adopted
    criterion = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(Model.parameters(), lr=learning_rate)
    # Compute MSE and Grad for each weight set


    for i in range(pop_size):
        weights = torch.Tensor(np.reshape(Population[i, :], (Dim+1, HiddenNum))).t()
        weights.requires_grad_(requires_grad=True)

        Model.Initialization(weights.cuda())
        outputs = Model(images)
        loss = criterion(outputs[0], images)
        FunctionValue[i, 1] = loss.cpu().detach().numpy()



        optimizer.zero_grad()
        loss.backward()
        Weight_Grad_Temp= np.reshape(Model.fc1.weight.grad.t().cpu().numpy(), (Dim * HiddenNum,))
        Weight_Grad_T_ = np.reshape(Model.fc2.weight.grad.cpu().numpy(), (Dim * HiddenNum,))

        Weight_Grad[i,:] = np.hstack((coperate_Weight_Grad(Weight_Grad_Temp,Weight_Grad_T_), Model.fc1.bias.grad.cpu().numpy()))
        optimizer.step()

        if flag:

            avg_loss = []
            avg_weight = weights
            min_loss = 3


            for step in range(Update_step[i]):
                New_weights = Model.get_weights().cpu().t()
                New_weights[weights==0] = 0
                Model.Initialization(New_weights.cuda())
                outputs = Model(images)  # here outputs contain two parts, which are final outputs and Hidden outputs
                loss = criterion(outputs[0], images)

                if loss < min_loss:
                    avg_weight = New_weights
                    min_loss = loss
                # print(i,step,loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            Update_weights = np.vstack((Update_weights, avg_weight.t().reshape((1,-1)) ) )

            # Update_weights = np.vstack((Update_weights,np.reshape(Model.get_weights().cpu().data.numpy(),(1,-1))))

    if flag:
        Update_weights[Population==0] = 0
        return FunctionValue, Weight_Grad, Update_weights
    else:
        return FunctionValue, Weight_Grad



def coperate_Weight_Grad(Weight_Grad_Temp,Weight_Grad_T_):
    Result_Grad = Weight_Grad_Temp + Weight_Grad_T_

    Temp_1 = Weight_Grad_Temp
    Temp_1[Temp_1>0] = 1
    Temp_1[Temp_1 < 0] = -1
    Temp_2 = Weight_Grad_T_
    Temp_2[Temp_2 > 0] = 1
    Temp_2[Temp_2 < 0] = -1
    zeroIndex = (Temp_1 + Temp_2) == 0

    #
    # Temp_2 = Weight_Grad_T_.copy()
    # prob = torch.rand(Weight_Grad_Temp.shape)>0.5
    # Weight_Grad_Temp[prob] = Temp_2[prob]
    # Weight_Grad_Temp[zeroIndex] = 0

    Result_Grad[zeroIndex] = 0

    return Result_Grad






