import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision可以帮助我们处理常用数据集，如MNIST，COCO, ImageNET等
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import scipy.fftpack as sci
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


class Net(nn.Module):
    def __init__(self, InputDim, HiddenNum, OutputDim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(InputDim, HiddenNum)
        # self.fc1.weight.requires_grad_(requires_grad=False)
        self.fc2 = nn.Linear(HiddenNum, OutputDim,bias=False)
        # self.fc2.weight.requires_grad_(requires_grad=False)

    def forward(self, X):
        X = torch.sigmoid(self.fc1(X))
        X1 = X
        self.fc2.weight.data = self.fc1.weight.data.t()
        X = torch.sigmoid(self.fc2(X))
        return X, X1

    def Initialization(self, weights_P):
        # self.fc1.weight.data = weights.data # weights 利用Tensor 创立 其 require_grads =  False  ,不能直接赋给  fc1.weights
        # self.fc2.weight.data = weights.t().data
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
                                download=False)  # 不从网络上download图片
    test_dataset = dsets.MNIST(root='Data/',  # 选择数据的根目录
                               train=False,  # 选择训练集
                               transform=transforms.ToTensor(),  # 转换成tensor变量
                               download=False)  # 不从网络上download图片
    T_Dim = np.array(train_dataset.train_data.shape)


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch,
                                               shuffle=True)  # 将数据打乱
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch,
                                              shuffle=True)  # 将数据打乱
    Dim = T_Dim[1] * T_Dim[2]

    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()

    return Dim, train_loader, test_loader


class autoencoder_softmax(nn.Module):
    def __init__(self, weights, classNum):
        super(autoencoder_softmax, self).__init__()
        self.layer_num = len(weights)
        self.feature_encoder = nn.Sequential()
        # Construct n layers autoencoder
        for i in range(self.layer_num):
            self.feature_encoder.add_module("layer" + str(i), nn.Linear(weights[i].shape[0] - 1, weights[i].shape[1]))
            self.feature_encoder[2 * i].weight.data = weights[i][:-1].t()
            self.feature_encoder[2 * i].bias.data = weights[i][-1]
            # self.feature_encoder.add_module("layer" + str(i) + "ActiveFunction", nn.ReLU())
            self.feature_encoder.add_module("layer" + str(i) + "ActiveFunction", nn.Sigmoid())
        # Construct Softmax classification layer
        # self.Softmax_layer =  nn.Sequential(nn.Linear(weights[i].shape[1],classNum), nn.Softmax(dim=1))
        self.Softmax_layer = nn.Sequential(nn.Linear(weights[i].shape[1], classNum), nn.LogSoftmax(dim=1))

    def forward(self, x):
        feature = self.feature_encoder(x)
        out = self.Softmax_layer(feature)
        return out, feature


def train_layer_wise_autoencoder(Dim, HiddenNum, trainloader, Model_trans, Gene):
    import time
    since = time.time()
    Model = Net(Dim, HiddenNum, Dim).cuda()

    count = 0
    Flag = False

    Loss = nn.MSELoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)
    # optimizer = torch.optim.SGD(Model.parameters(), lr=0.001)#, momentum=0.5

    save_loss = []

    for epoch in range(Gene):
        running_loss = 0.0
        if Flag == True:
            break
        for i, data in enumerate(trainloader, 0):
            if count>=25000:
                Flag = True
                break

            # get the inputs
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1).cuda()
            if Model_trans is not None:
                _, inputs = Model_trans(inputs)
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, _ = Model(inputs)
            loss = Loss(outputs, inputs)
            save_loss.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            count += 1

            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f, time:%.3f' %
                      (epoch + 1, i + 1, running_loss / 100,time.time()-since))
                running_loss = 0.0


    print('Finished Training')

    return Model.get_weights()


def train_autoencoder_n_layer_softmax(weights):  # train_data, train_label, test_data, test_label,weights

    Dim, trainloader, testloader = LoadData(batch=128)



    # Population = np.loadtxt('result\PopulationNon.txt')
    # FunctionValue = np.loadtxt('result\FunctionValueNon.txt')
    # selectindex = np.argmin(FunctionValue[:, 0])
    # Population_select = Population[selectindex]

    # Population_select = np.loadtxt('W.txt')
    # weights_1 = torch.Tensor(np.reshape(Population_select, (Dim + 1, -1)))
    # weights.append(weights_1)




    # Model = None
    # weights = []
    # layer_1_weights = train_layer_wise_autoencoder(Dim, 500, trainloader, Model, Gene=20)
    # weights.append(layer_1_weights.cpu().data)

    # Model = autoencoder_softmax(weights, 10).cuda()
    #
    # layer_2_weights = train_layer_wise_autoencoder(200, 200, trainloader,Model,Gene=10)
    #
    # weights.append(layer_2_weights.cpu().data)

    # del Model
    #
    # Model = autoencoder_softmax(weights, 10).cuda()
    #
    # layer_3_weights = train_layer_wise_autoencoder(320, 240, trainloader, Model, Gene=10)
    #
    # weights.append(layer_3_weights.cpu().data)
    #
    # del Model
    #
    # Model = autoencoder_softmax(weights, 10).cuda()
    #
    # layer_4_weights = train_layer_wise_autoencoder(240, 120, trainloader, Model, Gene=10)
    #
    # weights.append(layer_4_weights.cpu().data)
    #
    # del Model

    # weights.append( torch.Tensor(np.random.rand(Dim+1,200) ).cuda() )#*np.random.randint(0,2,(Dim,100))
    # weights.append(torch.Tensor(np.random.rand(200+1, 200)).cuda())
    # weights.append(torch.Tensor(np.random.rand(320+1, 240)).cuda())
    # weights.append(torch.Tensor(np.random.rand(240+1, 120)).cuda())

    Model = autoencoder_softmax(weights, 10).cuda()
    # Loss = nn.CrossEntropyLoss()  #nn.CrossEntropyLoss
    ## 多分类用的交叉熵损失函数，用这个 loss 前面不需要加 Softmax 层。
    Loss = nn.NLLLoss()
    # optimizer = torch.optim.Adam(Model.parameters(),lr=0.01)

    optimizer = torch.optim.SGD(Model.parameters(), lr=0.1, momentum=0.9)
    for epoch in range(20):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1).cuda()
            labels = labels.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs, _ = Model(inputs)
            # loss = F.nll_loss(outputs, labels)
            loss = Loss(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1).cuda()

            outputs, _ = Model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.cpu().data == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %.5f %%' % (
            100 * correct / total))

    return 100 * correct / total
# # #
if __name__ =='__main__':
    Dim, trainloader, testloader = LoadData(batch=128)

    weights = []


    Model = None
    layer_1_weights = train_layer_wise_autoencoder(Dim, 500, trainloader, Model, Gene=100)
    weights.append(layer_1_weights.cpu().data)

    #
    # acc = []
    # for i in range(30):
    #     acc.append(train_autoencoder_n_layer_softmax(weights))
    #
    # np.savetxt('result_comparison/acc_gsbx_new_knee_sigmoid.txt', np.array(acc), delimiter=' ')
    # print("hello world!!")
