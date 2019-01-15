import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import time

inputSize = 224
num_classes = 1000
batchSize = 100

# TODO: import the Data here


class DeepNeuralNet(nn.Module):

    def __init__(self):
        self.device = torch.device('cuda')
        self.depth = 3
        self.height = 3
        self.width = 3
        self.kernelsOne = 64
        self.kernelsTwo = 128
        self.kernelsThr = 256
        self.kernelsFou = 512
        self.kernelsFiv = 512
        self.finalSize = 7
        super(DeepNeuralNet, self).__init__()

        self.pool = nn.MaxPool2d(2)

        self.conv1 = nn.Conv2d(self.depth, self.kernelsOne, 3, padding=1).to(self.device)
        self.conv2 = nn.Conv2d(self.kernelsOne, self.kernelsOne, 3, padding=1).to(self.device)

        self.conv3 = nn.Conv2d(self.kernelsOne, self.kernelsTwo, 3, padding=1).to(self.device)
        self.conv4 = nn.Conv2d(self.kernelsTwo, self.kernelsTwo, 3, padding=1).to(self.device)

        self.conv5 = nn.Conv2d(self.kernelsTwo, self.kernelsThr, 3, padding=1).to(self.device)
        self.conv6 = nn.Conv2d(self.kernelsThr, self.kernelsThr, 3, padding=1).to(self.device)
        self.conv7 = nn.Conv2d(self.kernelsThr, self.kernelsThr, 3, padding=1).to(self.device)
        self.conv8 = nn.Conv2d(self.kernelsThr, self.kernelsThr, 3, padding=1).to(self.device)

        self.conv9 = nn.Conv2d(self.kernelsThr, self.kernelsFou, 3, padding=1).to(self.device)
        self.conv10 = nn.Conv2d(self.kernelsFou, self.kernelsFou, 3, padding=1).to(self.device)
        self.conv11 = nn.Conv2d(self.kernelsFou, self.kernelsFou, 3, padding=1).to(self.device)
        self.conv12 = nn.Conv2d(self.kernelsFou, self.kernelsFou, 3, padding=1).to(self.device)

        self.conv13 = nn.Conv2d(self.kernelsFou, self.kernelsFiv, 3, padding=1).to(self.device)
        self.conv14 = nn.Conv2d(self.kernelsFiv, self.kernelsFiv, 3, padding=1).to(self.device)
        self.conv15 = nn.Conv2d(self.kernelsFiv, self.kernelsFiv, 3, padding=1).to(self.device)
        self.conv16 = nn.Conv2d(self.kernelsFiv, self.kernelsFiv, 3, padding=1).to(self.device)

        self.full1 = nn.Linear(32768, 4096).to(self.device)
        self.full2 = nn.Linear(4096, 4096).to(self.device)
        self.full3 = nn.Linear(4096, 10).to(self.device)

        self.sofma = nn.Softmax(dim=1)


    def forward(self, input):
        retVal = F.relu(self.conv2(F.relu(self.conv1(input))))
        # retVal = self.pool(retVal)
        retVal = F.relu(self.conv4(F.relu(self.conv3(retVal))))
        # retVal = self.pool(retVal)
        retVal = F.relu(self.conv8(F.relu(self.conv7(F.relu(self.conv6(F.relu(self.conv5(retVal))))))))
        # retVal = self.pool(retVal)
        retVal = F.relu(self.conv12(F.relu(self.conv11(F.relu(self.conv10(F.relu(self.conv9(retVal))))))))
        retVal = self.pool(retVal)
        retVal = F.relu(self.conv16(F.relu(self.conv15(F.relu(self.conv14(F.relu(self.conv13(retVal))))))))
        retVal = self.pool(retVal)
        retVal = retVal.view(retVal.size()[0], -1)
        retVal = self.full1(retVal)
        retVal = self.full2(retVal)
        retVal = self.full3(retVal)
        # retVal = self.sofma(retVal)
        return retVal


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

