import torch
import torch.nn as nn
import torch.nn.functional as F

class GANDiscriminator(nn.Module):

    def __init__(self, firstLayer):
        super(GANDiscriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=[1, 2048], stride=36)
        self.conv2 = nn.Conv2d(2, 8, [1, 512], stride=18)
        self.conv3 = nn.Conv2d(8, 16, [1, 128], stride=9) # 64*128*128 + 128 number of parameters
        # self.conv4 = nn.Conv2d(128, 256, [1, 512], stride=5)
        # self.conv5 = nn.Conv2d(256, 512, [1, 256], stride=3)
        # self.conv6 = nn.Conv2d(512, 1024, [1, 512], stride=2)
        # self.conv7 = nn.Conv2d(1024, , 3)
        self.full1 = nn.Linear(firstLayer, 100)
        self.full2 = nn.Linear(100, 10)
        self.full3 = nn.Linear(10, 2)

    def forward(self, input):
        # print(input)
        # sample = F.leaky_relu(self.conv1(input))
        # print(sample.size())
        # sample = F.leaky_relu(self.conv2(sample))
        # print(sample.size())
        # sample = F.leaky_relu(self.conv3(sample))
        # print(sample.size())
        # sample = F.relu(self.conv4(sample))
        # print(sample.size())
        # sample = F.relu(self.conv5(sample))
        # sample = F.relu(self.conv6(sample))
        # print(sample.size())
        sample = self.getConvOutput(input)
        # print(sample.size())
        sample = F.leaky_relu(self.full1(sample))
        sample = F.leaky_relu(self.full2(sample))
        sample = self.full3(sample)
        return sample

    def getConvOutput(self, input):
        sample = F.leaky_relu(self.conv1(input))
        sample = F.leaky_relu(self.conv2(sample))
        sample = F.leaky_relu(self.conv3(sample))
        # print("shape after conv: ", sample.size())
        return sample.reshape(sample.size(0), -1)
