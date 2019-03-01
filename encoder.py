import torch
import torch.nn as nn
import torch.nn.functional as F

class encodertest(nn.Module):

    def __init__(self):
        super(encodertest, self).__init__()
        # self.tranConv1 = nn.ConvTranspose2d(1, 4, kernel_size=[256, 1], stride=1)
        # self.tranConv2 = nn.ConvTranspose2d(4, 16, kernel_size=[512, 1], stride=1)
        # self.tranConv3 = nn.ConvTranspose2d(16, 32, kernel_size=[1024, 1], stride=1)
        # self.tranConv4 = nn.ConvTranspose2d(32, 46, kernel_size=[128, 1], stride=1)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=[32768, 1], stride=3)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=[16384, 1], stride=2)
        self.conv3 = nn.Conv2d(8, 4, kernel_size=[8192, 1], stride=2)
        self.conv4 = nn.Conv2d(4, 1, kernel_size=[4096, 1], stride=2)
        self.fc = nn.Linear(8843, 6615)

    def forward(self, input):

        input = F.leaky_relu(self.conv1(input))
        # print(input.size())
        input = F.leaky_relu(self.conv2(input))
        # print(input.size())
        input = F.leaky_relu(self.conv3(input))
        # print(input.size())
        input = F.leaky_relu(self.conv4(input))
        # print(input.size())
        input = input.view(input.size()[0], -1)
        input = F.leaky_relu(self.fc(input))

        return input
