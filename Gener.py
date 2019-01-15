import torch
import torch.nn as nn
import torch.nn.functional as F

class GANGenerator(nn.Module):

    def __init__(self):
        super(GANGenerator, self).__init__()
        self.tranConv1 = nn.ConvTranspose2d(1, 4, kernel_size=[32, 1], stride=1)
        self.tranConv2 = nn.ConvTranspose2d(4, 16, kernel_size=[64, 1], stride=1)
        self.tranConv3 = nn.ConvTranspose2d(16, 32, kernel_size=[128, 1], stride=1)
        # self.tranConv4 = nn.ConvTranspose2d(1024, 2510, kernel_size=[4, 1], stride=1)
        # self.tranConv5 = nn.ConvTranspose2d(256, 512, kernel_size=[6, 1], stride=1)
        # self.tranConv6 = nn.ConvTranspose2d(512, 2450, kernel_size=[4, 1], stride=1)
        # self.tranConv7 = nn.ConvTranspose2d(1024, 1, kernel_size=[1, 1], stride=1)

        # print(self.tranConv1.stride[0])
        # print(self.tranConv1.stride[1])
        # print(self.tranConv1.padding[0])
        # print(self.tranConv1.padding[1])
        # print(self.tranConv1.output_padding[0])
        # print(self.tranConv1.output_padding[0])
        # print(self.tranConv1.kernel_size[0])
        # print(self.tranConv1.kernel_size[1])

    """Generate a sample with a single dimensional vector, the vector is passed as the the way it is generated might
    actually change"""
    # TODO: implement method that actually generates multiple samples given a multidimensional input
    def forward(self, input):
        # print(input.size())
        randVect = F.leaky_relu(self.tranConv1(input))
        # print(randVect.size())
        randVect = F.leaky_relu(self.tranConv2(randVect))
        # print(randVect.size())
        randVect = self.tranConv3(randVect)
        # print(randVect.size())
        # randVect = self.tranConv4(randVect)
        # print(randVect.size())
        # randVect = F.leaky_relu(self.tranConv5(randVect))
        # print(randVect.size())
        # randVect = self.tranConv6(randVect)
        # print(randVect.size())
        randVect = randVect.view(randVect.size()[0], -1)
        # print("Generator output Size", randVect.size())

        return randVect
        # randVect = self.tranConv7(randVect)
        # print(randVect.size())
