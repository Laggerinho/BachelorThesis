import torch
import torch.nn as nn
import torch.nn.functional as F

class GANLinearDisc(nn.Module):

    def __init__(self):
        super(GANLinearDisc, self).__init__()
        self.full1 = nn.Linear(10000, 1000)
        self.full2 = nn.Linear(1000, 10)
        self.full3 = nn.Linear(10, 2)

    def forward(self, input):
        sample = F.relu(self.full1(input))
        sample = F.relu(self.full2(sample))
        sample = self.full3(sample)
        return sample