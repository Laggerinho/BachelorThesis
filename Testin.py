import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# playing with convolutional layers
learningrate = 0.000001
device = torch.device('cuda')
samples = 100
height = 50
width = 50
depth = 3
kernels = 50
out_feat = 5
loss_fn = torch.nn.MSELoss(size_average=False)
dtype = torch.float

y = torch.ones([samples,out_feat], dtype=dtype, device=device)
x = torch.cuda.FloatTensor(samples, depth, height, width).normal_()
# , device=device, dtype=dtype, requires_grad=False
# print(x)
# m = nn.Conv1d(depth, kernels, 3).to(device)
m = nn.Conv2d(depth, kernels, 3, padding=1).to(device)
for param in m.parameters():
    param.requires_grad = True
fc1 = nn.Linear(kernels * height * width, out_feat).to(device)

optimizer = torch.optim.SGD(m.parameters(), lr=1e-4)
# print(m.kernel_size)


output = F.relu(m(x))
output = output.view(-1, num_flat_features(output))
output = fc1(output)
# print(output.size())
# print(y.size())

loss = (output - y).pow(2).sum()
print(loss)

m.zero_grad()
fc1.zero_grad()

loss.backward()

with torch.no_grad():
    for param in m.parameters():
        param -= learningrate * param.grad
    for param in fc1.parameters():
        param -= learningrate * param.grad
        
        
output = F.relu(m(x))
output = output.view(-1, num_flat_features(output))
output = fc1(output)

loss = (output - y).pow(2).sum()
print(loss)

for i in range(500):
    output = F.relu(m(x))
    output = output.view(-1, num_flat_features(output))
    output = fc1(output)
    loss = (output - y).pow(2).sum()
    m.zero_grad()
    fc1.zero_grad()
    loss.backward()
    with torch.no_grad():
        for param in m.parameters():
            param -= learningrate * param.grad
        for param in fc1.parameters():
            param -= learningrate * param.grad

print(loss)
# for param in m.parameters():
#     param -= learningrate * param.grad


"""
print("\n\n\n\n\n")
# print(output)
# TODO: the model should be my own neural network and the optimization should be done immediately. like crazy.
# print(m.weight)
# print('\n fuck')

target = torch.ones(m(x).size()).to(device)
exes = []
exes.append(x)

for i in range(500):
    output = m(x)
    # print(output)
    # target = torch.ones(output.size()).to(device)

    # maybe but just maybe we have to sum it up at the end. shouldn't matter for the differentiation because of the
    # linearity of it
    loss = loss_fn(output, target).sum()
    # print(loss)
    # print("\n" + str(x))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    exes.append(x)

"""
