import torch
import torch.nn as nn
from DeepCNN import DeepNeuralNet
import torchvision.datasets
import torchvision.transforms as transforms
from Samples import Net
import matplotlib.pyplot as plt

device = device = torch.device('cuda')

"""
test on classes
class MyMother(object):

    def __init__(self):
        print("new MyMother")

    def doIt(self, inpp):
        print("method call")
        print(inpp)

    def __call__(self, inpp):
        print("object call")
        print(inpp)

Alex = MyMother()

x = 'fuck you'
Alex.doIt(x)
Alex(x)
"""

"""
autograd try
lr = 0.01


x = torch.FloatTensor([900000])
x.requires_grad = True
print(x)
print(x.requires_grad)
# y = x * x
# out = y.mean()

# print(x.grad)

# out.backward()

# y.backward()
#
# print(x.grad)

for i in range(500):
    y = 5 * x
    loss = (y - 1505).pow(2)
    # x.zero_grad()
    loss.backward()
    print(x)
    with torch.no_grad():
        # print("fuck")
        x -= lr * x.grad
        x.grad.zero_()

print("Value of x: ", x)

"""

"""
playing with maxpool

pool = nn.MaxPool2d(2)

x = torch.cuda.FloatTensor(4, 8, 16).normal_()
print(x)

out = pool(x)

print(out)
"""

"""
playing with my class
"""

dataset = torchvision.datasets.CIFAR10("./Data", train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10("./Datatest", train=False, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# print(trainloader.__iter__()._get_batch())
# print(trainloader.__len__())

# i, data = enumerate(trainloader, 0)
model = Net().to(device)
# model = DeepNeuralNet().to(device)

criterion = nn.CrossEntropyLoss()
learningrate = 0.01
# print
running_loss = 0.0
# epoch=0

for epoch in range(3):

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        # print(labels)
        output = model(inputs.to(device))
        # print(inputs.to(device).shape)
        loss = criterion(output, labels.to(device))
        # print(loss)
        loss.backward()
        with torch.no_grad():
            for param in model.parameters():
                param -= learningrate * param.grad
        model.zero_grad()

        running_loss += loss.item()
        if i % 2000 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        inputs, labels = data
        prediction = model(inputs.to(device))
        _, prediction = torch.max(prediction.data, 1)
        total += labels.size(0)
        correct += (prediction == labels.to(device)).sum().item()

print(correct, total)


# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# print(images)

samples = 10
learningrate = 0.01

x = torch.cuda.FloatTensor(samples, 3, 224, 224).normal_()

# output = model.forward(x)
# print(output.size())
#
# y = x.view(x.size()[0], 150528)
#
# print(y.size())
#
# soma = nn.Softmax(dim=1)
#
# x = torch.cuda.FloatTensor(1, 5).normal_()
# print(x)
# x = soma(x)
# print(x)

# y = torch.cuda.LongTensor(samples, 1000)
# for i in range(samples):
#     y[i][0] = 1.

y = torch.cuda.LongTensor(samples)

# for param in model.parameters():
#     param.requires_grad = True

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
# for i in range(10):
#     output = model(x)
#
#     loss = criterion(output, y)
#     print(loss)
#     loss.backward()
#
#     with torch.no_grad():
#         for param in model.parameters():
#             param -= learningrate * param.grad
#     model.zero_grad()

# output = model(x)
#
# loss = (output - y).pow(2).sum()
# print(loss)
# model.zero_grad()
# loss.backward()
#
# with torch.no_grad():
#     for param in model.parameters():
#         param -= learningrate * param.grad
#
# output = model(x)
#
# loss = (output - y).pow(2).sum()
# print(loss)


"""
playing with size
x = torch.cuda.FloatTensor(10, 512, 7, 7)
y = x.view(10, 25088)

a = torch.cuda.FloatTensor(2, 3, 2, 2).normal_()
b = a.view(2, 12)
print(a)
print(b)
print(a.size()[0])
"""