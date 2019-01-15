import matplotlib
import audio
import torch
import torch.nn as nn
from DeepCNN import Net
import torchvision.datasets
import torchvision.transforms as transforms
matplotlib.use("agg")
import matplotlib.pyplot as plt

device = torch.device('cuda')

# plt.plot([1,2,3,4])
# plt.ylabel('some numbers')
# plt.savefig('foo.png')

dataset = torchvision.datasets.CIFAR10("./Data", train=True, download=True, transform=transforms.ToTensor())
testset = torchvision.datasets.CIFAR10("./Datatest", train=False, download=True, transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

# model = Net().to(device)
criterion = nn.CrossEntropyLoss()
learningrate = 0.01
running_loss = 0.0
plot_values = []
total = 0.0
correct = 0.0
acc_values = []

for repe in range(15):
    total = 0
    correct = 0
    model = Net().to(device)
    running_loss = 0.0
    for epoch in range(repe+1):
        running_loss = 0.0
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
            if i % 1000 == 999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                plot_values.append(running_loss)
                running_loss = 0.0
    plt.plot(plot_values)
    plt.ylabel('loss')
    plt.savefig('loss' + str(epoch + 1) + '.png')
    plot_values = []
    plt.close()

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            prediction = model(inputs.to(device))
            _, prediction = torch.max(prediction.data, 1)
            total += labels.size(0)
            correct += (prediction == labels.to(device)).sum().item()
        acc_values.append((correct/total)*100)

plt.plot(acc_values, 'ro')
plt.ylabel('accuracy')
plt.xlabel('epochs')
plt.savefig('Network accuracy')

# print(plot_values)

# plt.plot(plot_values)
# plt.ylabel('loss')
# plt.savefig('foo.png')