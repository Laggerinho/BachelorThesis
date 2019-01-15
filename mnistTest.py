import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import time
import os
import csv
import pandas as pd
import collections

"""
# Device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

# Hyper-parameters
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.01

# MNIST dataset
train_dataset = torchvision.datasets.MNIST(root='../../data',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root='../../data',
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

start = time.time()
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass
        # object model is callable through method __call__ check implementation of nn.Module for further information
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

end = time.time()

print("training time: " + str(end - start))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)
        # object model is callable through method __call__ check implementation of nn.Module for further information
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')

"""

"""Working on data

for folder in os.listdir("fma_small"):
    for song in os.listdir("fma_small/" + folder):
        os.rename("fma_small/" + folder + "/" + song, "fma_small/" + song,)
        print(song)
    print(folder)
    os.rmdir("fma_small/" + folder)
"""

"""csv files, creating a new one only containing the track id and the genre id"""
"""


def getGenreId(row):
    if isinstance(row, collections.Iterable):
        retVal = row
        apostroph = [pos for pos, char in enumerate(row) if char == '\'']
        return retVal[apostroph[2]+1:apostroph[3]]
    return '-1'


fields = ['track_id', 'track_genres']

df = pd.read_csv('raw_tracks.csv', skipinitialspace=True, usecols=fields)
# df.dropna()

# print(df['track_genres'])
previous = -1
for i, row in enumerate(df['track_genres']):
    # print(df.loc[i, 'track_genres'])
    df.loc[i, 'track_genres'] = getGenreId(row)
    # print(df.loc[i, 'track_genres'])
    # row.replace(row, getGenreId(row))
    # print(i)
    if(i>110000 or i < previous):
        print("something went terribly wrong")
        break
    previous = i

df.to_csv('out.csv')

print(df)

"""

"""
Moving the tracks into the corresponding folders that should be created if they don't exist
"""

"""
def getSongNumber(song):
    while(song[0] == '0'):
        song = song[1:]
    return song[:-4]

df = pd.read_csv('out.csv', skipinitialspace=True)
df.set_index('track_id', inplace=True)
# df.head()
# fields = ['track_id', 'track_genres']
# print(os.path.isdir('fma_small'))


# print(os.listdir('fma_small'))
for song in os.listdir('fma_small'):
    songNumber = int(getSongNumber(song))
    songGenre = df.loc[songNumber, "track_genres"]
    if os.path.isdir('fma_small/' + str(songGenre)):
        # print("true")
        os.rename("fma_small/" + song, "fma_small/" + str(songGenre) + "/" + song)
    else:
        os.makedirs('fma_small/' + str(songGenre))
        os.rename("fma_small/" + song, "fma_small/" + str(songGenre) + "/" + song)
        # print("false")

"""
"""
Cutting the tracks into pieces
"""