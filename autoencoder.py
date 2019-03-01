import torch
from torch.utils.data import DataLoader
import torchaudio
from Gener import GANGenerator
from encoder import encodertest
from MusicDatasetFolder import MusicDatasetFolder
import torch.nn as nn


device = torch.device('cuda')
generator = GANGenerator().to(device)
encoder = encodertest().to(device)
sampleSize = 392426
bottleneck = 6615
criterion = nn.BCEWithLogitsLoss()
learningrate = 0.1

fmaSmall = MusicDatasetFolder(root_path="smallAllSongs/", sampleSize=sampleSize)  #18
dataloader = DataLoader(fmaSmall, batch_size=20, num_workers=4, shuffle=True)

rand = torch.FloatTensor(1, 1, 392426, 1).normal_().to(device)
tocalc = encoder(rand)
print(tocalc.size())

# sound, sample_rate = torchaudio.load('efm.mp3')
# sound = sound[:,0].to(device)
# print(sound.size())
# output = encoder(sound[None, None, :, None])
# output = output[None, :, None]
# output = output.view(1, 1, 9213, 1)
# print("pra: ", output.size())
# output = generator(output)

# rand = torch.FloatTensor(1, 1, 9213, 1).normal_().to(device)
# sth = generator(rand)
#
# print(sound.size(), output.size(), sth.size())

for i, batch in enumerate(dataloader, 0):
    dataMusic = batch[:, None, :, None].to(device)
    targMusic = batch.to(device)
    # print(dataMusic.size())
    encoder.zero_grad()
    generator.zero_grad()

    encVals = encoder(dataMusic)
    # print(encVals.size())
    repMusi = generator(encVals[:, None, :, None])

    loss = criterion(repMusi, targMusic)
    print(loss)
    loss.backward()

    with torch.no_grad():
        for param in generator.parameters():
            param -= learningrate * param.grad
        for param in encoder.parameters():
            param -= learningrate * param.grad

    # print(encVals.size(), repMusi.size(), dataMusic.size(), batch.size())

