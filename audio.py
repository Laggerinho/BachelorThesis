import torchaudio

import os
import numpy
import torch
from MusicDatasetFolder import MusicDatasetFolder
torch.set_printoptions(threshold=numpy.nan)

samplelength = 441000 # 44100 represent one second

sound, sample_rate = torchaudio.load('efm.mp3')
# torchaudio.save('foo_save.mp3', sound, sample_rate) # saves tensor to file

print(sound.shape, sample_rate)

# print(sound[1500000][0])
# print(sound[10:1500, 0])

numberOfSamplesInSong = int(sound.shape[0]/samplelength)


print(numberOfSamplesInSong)
sampleTensor = torch.cuda.FloatTensor(numberOfSamplesInSong, samplelength)
# print(sampleTensor)

for i in range(numberOfSamplesInSong):
    sampleTensor[i, :] = sound[i*samplelength:(i+1)*samplelength, 0]

# print(sampleTensor[0,15601],sampleTensor[1,5651],sampleTensor[5,5651],sampleTensor[6,56498])

# print(os.listdir("fma_small"))
# someList = []

fmaSmall = MusicDatasetFolder(root_path ="fma_small/")

sample_rate = 0
sound = 0

# sound, sample_rate = fmaSmall.__getitem__("000002.mp3")

# print(sample_rate)