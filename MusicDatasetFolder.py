from torch.utils.data import Dataset
import torch
import torchaudio
import os
from random import randint

class MusicDatasetFolder(Dataset):

    """
    Create a class containing a folder with a music dataset
    the data in the folder shouldn't be manipulated after creation

    Args:
        root_path: string containing the folders with all .mp3 track inside no other files should be in there and
        it should end with a slash (/)
        """
    def __init__(self, root_path, sampleSize):
        self.root_path = root_path
        self.listLs = os.listdir(self.root_path)
        self.device = torch.device('cuda')
        self.sampleSize = sampleSize

    """
    get a special item
    Args:
        item: name of the desired mp3 file, it has to be in the dataset """
    def __getitem__(self, item):
        # print(item)
        sound, sample_rate = torchaudio.load(self.root_path + str(self.listLs[item]))
        sound = sound[0].view(-1)
        sound = self.getSoundInShape(sound)
        return sound

    """
    get the size of the dataset
    """
    def __len__(self):
        return len(os.listdir(self.root_path))

    def getListOfSongs(self):
        return os.listdir(self.root_path)

    def getSoundInShape(self, sound):
        soundSize = sound.size()[0]
        if (soundSize>self.sampleSize):
            start = randint(0, soundSize - self.sampleSize)
            return sound[start:start + self.sampleSize]
        target = torch.zeros(self.sampleSize)
        target[:soundSize] = sound
        return target