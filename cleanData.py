import os
import sys
from torch.utils.data import DataLoader
from MusicDatasetFolder import MusicDatasetFolder

fmaSmall = MusicDatasetFolder(root_path="smallAllSongs/", sampleSize=44100)
dataloader = DataLoader(fmaSmall, batch_size=1, num_workers=1)
boolean = True

# while (boolean):
#     boolean = False
#     fmaSmall = MusicDatasetFolder(root_path="smallAllSongs/", sampleSize=44100)
#     x = os.listdir('smallAllSongs')
#     try:
#         for i, song in enumerate(x, 0):
#             fmaSmall.__getitem__(i)
#     except:
#         print(sys.exc_info()[0])
#         os.remove('smallAllSongs/' + str(x[i]))
#         boolean = True
#         pass


# for i, batch in enumerate(dataloader):
#     dataMusic = batch[:, None, None, :]
#     print(i)



# x = os.listdir('smallAllSongs')
# for i, song in enumerate(x, 0):
#     try:
#         print(str(x[i]))
#         fmaSmall.__getitem__(i)
#     except:
#         print(sys.exc_info()[0])
#         print(i)
#         songName = str(x[i])
#         print(songName)
#         os.remove('smallAllSongs/' + songName)
#         pass




# folders = os.listdir('smallAllSongs')
# for folder in folders:
#     print(folder)
#     songs = os.listdir('smallAllSongs/' + folder)
#     for song in songs:
#         # print(song)
#         os.rename('smallAllSongs/' + folder + '/' + song, 'smallAllSongs/' + song)
#     os.rmdir('smallAllSongs/' + folder)