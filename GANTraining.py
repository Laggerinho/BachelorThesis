from torch.utils.data import DataLoader
from Gener import GANGenerator
from Discrim import GANDiscriminator
from MusicDatasetFolder import MusicDatasetFolder
from calculateParams import ParamCalulator
from LinearModelGAN import GANLinearDisc
import torch
import torch.nn as nn
import torchaudio

device = torch.device('cuda')
batchSize = 10
cpu = torch.device('cpu')
sigmo = nn.Sigmoid()
logistic = nn.LogSigmoid
criterion = nn.BCELoss()
print('initialize')
randVectSize = 8192
generator = GANGenerator().to(device)
discriminatorTest = GANDiscriminator(5).to(device)
calculator = ParamCalulator(randomVectorSize=randVectSize, generator=generator, discriminator=discriminatorTest)
sampleSize, disFirstFCSize = calculator.getParams()
print("Sample size in Seconds: ", sampleSize/44100.)
discriminator = GANDiscriminator(disFirstFCSize).to(device)
learningrate = 0.01
fmaSmall = MusicDatasetFolder(root_path="smallAllSongs/", sampleSize=sampleSize)  #18
dataloader = DataLoader(fmaSmall, batch_size=batchSize, num_workers=4)
targetReal = torch.zeros(batchSize, 2).to(device)
targetReal[:, 0] = 1.
targetFake = torch.zeros(batchSize, 2).to(device)
targetFake[:, 1] = 1.

print("parameters gen: ", sum(p.numel() for p in generator.parameters()))
print("parameters dis: ", sum(p.numel() for p in discriminator.parameters()))

# discriminator = GANLinearDisc().to(device)
for param in discriminator.parameters():
    param.requires_grad = True
for param in generator.parameters():
    param.requires_grad = True
# randVect = torch.FloatTensor(1, 1, 512, 1).normal_().to(device)
# print(randVect)
randomSample1 = torch.FloatTensor(1, 1, 1, 1323000).normal_().to(device)
randomSample2 = torch.FloatTensor(1, 1, 1, 1323000).normal_().to(device)


print('epochs')

for epoch in range(20):
    print("Epoch: ", epoch)
    for i, batch in enumerate(dataloader, 0):
        dataMusic = batch[:, None, None, :].to(device)
        # print("dataMusic Size: ", dataMusic.size())
        if(dataMusic.size()[0] != batchSize):
            break
        dataOut = discriminator(dataMusic)
        # dataOut = discriminator(randomSample1)
        print("Real, cat 1-0: ", dataOut)
        dataOut = sigmo(dataOut)
        # lossDisReal = (targetReal - dataOut).pow(2).sum()
        lossDisReal = criterion(dataOut, targetReal)
        # print("loss dis real: ", lossDisReal)

        discriminator.zero_grad()
        lossDisReal.backward()

        with torch.no_grad():
            for param in discriminator.parameters():
                param -= learningrate * param.grad

        randVect = torch.FloatTensor(batchSize, 1, randVectSize, 1).normal_().to(device)

        geneMusic = generator(randVect)
        geneOut = discriminator(geneMusic[:, None, None, :])
        # geneOut = discriminator(randomSample2)
        print("Fake, cat 0-1: ", geneOut)
        geneOut = sigmo(geneOut)
        # lossDisFake = (targetFake - geneOut).pow(2).sum()
        lossDisFake = criterion(geneOut, targetFake)
        # print("loss dis fake: ", lossDisFake)

        discriminator.zero_grad()
        lossDisFake.backward()

        with torch.no_grad():
            for param in discriminator.parameters():
                # print(param.grad.sum())
                param -= learningrate * param.grad

        geneMusic = generator(randVect)
        geneOut = discriminator(geneMusic[:, None, None, :])
        geneOut = sigmo(geneOut)
        # lossGenFake = (targetReal - geneOut).pow(2).sum()
        lossGenFake = criterion(geneOut, targetReal)
        # print("loss gen fake: ", lossGenFake)

        generator.zero_grad()
        lossGenFake.backward()

        with torch.no_grad():
            for param in generator.parameters():
                param -= learningrate * param.grad

        # if (epoch != 0):
        #     geneMusic = generator(randVect)
        #     geneOut = discriminator(geneMusic[:, None, None, :])
        #     geneOut = sigmo(geneOut)
        #     # lossGenFake = (targetReal - geneOut).pow(2).sum()
        #     lossGenFake = criterion(geneOut, targetReal)
        #     # print("loss gen fake: ", lossGenFake)
        #
        #     generator.zero_grad()
        #     lossGenFake.backward()
        #
        #     with torch.no_grad():
        #         for param in generator.parameters():
        #             param -= learningrate * param.grad

        # generator.zero_grad()
        # discriminator.zero_grad()



    # print(batch.size())
    # print(discriminator.discriminatorForward(batch))

# generator.generateSample(randVect)
# sound = generator.generateSample(randVect)
# tensor = sound[None, None, None, :]
# print(discriminator.discriminatorForward(tensor))
# tensor = fmaSmall.__getitem__(1)[None, None, None, :].to(device)
# print(discriminator.discriminatorForward(tensor))
# print(batch)
for i, vector in enumerate(geneMusic):
    print(vector)
    torchaudio.save('foo_save' + str(i) + '.mp3', vector.to(cpu), 44100) # saves tensor to file