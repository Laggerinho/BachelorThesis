import torch

class ParamCalulator(object):

    def __init__(self, randomVectorSize, generator, discriminator):
        self.randomVectorSize = randomVectorSize
        self.generator = generator
        self.discriminator = discriminator
        self.device = torch.device('cuda')

    def getParams(self):
        randVect = torch.FloatTensor(2, 1, self.randomVectorSize, 1).normal_().to(self.device)
        geneOut = self.generator(randVect)
        # print("generator out Size: ", geneOut.size())
        sampleSize = geneOut.size()[1]
        # print("Sample size: ", sampleSize)
        disConv = self.discriminator.getConvOutput(geneOut[:, None, None, :])
        # print("input discriminator: ", geneOut[:, None, None, :].size())
        disFirstFCSize = disConv.size()[1]
        # print("first layer discriminator: ", disFirstFCSize)

        return sampleSize, disFirstFCSize

