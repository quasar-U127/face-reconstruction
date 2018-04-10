import torch
from collections import OrderedDict


class Identity(torch.nn.Module):
    def forward(self, x):
        return x


def ConvBlock(numIn, numOut):
    return torch.nn.Sequential(
        torch.nn.BatchNorm2d(numIn),
        torch.nn.ReLU(),
        torch.nn.Conv2d(
            in_channels=numIn,
            out_channels=numOut/2,
            kernel_size=1
        ),
        torch.nn.BatchNorm2d(numOut/2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(
            in_channels=numOut/2,
            out_channels=numOut/2,
            kernel_size=3,
            stride=1,
            padding=1
        ),
        torch.nn.BatchNorm2d(numOut/2),
        torch.nn.ReLU(),
        torch.nn.Conv2d(
            in_channels=numOut/2,
            out_channels=numOut,
            kernel_size=1
        )
    )


def SkipLayer(numIn, numOut):
    if numIn == numOut:
        return Identity()
    else:
        return torch.nn.Conv2d(
            in_channels=numIn,
            out_channels=numOut,
            kernel_size=1
        )


class MiniResidualBlock(torch.nn.Module):
    def __init__(self, numIn, numOut):
        super(MiniResidualBlock, self).__init__()
        self.convBlock = ConvBlock(numIn, numOut)
        self.skipLayer = SkipLayer(numIn, numOut)

    def forward(self, x):
        residue = self.skipLayer(x)
        output = self.convBlock(x)
        output = output+residue
        return output


class ResidualBlock(torch.nn.Module):
    def __init__(self, numChannels, numBlocks):
        super(ResidualBlock, self).__init__()
        blocks = []
        for i in range(numBlocks):
            blocks.append(
                (
                    str(i),
                    MiniResidualBlock(
                        numIn=numChannels,
                        numOut=numChannels
                    )
                )
            )
        self.block = torch.nn.Sequential(OrderedDict(blocks))

    def forward(self, x):
        output = self.block(x)
        return output


class HourGlassBlock(torch.nn.Module):
    def __init__(self, numLevels, numChannels, numBlocks, input_size):
        super(HourGlassBlock, self).__init__()
        self.downScaler = torch.nn.MaxPool2d(kernel_size=2)
        self.preHourGlassResidueBlock = ResidualBlock(
            numChannels=numChannels,
            numBlocks=numBlocks
        )
        if numLevels > 1:
            self.hourGlassBlock = HourGlassBlock(
                numLevels=(numLevels-1),
                numChannels=numChannels,
                numBlocks=numBlocks,
                input_size=(input_size[0]/2, input_size[1]/2)
            )
        else:
            self.hourGlassBlock = Identity()

        self.postHourGlassResidueBlock = ResidualBlock(
            numChannels=numChannels,
            numBlocks=numBlocks
        )
        self.upScaler = torch.nn.Upsample(
            scale_factor=2)

        residueDimensions = (input_size[0] % 2, input_size[1] % 2)
        padding = (0, residueDimensions[0], 0, residueDimensions[1])

        self.sizeCorrector = torch.nn.ZeroPad2d(padding=padding)

        self.skipBlock = ResidualBlock(
            numChannels=numChannels,
            numBlocks=numBlocks
        )

    def forward(self, x):
        residue = x
        residue = self.skipBlock(residue)
        x = self.downScaler(x)
        x = self.preHourGlassResidueBlock(x)
        x = self.hourGlassBlock(x)
        x = self.postHourGlassResidueBlock(x)
        x = self.upScaler(x)
        x = self.sizeCorrector(x)
#         print x.shape
#         print residue.shape
        x = x + residue
        return x


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.part1 = torch.nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3
        )
        self.part2 = torch.nn.BatchNorm2d(64)
        self.part3 = torch.nn.ReLU()
        self.part4 = MiniResidualBlock(
            numIn=64,
            numOut=128
        )
        self.part5 = torch.nn.MaxPool2d(
            kernel_size=2
        )
        self.part6 = MiniResidualBlock(
            numIn=128,
            numOut=128
        )
        self.part7 = MiniResidualBlock(
            numIn=128,
            numOut=256
        )
        self.part8 = HourGlassBlock(
            numLevels=4,
            numChannels=256,
            numBlocks=4,
            input_size=(112, 112)
        )
        self.part9 = HourGlassBlock(
            numLevels=4,
            numChannels=256,
            numBlocks=4,
            input_size=(112, 112)
        )
        self.part10 = torch.nn.Upsample(size=(256, 256), mode="bilinear")

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        x = self.part3(x)
        x = self.part4(x)
        x = self.part5(x)
        x = self.part6(x)
        x = self.part7(x)
        x = self.part8(x)
        x = self.part9(x)
        x = self.part10(x)
        return x
