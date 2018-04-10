import numpy as np
import os
import re
from torch.utils.data import Dataset, DataLoader
import torch


class FaceDataset(Dataset):
    def __init__(self, filenames, path="dataset/"):
        self.datasetPath = path
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        sampleName = self.filenames[index][0:-len(".output")]
        x = np.load(self.datasetPath + sampleName+".input")
        y = np.load(self.datasetPath + sampleName+".output")
        x = np.swapaxes(x, 1, 2)
        x = np.swapaxes(x, 0, 1)
        # if self.transform:
        #     sample = self.transform(sample)

        return torch.from_numpy(x.astype(np.float)), torch.from_numpy(y.astype(np.float))


def GetData(percentage=0.85, path="dataset/"):
    filenames = [f for f in os.listdir(
        path) if re.match(pattern=r'.*\.output', string=f)]
    totalSize = len(filenames)
    partition = int(totalSize*percentage)
    trainData = FaceDataset(path=path, filenames=filenames[:partition])
    validationData = FaceDataset(path=path, filenames=filenames[partition:])
    return trainData, validationData

# def loadData(limit=2):
#     filenames = [f for f in os.listdir(
#         './dataset/') if re.match(pattern=r'.*\.output', string=f)]
#     trainingData = list()
#     # x_train = list()
#     # y_train = list()
#     for index in range(limit):
#         # x_train.append(x)
#         # y_train.append(y)
#         trainingData.append((x, y))
#     # x_train = np.stack(x_train)
#     # y_train = np.stack(y_train)
#     return trainingData
