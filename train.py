import torch
from torch.autograd import Variable
import dataset
import numpy as np
import torch.nn.functional as F
import trainer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import Model
import argparse


if (__name__ == "__main__"):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--model", type=str,
                           help="model weights file ")
    argParser.add_argument("--input", type=str,
                           help=" directory with data ", required=True)
    argParser.add_argument("--gpu", action="store_true",
                           help="to use gpu or not")
    argParser.add_argument("--output", type=str, required=True,
                           help="directory to save checkpoints")
    argParser.add_argument("--epochs", type=int,
                           required=True, help="number of epochs")
    argParser.add_argument("--batch", type=int,
                           required=True, help="size of batch")

    opt = argParser.parse_args()

    trainingData, validationData = dataset.GetData(path=opt.input)
    trainLoader = DataLoader(
        trainingData,
        batch_size=opt.batch,
        shuffle=True,
        num_workers=1
    )
    validationLoader = DataLoader(
        validationData,
        batch_size=opt.batch,
        shuffle=True,
        num_workers=1
    )

    model = Model.Net()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    trainedModel = trainer.train(
        model=model,
        trainLoader=trainLoader,
        validationLoader=validationLoader,
        criterion=criterion,
        optimizer=optimizer,
        opt=opt
    )
