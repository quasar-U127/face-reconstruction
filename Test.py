import argparse
import utils
import Model
import re
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable


def test(arguments):
    model = Model.Net()
    model = utils.load_model(
        model=model, weights_file_name=arguments.model)
    utils.preprocess_input(input_directory=arguments.input)

    if(arguments.gpu):
        model = model.cuda()

    inputs = [f for f in os.listdir(
        arguments.input) if re.match(pattern=r'.*\.input', string=f)]
    print "\n\n"
    for x in inputs:
        X = np.load(os.path.join(arguments.input, x))
        X = np.swapaxes(X, 1, 2)
        X = np.swapaxes(X, 0, 1)
        X = torch.from_numpy(X.astype(np.float))
        X = Variable(X.unsqueeze(0).float(), requires_grad=False)
        if(arguments.gpu):
            X = X.cuda()
        output = model(X)
        if(arguments.gpu):
            output = output.cpu()
        output = output.data.numpy()[0, :, :, :]
        output = output > 0.5
        outputFileName = os.path.join(arguments.input,
                                      x[0:-len(".input")]+".output")
        np.save(outputFileName, output)
        print "processed " + x + " to " + x[0:-len(".input")]+".output"
    print "done"


if (__name__ == "__main__"):
    argParser = argparse.ArgumentParser()
    argParser.add_argument("--model", type=str,
                           help="model weights file ", required=True)
    argParser.add_argument("--input", type=str,
                           help=" directory with images ", required=True)
    argParser.add_argument("--gpu", action="store_true",
                           help="to use gpu or not")
    args = argParser.parse_args()
    test(arguments=args)
