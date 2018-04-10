import torch
import numpy as np
import os
import re
import matplotlib.pyplot as plt


def load_model(model, weights_file_name):
    checkpoint = torch.load(weights_file_name)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def preprocess_input(input_directory):
    filenames = [f for f in os.listdir(
        input_directory) if re.match(pattern=r'.*\.jpg', string=f)]

    for filename in filenames:
        data = plt.imread(os.path.join(input_directory, filename))
        data = data.astype(np.float)
        data = data/256
        outputFile = open(name=os.path.join(
            input_directory, filename[0:-len(".jpg")]+".input"), mode="w+")
        np.save(file=outputFile, arr=data)
        print "converted " + \
            filename+" to " + \
            filename[0:-len(".jpg")]+".input"
