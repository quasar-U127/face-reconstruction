import torch
from torch.autograd import Variable
import dataset
import numpy as np
import torch.nn.functional as F
import trainer
from torch.utils.data import Dataset, DataLoader
import numpy as np

def IOUList(network,dataLoader,gpu=True):
    if gpu:
        network = network.cuda()
    iouList = [];
    iouList = torch.ones(len(dataLoader.dataset))
    for i,(x, y) in enumerate(dataLoader):
        x = Variable(x.float(), requires_grad=False)
        y = Variable(y.float(), requires_grad=False)
        if gpu:
            x = x.cuda()
            y = y.cuda()

        y_pred = network(x)
#         pred_y = torch.ge(y_pred.data,0.125)
        y = y.cpu()
        y_pred = y_pred.cpu()
        pred_numpy = y_pred.data.numpy() > 0.125
        y_numpy = y.data.numpy() > 0
        union = np.logical_or(pred_numpy,y_numpy)
        intersection = np.logical_and(pred_numpy,y_numpy)
        union = np.sum(np.sum(np.sum(union, axis=1), axis=1), axis=1)
        union = union.astype(np.float)
        intersection = np.sum(np.sum(np.sum(intersection, axis=1), axis=1), axis=1)
        iou= intersection/union
        iou = iou.tolist();
        iouList = iouList+iou
        print i,iou
    return iouList