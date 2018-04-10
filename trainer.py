from torch.autograd import Variable
import torch
import os


def save_model(state, logFile):
    filename = 'checkpoint_' + str(state['epoch'])
    torch.save(state, filename)
    logFile.write('MODEL SAVED as: ' + filename + '\n')
    logFile.flush()


def train(model, trainLoader, validationLoader, criterion, optimizer, opt):
    logFile = open(os.path.join(opt.output, "log.txt"), "w+")
    if opt.gpu:
        model = model.cuda()
    for epoch in range(opt.epochs):
        #         training code
        totalLoss = 0
        count = 0
        for (x, y) in trainLoader:
            x = Variable(x.float(), requires_grad=True)
            y = Variable(y.float(), requires_grad=False)
            if opt.gpu:
                x = x.cuda()
                y = y.cuda()

            if x.data.shape[0] == 1:
                continue
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            #logFile.write("epoch = \t"+str(epoch)+"\t batch \t"+str(count)+"\t loss \t=\t"+str(loss.data[0]) + '\n')
            totalLoss = totalLoss+loss.data[0]
            logFile.flush()
            loss.backward()
            optimizer.step()
            count = count+1
        logFile.write("epoch = " + str(epoch) +
                      "\t training avgloss = " + str(totalLoss/count) + '\n')
        logFile.flush()

#         validation code

        validationLoss = 0
        validationCount = 0
        for (x, y) in validationLoader:
            x = Variable(x.float(), requires_grad=False)
            y = Variable(y.float(), requires_grad=False)
            if opt.gpu:
                x = x.cuda()
                y = y.cuda()
            if x.data.shape[0] == 1:
                continue
            y_pred = model(x)

            loss = criterion(y_pred, y)
            validationLoss = validationLoss+loss.data[0]
            validationCount = validationCount+1
        logFile.write("epoch = " + str(epoch)+"\t validation avgloss = " +
                      str(validationLoss/validationCount) + '\n')
        logFile.flush()
        if epoch % 10 == 9:
            save_model({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
                logFile=logFile)
    return model
