import gc
import sys
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
torch.backends.cudnn.benchmark = True

import dataset
from models.AlexNet import *
from models.ResNet import *
from tqdm import tqdm
import platform


def run():
    # Parameters
    num_epochs = 500
    output_period = 100
    batch_size = 32

    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = resnet_34()
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)
    num_train_batches = len(train_loader)

    criterion = nn.CrossEntropyLoss().to(device)
    # optimizer is currently unoptimized
    # there's a lot of room for improvement/different optimizers
    # optimizer = optim.SGD(model.parameters(), lr=1e-3, nesterov=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.Nesterov(model.parameters(), lr=1e-3)

    best_val_accuracy = 0

    epoch = 1
    while epoch <= num_epochs:
        running_loss = 0.0
        for param_group in optimizer.param_groups:
            print('Current learning rate: ' + str(param_group['lr']))
        model.train()

        for batch_num, (inputs, labels) in enumerate(tqdm(train_loader), 1):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # print(labels)
            # print(labels.size())

            optimizer.zero_grad()
            outputs = model(inputs)
            # print(outputs.size())
            # print(torch.topk(outputs,5))
            # top5 = torch.topk(outputs,5)[1]
            # top52 = torch.topk(outputs,5)[0]
            # for i in range(len(inputs)):
            #     print("\n\nyooooooooo", i, "\n\n\n", labels[i].item(),"\n", top5[i], top52[i])
            #     print(top5[i][0])
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()
            running_loss += loss.item()

            if batch_num % output_period == 0:
                print('[%d:%.2f] loss: %.3f' %
                      (epoch, batch_num * 1.0 / num_train_batches,
                       running_loss / output_period))
                running_loss = 0.0
                gc.collect()

        gc.collect()

        model.eval()
        # model.
        # TODO: Calculate classification error and Top-5 Error
        # on training and validation datasets here
        printAndGetAccuracy(train_loader, device, model, "TRAINSET", epoch)
        _, val5 =printAndGetAccuracy(val_loader, device, model, "VALSET", epoch)

        if val5 > best_val_accuracy:
            best_val_accuracy = val5
            # save after every epoch that the validation accuracy improved
            model.train() #not sure if this matters lol
            torch.save(model.state_dict(), "models/model.%d" % epoch)


        gc.collect()
        epoch += 1

def printAndGetAccuracy(loader, device, model, name, epoch, max_iters=10000):
    num1 = 0
    num5 = 0
    total = 0
    for (inputs, labels) in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        top5 = torch.topk(outputs, 5)[1]
        # top1 = torch.topk(outputs,1)[1]
        for i in range(len(inputs)):
            # print("\nlabel:", labels[i].item(),"\nTop 5:", top5[i])
            top1 = top5[i][0]
            num1 += 1 if labels[i].item() == top1.item() else 0
            num5 += 1 if labels[i].item() in top5[i] else 0
            total += 1
        if total >= max_iters:
            break
    print(epoch, name, "TOP 1:", str(num1/total))
    print(epoch, name, "TOP 5:", str(num5/total))
    return (num1/total, num5/total)


if __name__ == '__main__' and platform.system() == "Windows":
    print('Starting training')
    run()
    print('Training terminated')
elif platform.system() != "Windows":
    print('Starting training NOTWINDOWS')
    run()
    print('Training terminated')