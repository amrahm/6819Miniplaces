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
from models.SENet import *
from tqdm import tqdm
import platform

# Parameters
number_of_epochs = 500
output_period = 100
size_of_batch = 10
# model_to_use = se_resnet50()
# model_to_use = se_resnext50_32x4d()
model_to_use = resnet_18()


def run(num_epochs, out_period, batch_size, model):
    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader, val_loader = dataset.get_data_loaders(batch_size)

    if len(sys.argv) > 1:  #output file for val set
        epoch = sys.argv[1]  #take number
        print("loading models/model.%s" % epoch)
        model.load_state_dict(torch.load("models/model.%s" % epoch))
        model.eval()

        # Opens file to write results to, will overwrite existing files
        out_file = open("resultsVAL.txt", "w")
        total = 0
        for (inputs, labels) in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            top5 = torch.topk(outputs, 5)[1]
            # path = "test/" + '{0:08d}'.format(i) + ".jpg"
            for i in range(len(inputs)):
                filename = val_loader.dataset.samples[total][0]
                # formats string in the structure of "val/39/00000132.jpg 1 3 5 6 9"
                path_top5 = filename

                for j in top5[i]:
                    path_top5 = path_top5 + " " + str(j.item())
                out_file.write(path_top5 + "\n")
                # print(labels[i], "TOP5:", top5[i])
                # print(path_top5)
                total += 1

            gc.collect()
        #remove final newline
        out_file.seek(out_file.tell() - 2)
        out_file.truncate()
    else:  #print accuracy for all epochs on val set
        epoch = 1
        while epoch <= num_epochs:
            print("loading models/model.%s" % epoch)
            model.load_state_dict(torch.load("models/model.%s" % epoch))
            model.eval()

            # Calculate classification error and Top-5 Error
            # on training and validation datasets here
            printAccuracy(val_loader, device, model, "VALSET", epoch)

            gc.collect()
            epoch += 1


def printAccuracy(loader, device, model, name, epoch, max_iters=10000):
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
    print(epoch, name, "TOP 1:", str(num1 / total))
    print(epoch, name, "TOP 5:", str(num5 / total))


if __name__ == '__main__' and platform.system() == "Windows":
    print('Starting VALTESTING')
    run(number_of_epochs, output_period, size_of_batch, model_to_use)
    print('VALTESTING terminated')
elif platform.system() != "Windows":
    print('Starting VALTESTING NOTWINDOWS')
    run(number_of_epochs, output_period, size_of_batch, model_to_use)
    print('VALTESTING terminated')
