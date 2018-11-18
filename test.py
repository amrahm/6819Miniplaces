import gc
import platform
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from tqdm import tqdm

import dataset
from models.AlexNet import *
from models.ResNet import *
from models.SENet import *
#import parameters (not sure if all are needed)
number_of_epochs = 500
output_period = 100
size_of_batch = 32
model_to_use = se_resnext50_32x4d()

torch.backends.cudnn.benchmark = True

def run(num_epochs, out_period, batch_size, model):
    # setup the device for running
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epoch = 8  #default epoch to load
    if len(sys.argv) > 1:  #first sysarg is name of script
        epoch = sys.argv[1]  #take number
    # load model
    print("loading models/model.%s" % epoch)
    model.load_state_dict(torch.load("models/model.%s" % epoch))
    model.eval()

    val_loader, test_loader = dataset.get_val_test_loaders(batch_size)

    # Opens file to write results to, will overwrite existing files
    out_file = open("results.txt", "w")
    total = 0
    for (inputs, labels) in tqdm(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        top5 = torch.topk(outputs, 5)[1]
        # path = "test/" + '{0:08d}'.format(i) + ".jpg"
        for i in range(len(inputs)):
            #do we need the whole path?
            # https://piazza.com/class/jllg8twvahl3pc?cid=696
            #lmao, why did they give us the example in the other format
            # no they want it like this: test/00000001.jpg
            # not like ./data/test/999/00000001.jpg
            # oh gotcha
            filename = test_loader.dataset.samples[total][0][-12:-1]
            # print(filename)
            # formats string in the structure of "test/00000132.jpg 1 3 5 6 9"
            path_top5 = "test/" + filename + "g"  #lol, SFB ETU RFC
            #I should send a screenshot to max goldman
            #with comments as communication left in
            #also, it all looks good, lets submit
            # :thumbs_up:
            # Also i have been trying to use the chat feature but it might
            # be broken
            #I don't have the extra addon for it :thumbs_down:
            #lol ok, I'll download it for next time
            # i like this method of communication though
            # its immersive and prevents context switching
            #lol
            # i feel like we will hit something like 69% for top5 accuracy
            #i fuckin bet
            #your kerb is menioa, right? yeeee
            #Oh fuck I submitted this one, not the one from aws lol
            #ok lemme run it there
            for j in top5[i]:
                path_top5 = path_top5 + " " + str(j.item())
            out_file.write(path_top5 + "\n")
            print(labels[i], "TOP5:", top5[i])
            print(path_top5)
            total += 1
        # # top1 = torch.topk(outputs,1)[1]
        # for i in range(len(inputs)):
        #     # print("\nlabel:", labels[i].item(),"\nTop 5:", top5[i])
        #     top1 = top5[i][0]
        #     num1 += 1 if labels[i].item() == top1.item() else 0
        #     num5 += 1 if labels[i].item() in top5[i] else 0
        #     total += 1

        gc.collect()


if __name__ == '__main__' and platform.system() == "Windows":
    print('Starting testing')
    run(number_of_epochs, output_period, size_of_batch, model_to_use)
    print('Testing terminated')
elif platform.system() != "Windows":
    print('Starting testing NOTWINDOWS')
    run(number_of_epochs, output_period, size_of_batch, model_to_use)
    print('Testing terminated')
