import time
import copy
import torchvision.models as models
import torch
from torch.optim import lr_scheduler
import torch.nn as nn
from prepData import prep_data
from changeData import *


# This file should only be run with data already created
def train_model(model, criterion, optimizer, scheduler, trainLoad, validLoad, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                d = trainLoad
                x = 0
            else:
                model.eval()
                x=1
                d = validLoad
            dataset_sizes = [len(trainLoad), len(validLoad)]
            running_loss = 0.0
            running_corrects = 0

            for batch, (inputs, labels) in enumerate(d):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels[0])

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[x]
            epoch_acc = running_corrects.double() / dataset_sizes[x]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model


# data prep
trainLoad, validLoad, testLoad = prep_data("./data/original/", "./labels.csv")
trainLoad1, validLoad1, testLoad1 = prep_data("./data/change1/", "./labels.csv")
trainLoad2, validLoad2, testLoad2 = prep_data("./data/RCplot/", "./labels.csv")
trainLoad3, validLoad3, testLoad3 = prep_data("./data/RCplot1/", "./labels.csv")


net = models.resnet18(pretrained=True)
for param in net.parameters():
    param.requires_grad = False

num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
optimizer = torch.optim.SGD(net.fc.parameters(), lr=0.001, momentum=0.9)
net = net.to(device)
start = time.time()

criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
model_ft = train_model(net, criterion, optimizer, exp_lr_scheduler,
                       trainLoad, validLoad, num_epochs=25)
model_ft1 = train_model(net, criterion, optimizer, exp_lr_scheduler,
                       trainLoad1, validLoad1, num_epochs=25)
model_ft2 = train_model(net, criterion, optimizer, exp_lr_scheduler,
                       trainLoad2, validLoad2, num_epochs=25)
model_ft3 = train_model(net, criterion, optimizer, exp_lr_scheduler,
                       trainLoad3, validLoad3, num_epochs=25)