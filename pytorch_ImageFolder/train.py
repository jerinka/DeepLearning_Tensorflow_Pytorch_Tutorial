import numpy as np
import pandas as pd
import os
import random
import time

import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from PIL import Image
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Pytorch Convolutional Neural Network Model Architecture
class CatAndDogConvNet(nn.Module):

    def __init__(self):
        super().__init__()

        # onvolutional layers (3,16,32)
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=(5, 5), stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=(5, 5), stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=(3, 3), padding=1)

        # conected layers
        self.fc1 = nn.Linear(in_features= 64 * 6 * 6, out_features=500)
        self.fc2 = nn.Linear(in_features=500, out_features=50)
        self.fc3 = nn.Linear(in_features=50, out_features=2)


    def forward(self, X):

        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2)

        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2)

        X = X.view(X.shape[0], -1)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        X = F.softmax(X, dim=1)

        return X

def train():
    dataset_train = datasets.ImageFolder('../DB/cat_dog/train', transform=transform)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=32, shuffle=True)

    dataset_val = datasets.ImageFolder('../DB/cat_dog/val', transform=transform)
    dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=32, shuffle=True)

    images, labels = next(iter(dataloader_val))



    if 1:
        # Create instance of the model
        model = CatAndDogConvNet()

        losses = []
        accuracies = []
        epoches = 10
        start = time.time()
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

        # Model Training...
        for epoch in range(epoches):

            epoch_loss = 0
            epoch_accuracy = 0
            
            for X, y in dataloader_train:
                preds = model(X)
                loss = loss_fn(preds, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                accuracy = ((preds.argmax(dim=1) == y).float().mean())
                epoch_accuracy += accuracy
                epoch_loss += loss
                print('.', end='', flush=True)

            epoch_accuracy = epoch_accuracy/len(dataloader_train)
            accuracies.append(epoch_accuracy)
            epoch_loss = epoch_loss / len(dataloader_train)
            losses.append(epoch_loss)

            print("\n --- Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, time: {}".format(epoch, epoch_loss, epoch_accuracy, time.time() - start))

            # test set accuracy
            with torch.no_grad():

                test_epoch_loss = 0
                test_epoch_accuracy = 0

                for test_X, test_y in dataloader_val:

                    test_preds = model(test_X)
                    test_loss = loss_fn(test_preds, test_y)

                    test_epoch_loss += test_loss            
                    test_accuracy = ((test_preds.argmax(dim=1) == test_y).float().mean())
                    test_epoch_accuracy += test_accuracy

                test_epoch_accuracy = test_epoch_accuracy/len(dataloader_val)
                test_epoch_loss = test_epoch_loss / len(dataloader_val)

                print("Epoch: {}, test loss: {:.4f}, test acc: {:.4f}, time: {}\n".format(epoch, test_epoch_loss, test_epoch_accuracy, time.time() - start))

    #     torch.save(model.state_dict(), 'model1.pt')

    # model = CatAndDogConvNet()
    # model.load_state_dict(torch.load('model1.pt'))
    # model.eval()

    print(model)

    pred_list = []
    gt_list=[]
    start = time.time()

    with torch.no_grad():

        test_epoch_loss = 0
        test_epoch_accuracy = 0

        for test_X, test_y in dataloader_train:

            test_preds = model(test_X)
       
            test_accuracy = (test_preds.argmax(dim=1) == test_y).float().mean()
            print('test_accuracy:',test_accuracy)
            test_epoch_accuracy += test_accuracy

        test_epoch_accuracy = test_epoch_accuracy/len(dataloader_val)

        print("test acc: {:.4f}, time: {}\n".format( test_epoch_accuracy, time.time() - start))

    breakpoint()
    print('state dict:',model.state_dict().keys())

    checkpoint = { 'state_dict': model.state_dict(),'optimizer' :optimizer.state_dict()}
    torch.save(checkpoint, 'Checkpoint.pth')
    model2  = torch.load('Checkpoint.pth')

if __name__=='__main__':
    train()