# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:19:39 2019

@author: shishi
"""

'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import rotate



import os
import argparse

from models import *
from utils import progress_bar


transformer1=transforms.Compose([
    transforms.RandomRotation(25),
    transforms.ToTensor()
])

transformer2=transforms.Compose([
    transforms.ToTensor()
])


trainset1=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transformer1)
trainset2=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transformer2)

trainloader1=torch.utils.data.DataLoader(trainset1)
trainloader2=torch.utils.data.DataLoader(trainset2)
trainloader=zip(trainloader1,trainloader2)

net = ResNet18()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
regularization_loss=nn.MSELoss()


net.train()
for (inputs1, targets1),(inputs2, targets2) in trainloader:
        optimizer.zero_grad()
        outputs1 = net(inputs1)
        outputs2 = net(inputs2)
        loss = criterion(outputs1, targets1)
        regularization=regularization_loss(outputs1,outputs2)
        total_loss=loss+regularization
        total_loss.backward()
        optimizer.step()
        print(targets1)



