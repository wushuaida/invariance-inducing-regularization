'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch




transformer1=transforms.Compose([
    transforms.RandomRotation(25),
    transforms.ToTensor()
])

transformer2=transforms.Compose([
    transforms.ToTensor()
])


trainset1=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transformer1)
trainset2=torchvision.datasets.CIFAR10(root='./data',train=True,download=True,transform=transformer2)

trainloader1=torch.utils.data.DataLoader(trainset1, batch_size=128,
                                          shuffle=False, num_workers=2)
trainloader2=torch.utils.data.DataLoader(trainset2, batch_size=128,
                                          shuffle=False, num_workers=2)
trainloader=zip(trainloader1,trainloader2)




transform_test = transforms.Compose([
    transforms.ToTensor(),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')




#show image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()








#inputs, classes = next(iter(trainset))   take a sample from dataset

# Model
print('==> Building model..')
# net = VGG('VGG19')
net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
#net = EfficientNetB0()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
regularization_loss=nn.MSELoss()


def adv_train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, ((inputs1, targets1),(inputs2, targets2)) in enumerate(trainloader):
        inputs1, targets1 = inputs1.to(device), targets1.to(device)
        inputs2, targets2 = inputs2.to(device), targets2.to(device)
        optimizer.zero_grad()
        outputs1 = net(inputs1)
        outputs2 = net(inputs2)
        loss = criterion(outputs1, targets1)
        regularization=regularization_loss(outputs1,outputs2)
        total_loss=loss+regularization
        total_loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs1.max(1)
        total += targets1.size(0)
        correct += predicted.eq(targets1).sum().item()

        progress_bar(batch_idx, 390, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))




def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


        
for epoch in range(0, 200+1):
    trainloader=zip(trainloader1,trainloader2)
    adv_train(epoch)
    test(epoch)
