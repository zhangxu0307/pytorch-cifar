'''Train CIFAR10 with PyTorch.'''

import torch as th
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
from torch.autograd import Variable

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

from utils import saveModel, loadModel


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()


def adjust_lr(optimizer, epoch, lamda=50):

    # exp adjust
    # lr = args.lr*(0.1**(epoch//lamda))
    # if lr >= 1e-6:
    #     for param_group in optimizer.param_groups:
    #         param_group['lr'] = lr

    # schedule lr adjust
    lr = 0.1
    if epoch >= 0 and epoch < 150:
        lr = 0.1
    if epoch >= 150 and epoch < 250:
        lr = 0.01
    if epoch >= 250 and epoch < 350:
        lr = 0.001

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def loadCIFAR10(batchSize):
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader


def train(model, batchSize, epoch, checkPoint, savePoint, modelPath,  curEpoch=0, best_acc = 0, useCuda=True, earlyStop=True, tolearnce=2):

    tolerance_cnt = 0
    step = 0

    if useCuda:
        model = model.cuda()

    ceriation = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(net.parameters(), lr=args.lr)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    trainLoader, testLoader = loadCIFAR10(batchSize=batchSize)

    for i in range(curEpoch, curEpoch+epoch):

        model.train()

        # trainning
        sum_loss = 0

        for batch_idx, (x, target) in enumerate(trainLoader):

            optimizer.zero_grad()
            adjust_lr(optimizer, epoch)

            if useCuda:
                x, target = x.cuda(), target.cuda()

            x, target = Variable(x), Variable(target)
            out = model(x)

            loss = ceriation(out, target)
            sum_loss += loss.item()

            loss.backward()
            optimizer.step()

            step += 1

            if (batch_idx + 1) % checkPoint == 0 or (batch_idx + 1) == len(trainLoader):
                print('==>>> epoch: {}, batch index: {}, train loss: {:.6f}'.format(i, batch_idx + 1, sum_loss/(batch_idx+1)))

            # save model every savepoint steps
            # if (step + 1) % savePoint == 0:
            #     saveModel(model, i, best_acc, modelPath)
            #     print("----------save finish----------------")

        acc = test(net, testLoader, useCuda=True)

        if earlyStop:
            # early stopping
            if acc < best_acc:
                tolerance_cnt += 1
            else:
                best_acc = acc
                tolerance_cnt = 0
            if tolerance_cnt >= tolearnce:
                print("early stopping training....")
                break

        # save model when test acc is highest
        if best_acc < acc:
            saveModel(model, epoch, acc, modelPath)

    saveModel(model, epoch, best_acc, modelPath)


def test(model, testLoader, useCuda=True):

    correct_cnt, sum_loss = 0, 0
    total_cnt = 0

    model.eval()

    for batch_idx, (x, target) in enumerate(testLoader):
        x, target = Variable(x, volatile=True), Variable(target, volatile=True)
        if useCuda:
            x, target = x.cuda(), target.cuda()
        out = model(x)

        _, pred_label = torch.max(out.data, 1)
        total_cnt += x.data.size()[0]
        correct_cnt += (pred_label == target.data).sum()
        correct_cnt = correct_cnt.item()

    acc = (correct_cnt * 1.0 / float(total_cnt))
    print("acc:", acc)
    return acc


if __name__ == '__main__':

    # Model

    modelPath = "./model/mobilenetv2"

    print('==> Building model..')
    # net = VGG('VGG19')
    # net = ResNet18()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    #net = SENet18()

    if args.resume:
        # Load checkpoint.
        net, best_acc, curEpoch = loadModel(modelPath, net)
    else:
        best_acc = 0
        curEpoch = 0

    print("current epoch:", curEpoch)
    print("current best acc:", best_acc)

    use_cuda = torch.cuda.is_available()
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    #cudnn.benchmark = True

    train(model=net, batchSize=128, epoch=350, checkPoint=10, savePoint=500, modelPath=modelPath,
          useCuda=True, best_acc=best_acc, curEpoch=curEpoch)




