import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time
import os
from torch import nn
from torch.nn import init
import math
import sys
from fashion_mnist.utils import *
import torch.nn.functional as F


def evaluate_accuracy(data_iter, net):
    net.eval()
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        X = X.cuda()
        y = y.cuda()
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    net.train()
    return acc_sum / n


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover',
                   'dress', 'coat', 'sandal', 'shirt',
                   'sneaker', 'bag', 'ankleboot']
    return [text_labels[int(i)] for i in labels]


def get_state_dict(model):
    return model.state_dict()


class FlattenLayer(torch.nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):  # x shape: (batch, *, *, ...)
        return x.view(x.shape[0], -1)


class GlobalAvgPool2d(torch.nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


def get_iter(data_dir, batch_size, re_size):
    if sys.platform.startswith('win'):
        num_workers = 0  # 0表示不不⽤用额外的进程来加速读取数据
    else:
        num_workers = 4

    transform = []
    transform.append(torchvision.transforms.Resize(size=re_size))
    transform.append(torchvision.transforms.ToTensor())
    transform = torchvision.transforms.Compose(transform)

    mnist_train = torchvision.datasets.FashionMNIST(root=data_dir,
                                                    train=True, download=True, transform=transform)
    mnist_test = torchvision.datasets.FashionMNIST(root=data_dir,
                                                   train=False, download=True, transform=transform)
    train_iter = torch.utils.data.DataLoader(mnist_train,
                                             batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(mnist_test,
                                            batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter, test_iter
