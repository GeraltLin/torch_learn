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

def get_iter(data_dir, batch_size,re_size):
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
    return train_iter,test_iter