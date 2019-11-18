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

# current_dir = os.path.dirname(os.path.realpath(__file__))
save_dir = 'save_models/model.pt'
class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


net = nn.Sequential(
    #   (224 + 2*3 - 1*(8-1)-1)/2 =112
        nn.Conv2d(1, 64, kernel_size=8, stride=2, padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
    # (112+2*1-1*(3-1)-1)/2=56
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

net.add_module("resnet_block1", resnet_block(64, 64, 2, first_block=True))
net.add_module("resnet_block2", resnet_block(64, 128, 2))
net.add_module("resnet_block3", resnet_block(128, 256, 2))
net.add_module("resnet_block4", resnet_block(256, 512, 2))

net.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
net.add_module("fc", nn.Sequential(FlattenLayer(), nn.Linear(512, 10)))

X = torch.rand((1, 1, 224, 224))
for name, layer in net.named_children():
    X = layer(X)
    print(name, ' output shape:\t', X.shape)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 1
train_iter,test_iter = get_iter(data_dir = '../Datasets/FashionMNIST',batch_size=128,re_size=224)


def train():
    model = net

    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_batches = len(train_iter)
    for epoch in range(epochs):
        train_l_batch, train_acc_batch, n = 0.0, 0.0, 0

        for i, (x, y) in enumerate(train_iter):
            # print(x,y)
            model = model.to(device)
            model.train()
            # print(model)
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = cross_entropy_loss(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_l_batch = loss.item()
            train_acc_batch = (y_hat.argmax(dim=1) == y).sum().item() / y.shape[0]
            if i % 10 == 0:
                print('epoch %d,%d /%d, loss %.4f, train acc %.3f'
                      % (epoch + 1, i, num_batches, train_l_batch, train_acc_batch,
                         ))
    evel_acc = evaluate_accuracy(test_iter, model)
    print('epoch %d, evel_acc %.3f'
          % (epoch + 1, evel_acc,
             ))

    x, y = iter(test_iter).next()
    x = x.to(device)
    input_x = x[0].view(1, 1, 224, 224)
    predict_y = model(input_x).argmax(dim=1)
    print(get_fashion_mnist_labels(predict_y))


if __name__ == '__main__':
    train()