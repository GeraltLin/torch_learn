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

class Inception(nn.Module):
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, in_c, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 线路1，单1 x 1卷积层
        self.p1_1 = nn.Conv2d(in_c, c1, kernel_size=1)
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = nn.Conv2d(in_c, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = nn.Conv2d(in_c, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_c, c4, kernel_size=1)

    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))
        return torch.cat((p1, p2, p3, p4), dim=1)  # 在通道维上连结输出


b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   GlobalAvgPool2d())

##########################################################################################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
epochs = 5
train_iter,test_iter = get_iter(data_dir = '../Datasets/FashionMNIST',batch_size=512,re_size=96)

#######################################################################################################

def train():
    model = nn.Sequential(b1, b2, b3, b4, b5, FlattenLayer(), nn.Linear(1024, 10))
    X = torch.rand(1, 1, 96, 96)
    for blk in model.children():
        X = blk(X)
        print('output shape: ', X.shape)
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
    input_x = x[0].view(1, 1, 96, 96)
    predict_y = model(input_x).argmax(dim=1)
    print(get_fashion_mnist_labels(predict_y))


if __name__ == '__main__':
    train()
