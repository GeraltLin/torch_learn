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
data_dir = '../Datasets/FashionMNIST'
save_dir = 'save_models/model.pt'
transform = []
transform.append(torchvision.transforms.Resize(size=224))
transform.append(torchvision.transforms.ToTensor())
transform = torchvision.transforms.Compose(transform)




def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU(),
                        nn.Conv2d(out_channels, out_channels, kernel_size=1),
                        nn.ReLU())
    return blk

net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, stride=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, stride=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, stride=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, stride=1, padding=1),
    # 全局平均池化层可通过将窗口形状设置成输入的高和宽实现
    nn.AvgPool2d(kernel_size=5),
    # 将四维的输出转成二维的输出，其形状为(批量大小, 10)
    FlattenLayer())



mnist_train = torchvision.datasets.FashionMNIST(root=data_dir,
                                                train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.FashionMNIST(root=data_dir,
                                               train=False, download=True, transform=transform)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 512
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不不⽤用额外的进程来加速读取数据
else:
    num_workers = 4
num_workers = 4
epochs = 5
train_iter = torch.utils.data.DataLoader(mnist_train,
                                         batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,
                                        batch_size=batch_size, shuffle=False, num_workers=num_workers)



# for name,param in model.named_parameters():
#     init.normal_(param, mean=0, std=1 )
def train():
    fc_features = 512 * 7 * 7
    fc_hidden_units = 4096  # 任意


    model = net
    X = torch.rand(1, 1, 224, 224)

    for name, blk in model.named_children():
        X = blk(X)
        print(name, 'output shape: ', X.shape)


    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_batches = math.ceil(len(mnist_train) / batch_size)
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
    # load_model_predict()
