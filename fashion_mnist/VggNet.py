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

conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256),
             (2, 256, 512), (2, 512, 512))





def vgg_block(num_convs, in_channels, out_channels):

    blk = []
    for i in range(num_convs):
        if i == 0:
            blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*blk)



def VggNet(conv_arch,fc_features,fc_hidden_units=4096):
    net = nn.Sequential()
    # 卷积层部分
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_" + str(i + 1), vgg_block(num_convs, in_channels, out_channels))
    # 全连接层部分
    net.add_module("fc", nn.Sequential(FlattenLayer(),
                                       nn.Linear(fc_features, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, fc_hidden_units),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Linear(fc_hidden_units, 10)
                                       ))
    return net

mnist_train = torchvision.datasets.FashionMNIST(root=data_dir,
                                                train=True, download=True, transform=transform)
mnist_test = torchvision.datasets.FashionMNIST(root=data_dir,
                                               train=False, download=True, transform=transform)
import time

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

    ratio = 8
    small_conv_arch = [(1, 1, 64 // ratio), (1, 64 // ratio, 128 // ratio), (2, 128 // ratio, 256 // ratio),
                       (2, 256 // ratio, 512 // ratio), (2, 512 // ratio, 512 // ratio)]
    model = VggNet(small_conv_arch, fc_features // ratio, fc_hidden_units // ratio)


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
