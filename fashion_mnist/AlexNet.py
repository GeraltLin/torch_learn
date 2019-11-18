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

fc_features = 512 *7 *7
fc_hidden_units = 4096 # 任意

def vgg_block(num_convs,in_channels,out_channels):
    blk = []
    for i in range(num_convs):
        if i ==0:
            blk.append(nn.Conv2d(in_channels, out_channels,
                                 kernel_size =3, padding=1))
        else:
            blk.append(nn.Conv2d(out_channels, out_channels,
                                 kernel_size=3, padding=1))
        blk.append(nn.ReLU())
    blk.append(nn.MaxPool2d(kernel_size=2, stride =2))
    # * blk 元祖
    return nn.Sequential(* blk)

def vgg(conv_arch,fc_layers,fc_hidden_units=4096):
    net = nn.Sequential()
    for i, (num_convs, in_channels, out_channels) in enumerate(conv_arch):
        net.add_module("vgg_block_"+str(i+1),vgg)

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


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # # 减⼩小卷积窗⼝，使⽤用填充为2来使得输⼊入与输出的⾼高和宽⼀致，
            #  且增⼤大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使⽤用更更⼩小的卷积窗⼝。
            # 除了最后的卷积层外，进一步
            # 增⼤大了了输出通道数。
            # 前两个卷积层后不不使⽤用池化层来减小输⼊的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )

        self.fc = nn.Sequential(
            # (28-4)/2 = 12 (12-4)/2 = 4
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这⾥里里使⽤用Fashion-MNIST，所以⽤用类别数为10，⽽而⾮非论⽂文
            # 中的1000
            nn.Linear(4096, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


# for name,param in model.named_parameters():
#     init.normal_(param, mean=0, std=1 )
def train():
    model = AlexNet()

    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_batches = math.ceil(len(mnist_train) / batch_size)
    for epoch in range(epochs):
        train_l_batch, train_acc_batch, n = 0.0, 0.0, 0

        for i, (x, y) in enumerate(train_iter):
            # print(x,y)
            model = model.to(device)
            model.train()
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
