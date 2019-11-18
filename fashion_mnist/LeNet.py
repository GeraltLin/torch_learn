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
mnist_train = torchvision.datasets.FashionMNIST(root=data_dir,
train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root=data_dir,
train=False, download=True, transform=transforms.ToTensor())
import time
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0 # 0表示不不⽤用额外的进程来加速读取数据
else:
    num_workers = 4
epochs = 1
train_iter = torch.utils.data.DataLoader(mnist_train,
batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,
batch_size=batch_size, shuffle=False, num_workers=num_workers)


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1,6,5),
            nn.BatchNorm2d(6),

            nn.ReLU(),

            nn.MaxPool2d(2,2),
            nn.Conv2d(6,16,5),
            nn.BatchNorm2d(16),

            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            #(28-4)/2 = 12 (12-4)/2 = 4
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output






# for name,param in model.named_parameters():
#     init.normal_(param, mean=0, std=1 )
def train():
    model = LeNet()

    cross_entropy_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    num_batches = math.ceil(len(mnist_train) / batch_size)
    for epoch in range(epochs):
        train_l_batch, train_acc_batch, n = 0.0, 0.0, 0

        for i,(x,y) in enumerate(train_iter):
            # print(x,y)
            model = model.to(device)
            model.train()
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            loss = cross_entropy_loss(y_hat,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_l_batch = loss.item()
            train_acc_batch = (y_hat.argmax(dim=1) ==y).sum().item()/y.shape[0]
            if i%10==0:
                print('epoch %d,%d /%d, loss %.4f, train acc %.3f'
                      % (epoch + 1,i,num_batches,train_l_batch, train_acc_batch,
                         ))
    evel_acc = evaluate_accuracy(test_iter,model)
    print('epoch %d, evel_acc %.3f'
          % (epoch + 1,  evel_acc,
         ))

    x,y = iter(test_iter).next()
    x = x.to(device)
    input_x = x[0].view(1,1,28,28)
    predict_y = model(input_x).argmax(dim = 1)
    print(get_fashion_mnist_labels(predict_y))





if __name__ == '__main__':
    train()
    # load_model_predict()