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
batch_size = 512
if sys.platform.startswith('win'):
    num_workers = 0 # 0表示不不⽤用额外的进程来加速读取数据
else:
    num_workers = 4
epochs = 10
train_iter = torch.utils.data.DataLoader(mnist_train,
batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test,
batch_size=batch_size, shuffle=False, num_workers=num_workers)


class LinearNet(nn.Module):
    def __init__(self,num_inputs, num_outputs):
        super(LinearNet,self).__init__()
        self.linear_1 = nn.Linear(num_inputs,50)
        self.act = nn.ReLU()
        self.linear_2 = nn.Linear(50,num_outputs)
        self.linear_3 = nn.Linear(50,50)

    def forward(self, x):
        x = self.linear_1(x.view(x.shape[0],-1))
        x = self.act(x)
        x = self.linear_3(x)
        x = self.act(x)
        x = self.linear_3(x)
        x = self.act(x)
        x = self.linear_3(x)
        x = self.act(x)

        y = self.linear_2(x)
        return y





# for name,param in model.named_parameters():
#     init.normal_(param, mean=0, std=1 )
def train():
    model = LinearNet(784, 10)

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
    predict_y = model(x[0]).argmax(dim = 1)
    print(get_fashion_mnist_labels(predict_y))
    for name,param in model.named_parameters():
        print(name)

    print(get_state_dict(model))
    print(get_state_dict(optimizer))
    torch.save(model.state_dict(),save_dir)

def load_model_predict():
    model = LinearNet(784, 10)
    model.load_state_dict(torch.load(save_dir))
    x, y = iter(test_iter).next()
    predict_y = model(x[0]).argmax(dim=1)
    print(get_fashion_mnist_labels(predict_y))


if __name__ == '__main__':
    train()
    # load_model_predict()