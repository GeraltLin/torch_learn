import time
import math
import numpy as np
import torch
from torch import nn,optim
import torch.nn.functional as F
from jaychou_lyrics.utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

corpus_indices, char_to_idx, idx_to_char, vocab_size = load_data_jay_lyrics('../Datasets/jayzhou_lyrics/jaychou_lyrics.txt')
num_hidden = 256
rnn_layer = nn.LSTM(input_size=vocab_size,hidden_size=num_hidden)

class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state): # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        # X :seq_len, batch, input_size
        X = to_onehot(inputs, vocab_size) # X是个list
        # 合并形成batch
        Y, self.state = self.rnn(torch.stack(X), state)

        # Y : seq_len, batch, ouput_size
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state

def train_and_predict_rnn(model, num_hiddens, vocab_size, device,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device) # 相邻采样

        for X, Y in data_iter:
            X = X.to(device)
            Y = Y.to(device)
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()
            (output, state) = model(X, state) # output: 形状为(num_steps * batch_size, vocab_size)


            y = torch.transpose(Y, 0, 1).contiguous().view(-1)

            l = loss(output, y.long())
            optimizer.zero_grad()
            l.backward()
            nn.utils.clip_grad_norm(model.parameters(), clipping_theta)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        try:
            perplexity = math.exp(l_sum / n)
        except OverflowError:
            perplexity = float('inf')
        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f' % (
                epoch + 1, perplexity))
num_epochs, batch_size, lr, clipping_theta = 500, 32, 1e-3, 1e-2 # 注意这里的学习率设置
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

model = RNNModel(rnn_layer, vocab_size)
num_steps = 10
batch_size = 100
state = None

train_and_predict_rnn(model, num_hidden, vocab_size, device,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
