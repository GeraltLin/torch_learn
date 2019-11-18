import time
import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from jaychou_lyrics.utils import *
from torchtext import data
import torchtext
data_path = '../Datasets/jayzhou_lyrics'



def tokenizer(text):
    return list(text)
TEXT = data.Field(sequential=True,tokenize=tokenizer)


train = torchtext.datasets.LanguageModelingDataset(path="../Datasets/jayzhou_lyrics/jaychou_lyrics.csv",
    text_field=TEXT)

TEXT.build_vocab(train)
vocab = TEXT.vocab
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_iter, val_iter  = torchtext.data.BPTTIterator.splits(
    (train,train),  batch_sizes=(100,100), device=device, bptt_len=10, repeat=False, shuffle=False)

vocab_size = len(vocab)
print(vocab_size)
print(vocab.itos)
#
# for i in range(5):
#     print(i)
#     train_iter_bath = iter(train_iter)
#     for batch in train_iter_bath:
#         print(batch.text.size())
        # print(" ".join([TEXT.vocab.itos[i] for i in batch.text[:, 0].data])) #其中0指代的是第nth batch
        # print(" ".join([TEXT.vocab.itos[i] for i in batch.target[:,0].data]))







num_hidden = 256
rnn_layer = nn.LSTM(input_size=vocab_size, hidden_size=num_hidden)


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, vocab_size)
        self.state = None

    def forward(self, inputs, state):  # inputs: (batch, seq_len)
        # 获取one-hot向量表示
        # X :seq_len, batch, input_size
        X = to_onehot(inputs, vocab_size)  # X是个list
        # 合并形成batch
        Y, self.state = self.rnn(torch.stack(X), state)

        # Y : seq_len, batch, ouput_size
        # 全连接层会首先将Y的形状变成(num_steps * batch_size, num_hiddens)，它的输出
        # 形状为(num_steps * batch_size, vocab_size)
        # print(Y.size())
        output = self.dense(Y.view(-1, Y.shape[-1]))
        return output, self.state


def train_and_predict_rnn(model, device,
                          num_epochs, lr, clipping_theta,
                          pred_period,train_iter):
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    state = None
    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        train_iter_bath = iter(train_iter)

        # data_iter = data_iter_consecutive(corpus_indices, batch_size, num_steps, device)  # 相邻采样
        for batch in train_iter_bath:
            X = batch.text.data.transpose(0,1).to(device)
            Y = batch.target.data.transpose(0,1).to(device)
            if state is not None:
                # 使用detach函数从计算图分离隐藏状态, 这是为了
                # 使模型参数的梯度计算只依赖一次迭代读取的小批量序列(防止梯度计算开销太大)
                if isinstance(state, tuple):  # LSTM, state:(h, c)
                    state = (state[0].detach(), state[1].detach())
                else:
                    state = state.detach()

            (output, state) = model(X, state)  # output: 形状为(num_steps * batch_size, vocab_size)
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

            torch.save(model.state_dict(),'./save_models/rnnlm.pt')


def predict_rnn_pytorch(prefix, num_chars, model, device, idx_to_char,
                        char_to_idx):
    state = None
    output = [char_to_idx[prefix[0]]]  # output会记录prefix加上输出
    for t in range(num_chars + len(prefix) - 1):
        X = torch.tensor([output[-1]], device=device).view(1, 1)
        if state is not None:
            if isinstance(state, tuple):  # LSTM, state:(h, c)
                state = (state[0].to(device), state[1].to(device))
            else:
                state = state.to(device)

        (Y, state) = model(X, state)  # 前向计算不需要传入模型参数
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(dim=1).item()))
    return ''.join([idx_to_char[i] for i in output])


num_epochs, batch_size, lr, clipping_theta = 500, 32, 1e-3, 1e-2  # 注意这里的学习率设置
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']

model = RNNModel(rnn_layer, vocab_size)
num_steps = 10
batch_size = 100
state = None

# train_and_predict_rnn(model, device,
#                       num_epochs, lr, clipping_theta,
#                       pred_period,train_iter)

model.load_state_dict(torch.load('./save_models/rnnlm.pt', map_location="cuda:0"))
ans = predict_rnn_pytorch(prefix='爱情',num_chars=20,model=model,device='cpu',idx_to_char=vocab.itos,char_to_idx=vocab.stoi)
print(ans)