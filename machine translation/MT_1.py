from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cuda")

Lang1 = Field(eos_token='<eos>')
Lang2 = Field(init_token='<sos>')

train = TranslationDataset(path='../Datasets/MT_data/',
                           exts=('eng-fra.train.en', 'eng-fra.train.fr'),
                           fields=[('Lang1', Lang1), ('Lang2', Lang2)])

train_iter, val_iter, test_iter = BucketIterator.splits((train, train, train),
                                                        batch_size=256, repeat=False)
Lang1.build_vocab(train)
Lang2.build_vocab(train)

# for i, train_batch in enumerate(train_iter):
#     print('Lang1  : \n', [Lang1.vocab.itos[x] for x in train_batch.Lang1.data[:,0]])
#     print('Lang2 : \n', [Lang2.vocab.itos[x] for x in train_batch.Lang2.data[:,0]])
print(Lang1.vocab.stoi['<sos>'])
print(Lang2.vocab.itos)


class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(input_size=embed_size,
                          hidden_size=num_hiddens,
                          num_layers=num_layers,
                          dropout=drop_prob)

    def forward(self, inputs, state):
        embedding = self.embedding(inputs)

        return self.rnn(embedding, state)

    def begin_state(self):
        return None


def attention_model(input_size, attention_size):
    model = nn.Sequential(nn.Linear(input_size,
                                    attention_size,
                                    bias=False),
                          nn.Tanh(),
                          nn.Linear(attention_size,
                                    1,
                                    bias=False))
    return model


def attention_forward(model, enc_states, dec_state):
    # 将解码器隐藏状态广播到和编码器隐藏状态形状相同后进⾏行行连结
    dec_states = dec_state.unsqueeze(dim=0).expand_as(enc_states)

    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)

    # 10 4 16

    e = model(enc_and_dec_states)  # 形状为(时间步数, 批量量⼤大⼩小, 1)
    alpha = F.softmax(e, dim=0)  # 在时间步维度做softmax运算
    # print(alpha.size())
    # print(enc_states.size())
    # print((alpha * enc_states).size())
    # print((alpha * enc_states).sum(dim=0).size())
    # print('hi',(alpha * enc_states).sum(dim=0).size())

    return (alpha * enc_states).sum(dim=0)  # 返回背景变量量


class Decoder(nn.Module):
    def __init__(self,
                 vocab_size, embed_size,
                 num_hiddens, num_layers,
                 attention_size, drop_prob=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(2 * num_hiddens,
                                         attention_size).to(device)
        # GRU的输⼊入包含attention输出的c和实际输⼊入, 所以尺寸是2 * embed_size
        self.rnn = nn.GRU(2 * embed_size, num_hiddens, num_layers,
                          dropout=drop_prob)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, enc_states):
        """
        cur_input shape: (batch, )
        state shape: (num_layers, batch, num_hiddens)
        """
        # 使⽤用注意⼒力力机制计算背景向量量
        c = attention_forward(self.attention, enc_states,
                              state[-1])
        # 将嵌⼊入后的输⼊入和背景向量量在特征维连结

        input_and_c = torch.cat((self.embedding(cur_input), c), dim=1)  # (批量量⼤大⼩小, 2*embed_size)
        # 为输⼊入和背景向量量的连结增加时间步维，时间步个数为1
        output, state = self.rnn(input_and_c.unsqueeze(0), state)
        # 移除时间步维，输出形状为(批量量⼤大⼩小, 输出词典⼤大⼩小)
        output = self.out(output).squeeze(dim=0)
        return output, state

    def begin_state(self, enc_state):
        # 直接将编码器器最终时间步的隐藏状态作为解码器器的初始隐藏状态
        return enc_state


def batch_loss(encoder, decoder, X, Y, loss):
    batch_size = X.shape[1]
    enc_state = encoder.begin_state()
    enc_outputs, enc_state = encoder(X, enc_state)
    # 初始化解码器器的隐藏状态
    dec_state = decoder.begin_state(enc_state).to(device)
    # 解码器器在最初时间步的输⼊入是BOS
    dec_input = torch.tensor([Lang2.vocab.stoi['<sos>']] * batch_size).to(device)
    # 我们将使⽤用掩码变量量mask来忽略略掉标签为填充项PAD的损失
    mask, num_not_pad_tokens = torch.ones(batch_size, ).to(device), 0
    l = torch.tensor([0.0]).to(device)
    for y in Y:  # Y shape: (batch, seq_len)


        dec_output, dec_state = decoder(dec_input, dec_state,
                                        enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y  # 使⽤用强制教学
        num_not_pad_tokens += mask.sum().item()
        # 将PAD对应位置的掩码设成0, 原⽂文这⾥里里是 y != out_vocab.stoi[EOS],
        # 感觉有误
        mask = mask * (y != Lang2.vocab.stoi['<pad>']).float().to(device)
    return l / num_not_pad_tokens


def train(encoder, decoder, lr, num_epochs):
    enc_optimizer = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss(reduction='none')
    for epoch in range(num_epochs):
        l_sum = 0.0
        for i, train_batch in enumerate(train_iter):
            X = train_batch.Lang1.data.to(device)
            Y = train_batch.Lang2.data.to(device)
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            encoder = encoder.to(device)
            decoder = decoder.to(device)
            l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_optimizer.step()
            dec_optimizer.step()
            l_sum += l.item()
        if (epoch + 1) % 1 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum /
                                           len(train_iter)))


embed_size, num_hiddens, num_layers = 64, 64, 2
attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
encoder = Encoder(len(Lang1.vocab), embed_size, num_hiddens, num_layers, drop_prob)
decoder = Decoder(len(Lang2.vocab), embed_size, num_hiddens, num_layers, attention_size, drop_prob)
train(encoder, decoder, lr, num_epochs)



