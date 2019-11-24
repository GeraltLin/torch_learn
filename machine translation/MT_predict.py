from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from torch import nn
import torch
import torch.nn.functional as F

device = torch.device("cpu")

# Lang1 = Field(eos_token='<eos>')
# Lang2 = Field(init_token='<sos>',eos_token='<eos>')
#
# train = TranslationDataset(path='../Datasets/MT_data/',
#                            exts=('eng-fra.train.fr', 'eng-fra.train.en'),
#                            fields=[('Lang1', Lang1), ('Lang2', Lang2)])
#
# train_iter, val_iter, test_iter = BucketIterator.splits((train, train, train),
#                                                         batch_size=256, repeat=False)
# Lang1.build_vocab(train)
# Lang2.build_vocab(train)

# for i, train_batch in enumerate(train_iter):
#     print('Lang1  : \n', [Lang1.vocab.itos[x] for x in train_batch.Lang1.data[:,0]])
#     print('Lang2 : \n', [Lang2.vocab.itos[x] for x in train_batch.Lang2.data[:,0]])
# print(Lang1.vocab.itos)
# print(Lang2.vocab.itos)
# print(Lang2.vocab.stoi['<sos>'])


class Encoder(nn.Module):

    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, drop_prob):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(input_size=embed_size,
                          hidden_size=num_hiddens,
                          num_layers=num_layers,
                          dropout=drop_prob,
                          batch_first=True)

    def forward(self, inputs, state):
        embedding = self.embedding(inputs)

        return self.rnn(embedding, state)

    def begin_state(self):
        return None

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
                          dropout=drop_prob,batch_first=True)
        self.out = nn.Linear(num_hiddens, vocab_size)

    def forward(self, cur_input, state, enc_states):
        """
        cur_input shape: (batch, )
        state shape: (num_layers, batch, num_hiddens)
        """
        # 使⽤用注意⼒力力机制计算背景向量量
        c = attention_forward(self.attention, enc_states,
                              state[-1])
        # print('c',c.size()) #256 64
        # 将嵌⼊入后的输⼊入和背景向量量在特征维连结

        input_and_c = torch.cat((self.embedding(cur_input), c), dim=1)  # (批量量⼤大⼩小, 2*embed_size)
        # print('=',input_and_c.size())
        # 为输⼊入和背景向量量的连结增加时间步维，时间步个数为1
        output, state = self.rnn(input_and_c.unsqueeze(1), state)
        # 移除时间步维，输出形状为(批量量⼤大⼩小, 输出词典⼤大⼩小)
        output = self.out(output).squeeze(dim=1)
        return output, state

    def begin_state(self, enc_state):
        # 直接将编码器器最终时间步的隐藏状态作为解码器器的初始隐藏状态
        return enc_state

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
    # print(dec_state.size())
    # print(enc_states.size())
    dec_states = dec_state.unsqueeze(dim=1).expand_as(enc_states)

    enc_and_dec_states = torch.cat((enc_states, dec_states), dim=2)
    # 10 4 16
    e = model(enc_and_dec_states)  # 形状为(批量量⼤⼩,时间步数, 1)
    alpha = F.softmax(e, dim=1)  # 在时间步维度做softmax运算
    # print(alpha.size())
    # print(enc_states.size())
    # print((alpha * enc_states).size())
    # print((alpha * enc_states).sum(dim=0).size())
    # print('hi',(alpha * enc_states).sum(dim=1).size())
    return (alpha * enc_states).sum(dim=1)  # 返回背景变量量





import json
with open('encoder_vocab.json','r',encoding='utf8') as f:
    encoder_vocab_stoi = json.load(f,encoding ='utf8')
    encoder_vocab_itos = {v:k for k,v in encoder_vocab_stoi.items()}
with open('decoder_vocab.json','r',encoding='utf8') as f:
    decoder_vocab_stoi = json.load(f,encoding ='utf8')
    decoder_vocab_itos = {v:k for k,v in decoder_vocab_stoi.items()}


embed_size, num_hiddens, num_layers = 100, 100, 1
attention_size, drop_prob, lr, batch_size, num_epochs = 100, 0.5, 0.01, 2, 80
encoder = Encoder(len(encoder_vocab_stoi), embed_size, num_hiddens, num_layers, drop_prob)
decoder = Decoder(len(decoder_vocab_stoi), embed_size, num_hiddens, num_layers, attention_size, drop_prob)

encoder_save_dir = 'save_models/encoder.pt'
encoder.load_state_dict(torch.load(encoder_save_dir,map_location=device))
decoder_save_dir = 'save_models/decoder.pt'
decoder.load_state_dict(torch.load(decoder_save_dir,map_location=device))



def translate(encoder, decoder, input_seq, max_seq_len):
    encoder.eval()
    decoder.eval()
    in_tokens = input_seq.split(' ')
    in_tokens += ['<eos>'] + ['<pad>'] * (max_seq_len - len(in_tokens) - 1)
    enc_input = torch.tensor([[ encoder_vocab_stoi[tk] for tk in
    in_tokens]]) # batch=1
    enc_state = encoder.begin_state()
    enc_output, enc_state = encoder(enc_input.view(1, -1), enc_state)
    dec_input = torch.tensor([decoder_vocab_stoi['<sos>']])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state,
        enc_output)
        pred = dec_output.argmax(dim=1)
        pred_token = decoder_vocab_itos[int(pred.item())]
        if pred_token == '<eos>': # 当任⼀一时间步搜索出EOS时，输出序列列即完成
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens



input_seq = 'Voulez-vous aller à un match de football ?'
output_tokens = translate(encoder, decoder, input_seq, 10)
print(output_tokens)