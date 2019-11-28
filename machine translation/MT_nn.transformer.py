import math
import torch
import torch.nn as nn
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from torch import nn
import torch
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

device = torch.device("cuda")

Lang1 = Field(eos_token='<eos>')

Lang2 = Field(init_token='<sos>',eos_token='<eos>')
train = TranslationDataset(path='../Datasets/MT_data/',
                           exts=('eng-fra.train.fr', 'eng-fra.train.en'),
                           fields=[('Lang1', Lang1), ('Lang2', Lang2)])

train_iter, val_iter, test_iter = BucketIterator.splits((train, train, train),
                                                        batch_size=16, repeat=False)
Lang1.build_vocab(train)
Lang2.build_vocab(train)

# for i, train_batch in enumerate(train_iter):
#     print('Lang1  : \n', [Lang1.vocab.itos[x] for x in train_batch.Lang1[0].data[:, 0]])
#     print('Lang1  : \n', train_batch.Lang1[1].data[0])
#     print('Lang2 : \n', [Lang2.vocab.itos[x] for x in train_batch.Lang2[0].data[:, 0]])
#     print('Lang2 : \n', train_batch.Lang2[1].data[0])
    # print('Lang2 : \n', [Lang2.vocab.itos[x] for x in train_batch.Lang2[0].data[:,0]])


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=20):
        """
        d_model 常等于 embed_size
        (1/10000)^(2i/dmodel) = e^(2i)*(-log(10000)/dmodel)
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # register_buffer 不被更新的量
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class LanguageTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, max_seq_length, pos_dropout, trans_dropout):
        super().__init__()
        self.d_model = d_model
        self.embed_src = nn.Embedding(src_vocab_size, d_model)
        self.embed_tgt = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, pos_dropout, max_seq_length)

        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                                          trans_dropout)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask):
        # src = rearrange(src, 'n s -> s n')
        # tgt = rearrange(tgt, 'n t -> t n')
        # print(src.size())

        src = self.pos_enc(self.embed_src(src) * math.sqrt(self.d_model))
        tgt = self.pos_enc(self.embed_tgt(tgt) * math.sqrt(self.d_model))
        src = src.transpose(1, 0)
        tgt = tgt.transpose(1, 0)

        # print(src.size())
        # print(tgt.size())
        output = self.transformer(src, tgt,  src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask,tgt_mask=tgt_mask)
        # output = rearrange(output, 't n e -> n t e')
        output = output.transpose(0,1)
        # print(output.size())
        # # print(Y.view(-1, Y.shape[-1]))
        return self.fc(output)

    def generate_square_subsequent_mask(self, sz):
        r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
            Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


num_epochs = 1000
max_seq_length = 20
src_vocab_size = len(Lang1.vocab.itos)
tgt_vocab_size = len(Lang2.vocab.itos)
d_model = 512
num_encoder_layers = 2
num_decoder_layers = 2
dim_feedforward = 512
nhead = 8
pos_dropout = 0.1
trans_dropout = 0.1
model = LanguageTransformer(src_vocab_size,
                            tgt_vocab_size,
                            d_model, nhead,
                            num_encoder_layers,
                            num_decoder_layers,
                            dim_feedforward,
                            max_seq_length,
                            pos_dropout,
                            trans_dropout)
lr = 0.01 # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
print(Lang1.vocab.itos)
print(Lang2.vocab.itos)

# print(Lang1.vocab.stoi['<pad>'])
# print(Lang2.vocab.stoi['<pad>'])


# for i, train_batch in enumerate(train_iter):
#     print('Lang1  : \n', [Lang1.vocab.itos[x] for x in train_batch.Lang1.data[:,0]])
#     print('Lang2 : \n', [Lang2.vocab.itos[x] for x in train_batch.Lang2.data[:,0]])
#



#
for epoch in range(num_epochs):
    l_sum = 0.0
    for i, train_batch in enumerate(train_iter):
        X = train_batch.Lang1.data.transpose(0, 1).to(device)

        label = train_batch.Lang2.data.transpose(0, 1).to(device)

        Y = label[:,:-1]


        # src_key_padding_mask = X.clone()
        src_key_padding_mask = X == 1

        # tgt_key_padding_mask = Y.clone()
        tgt_key_padding_mask = Y == 1

        # src_key_padding_mask = torch.ones_like(X, dtype=torch.uint8)
        # pad_x = train_batch.Lang1[1].data
        # for index, num in enumerate(pad_x.tolist()):
        #     src_key_padding_mask[0:num, index] = 0
        # src_key_padding_mask = src_key_padding_mask == 1
        # src_key_padding_mask = src_key_padding_mask.permute(1,0)
        #
        # tgt_key_padding_mask = torch.ones_like(Y, dtype=torch.uint8)
        # pad_y = train_batch.Lang2[1].data
        # for index, num in enumerate(pad_y.tolist()):
        #     tgt_key_padding_mask[0:num, index] = 0
        # tgt_key_padding_mask = tgt_key_padding_mask == 1
        # tgt_key_padding_mask = tgt_key_padding_mask.permute(1,0)

        memory_key_padding_mask = src_key_padding_mask.clone().to(device)
        tgt_mask = model.generate_square_subsequent_mask(sz=Y.size()[1]).to(device)
        # print(tgt_mask)

        # print(src_key_padding_mask)
        model = model.to(device)
        outputs = model(X, Y, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask,
                                  tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)

        tgt_out = label[:,1:]
        # print(tgt_out.tolist())
        # print(outputs[0])
        # print(tgt_out.tolist())
        # tgt_out = .reshape(-1)# (b o)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        tgt_out =  tgt_out.reshape(-1)

        loss = F.cross_entropy(
            outputs,
            tgt_out,
        )
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_acc_batch = (outputs.argmax(dim=1) == tgt_out).sum().item() / tgt_out.shape[0]

        # print(label[1:,:][:,0])
        # print(tgt_mask)
        # print('loss',loss.item())



        l_sum += loss.item()
    if (epoch + 1) % 1 == 0:
        print("epoch %d, loss %.3f" % (epoch + 1, l_sum /
                                       len(train_iter)))
        print(train_acc_batch)
        print(outputs.argmax(dim=1).tolist())
        print(tgt_out.tolist())

        # print(outputs)
        # print(outputs.argmax(dim=1))


# #
# torch.save(self.state_dict(), self.seq2seq_save_dir)