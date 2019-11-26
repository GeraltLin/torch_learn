import math
import torch
import torch.nn as nn
from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset
from torch import nn
import torch
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer


Lang1 = Field(eos_token='<eos>')
Lang2 = Field(init_token='<sos>')
train = TranslationDataset(path='../Datasets/MT_data/',
                           exts=('eng-fra.train.fr', 'eng-fra.train.en'),
                           fields=[('Lang1', Lang1), ('Lang2', Lang2)])

train_iter, val_iter, test_iter = BucketIterator.splits((train, train, train),
                                                        batch_size=256, repeat=False)
Lang1.build_vocab(train)
Lang2.build_vocab(train)

class TransformerModel(nn.Module):

    def __init__(self, enc_vocab_size, dec_vocab_size, embed_size, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = TransformerEncoderLayer(embed_size, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(enc_vocab_size, embed_size)
        self.embed_size = embed_size
        self.decoder = nn.Linear(embed_size, dec_vocab_size)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        """
        ([[0., 0., 0., 0., 0.],
        [-inf, 0., 0., 0., 0.],
        [-inf, -inf, 0., 0., 0.],
        [-inf, -inf, -inf, 0., 0.],
        [-inf, -inf, -inf, -inf, 0.]])
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
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

import torchtext
from torchtext.data.utils import get_tokenizer
TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)

enc_vocab_size = len(Lang1.vocab.stoi)
dec_vocab_size = len(Lang2.vocab.stoi)

embed_size = 200
nhid = 200 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value

model = TransformerModel(enc_vocab_size, dec_vocab_size, embed_size, nhead, nhid, nlayers, dropout).to(device)