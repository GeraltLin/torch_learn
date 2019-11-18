import torch
from NER_task.SequenceTaggingDataset import SequenceTaggingDataset
from torchtext import data,datasets
from torchtext import data
import torch
from torch import nn
from torchcrf import CRF
device = torch.device("cuda")


def light_tokenize(sequence: str):
    return [sequence]
TEXT = data.Field(sequential=True, tokenize=light_tokenize,include_lengths=True)
LABEL = data.ReversibleField(sequential=True, tokenize=light_tokenize, unk_token=None, is_target=True)

save_dir = 'save_models/model.pt'

train = SequenceTaggingDataset(
        path='../Datasets/NER_data/train.txt',separator=' ',
        fields=[('text', TEXT),
                ('label', LABEL)])

valid = SequenceTaggingDataset(
        path='../Datasets/NER_data/valid.txt',separator=' ',
        fields=[('text', TEXT),
                ('label', LABEL)])

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter = data.BucketIterator.splits(
    (train, valid),  # 构建数据集所需的数据集
    batch_sizes=(64, 64),
    # 如果使用gpu，此处将-1更换为GPU的编号
    device=device,
    # the BucketIterator needs to be told what function it should use to group the data.
    sort_key=lambda x: len(x.text),
    sort_within_batch=False,
    repeat=False
)
print(TEXT.vocab.stoi)
print(LABEL.vocab.stoi)

# for i, train_batch in enumerate(train_iter):
#     # print('text  : \n', train_batch.text)
#     # print('label : \n', train_batch.label)
#     # print(train_batch.text[0].size())
#     # print(train_batch.text)
#     # print(train_batch.label)
#     print(train_batch.label)
#     print(train_batch.label.size())
#     print(train_batch.label[:,1])
#     print('t',train_batch.text[0][:,])
#     print('l',train_batch.text[1])
    # print(train_batch.label[0].size())
# for i, valid_batch in enumerate(val_iter):
#     print(valid_batch.text.size())

    # print('text  : \n', valid_batch.text)
    # print('label : \n', valid_batch.label)


class LSTM(nn.Module):

    def __init__(self,istrain=True):
        super(LSTM, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 200)  # embedding之后的shape: torch.Size([200, 8, 300])
        # 若使用预训练的词向量，需在此处指定预训练的权重
        # embedding.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=200, hidden_size=128, num_layers=1,batch_first = True)  # torch.Size([200, 8, 128])
        self.decoder = nn.Linear(128,len(LABEL.vocab))
        self.dropout = nn.Dropout(p = 0.1)
        self.crf= CRF(len(LABEL.vocab), batch_first=True)
        self.istrain = istrain

    def forward(self, input_data):
        # sentence: batch,seq_len
        # print(sentence.size())
        sentence,label,mask = input_data
        embeds = self.word_embeddings(sentence)
        # embeds:  batch,seq_len,input_size
        # print(embeds.size())
        # LSTM input (batch, seq_len, input_size)
        # LSTM output (batch, seq_len, num_directions * hidden_size)
        lstm_out, (hn, cn)  = self.lstm(embeds) # lstm_out:200x8x128
        # print(lstm_out.size())
        # print(lstm_out.size())
        # 取最后一个时间步

        final = lstm_out  # 8*128
        #print(final.size())

        y_hat = self.decoder(final)  # 8*2
        y_hat = self.dropout(y_hat)
        if self.istrain:
            out_put = -self.crf(y_hat, label, reduction='mean', mask=mask)
        else:
            out_put = self.crf.decode(y_hat)
        return out_put




#
#
model = LSTM()
model.load_state_dict(torch.load(save_dir))

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# crf = CRF(len(LABEL.vocab), batch_first=True).to(device)
epochs = 50
for epoch in range(epochs):
    print('----------- epoch {} ------------'.format(epoch))

    for i, batch in enumerate(train_iter):
            model.train()
            # print(batch.text.data.size())
            X = batch.text[0].data.transpose(0, 1).to(device)
            # print(X.size()) # batch_size lenth
            mask = torch.zeros_like(X,dtype = torch.uint8)
            max_num = X.size()[-1]
            pad_x = batch.text[1].data
            for index, num in enumerate(pad_x.tolist()):
                mask[index,0:num] = 1
                # mask[]
            # pad = batch.text[1].data.transpose(0, 1).to(device)
            # print(X.size())
            # X = batch.text.data.to(device)

            y = batch.label.data.transpose(0, 1).to(device)
            input_data = (X,y,mask)
            loss = model(input_data)
            # y_hat = model(X)
            # loss = -model.crf(y_hat,y,reduction='mean',mask=mask)
            # print(y_hat.size())
            print(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
torch.save(model.state_dict(),save_dir)
#
# #
#