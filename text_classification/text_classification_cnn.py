from torchtext import data
from torchtext.data import Iterator, BucketIterator
import torch
import jieba
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch import nn
import re
import torch.nn.functional as F



def tokenizer(text): # create a tokenizer function
    regex = re.compile(r'[^\u4e00-\u9fa5aA-Za-z0-9]')
    text = regex.sub(' ', text)
    return [word for word in jieba.cut(text) if word.strip()]


TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=False, fix_length=50)
LABEL = data.Field(sequential=False, use_vocab=False)
data_path = '../Datasets/text_classification/ch_classifivation.csv'
full_dataset = pd.read_csv(data_path,usecols = ['text','label'])
# full_dataset.to_csv('ch_classifivation.csv',index=None)
x,y = full_dataset['text'],full_dataset['label']
# 测试集为30%，训练集为70%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
train_data = (x_train,y_train)
valid_data = (x_test,y_test)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class MyDataset(data.Dataset):
    def __init__(self, csv_data, text_field, label_field, test=False, **kwargs):
        fields = [('text',text_field),('label',label_field)]
        examples = []
        x_data,y_data = csv_data
        if test:
            # 如果为测试集，则不加载label
            for text in tqdm(x_data):
                examples.append(data.Example.fromlist([text, None], fields))
        else:
            for text, label in tqdm(zip(x_data, y_data)):
                # Example: Defines a single training or test example.Stores each column of the example as an attribute.
                examples.append(data.Example.fromlist([text, label], fields))
            # 之前是一些预处理操作，此处调用super初始化父类，构造自定义的Dataset类。
        super(MyDataset, self).__init__(examples, fields, **kwargs)




def data_iter(train_data, valid_data, TEXT, LABEL):
    train = MyDataset(train_data, text_field=TEXT, label_field=LABEL, test=False)
    valid = MyDataset(valid_data, text_field=TEXT, label_field=LABEL, test=False)

    TEXT.build_vocab(train)
    train_iter, val_iter = BucketIterator.splits(
        (train, valid),  # 构建数据集所需的数据集
        batch_sizes=(100, 100),
        # 如果使用gpu，此处将-1更换为GPU的编号
        device=device,
        # the BucketIterator needs to be told what function it should use to group the data.
        sort_key=lambda x: len(x.text),
        sort_within_batch=False,
        repeat=False
    )
    return train_iter, val_iter

#test_iter = Iterator(test, batch_size=8, device=-1, sort=False, sort_within_batch=False, repeat=False)

train_iter, val_iter = data_iter(train_data, valid_data, TEXT, LABEL)
# for i in range(5):
#     for batch in train_iter:
#         # print(batch.text)
#         print(batch.label.size())
    # print(" ".join([TEXT.vocab.itos[i] for i in batch.label[:, 0].data]))

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d, self).__init__()
    def forward(self, x):
        return F.max_pool1d(x, kernel_size=x.shape[2])


# x shape: (batch_size, channel, seq_len)
# return shape: (batch_size, channel, 1)



class CNN(nn.Module):

    def __init__(self,kernel_size,num_channels):
        super(CNN, self).__init__()
        self.word_embeddings = nn.Embedding(len(TEXT.vocab), 200)  # embedding之后的shape: torch.Size([200, 8, 300])
        # 若使用预训练的词向量，需在此处指定预训练的权重
        # embedding.weight.data.copy_(weight_matrix)
        self.lstm = nn.LSTM(input_size=200, hidden_size=128, num_layers=1,batch_first = True)  # torch.Size([200, 8, 128])
        self.decoder = nn.Linear(sum(num_channels), 2)
        self.global_max_pool = GlobalMaxPool1d()
        self.dropout = nn.Dropout(p = 0.1)
        self.convs = nn.ModuleList()

        for c,k in zip(num_channels,kernel_size):
            self.convs.append(nn.Conv1d(in_channels = 200, out_channels = c, kernel_size = k))


    def forward(self, sentence):
        # sentence: batch,seq_len
        # print(sentence.size())
        # embeds:  batch,seq_len,input_size
        embeds = self.word_embeddings(sentence)
        inputs = embeds.permute(0,2,1)
        # inputs:  batch,input_size,seq_len
        encoding = torch.cat([self.global_max_pool(F.relu(conv(inputs))).squeeze(-1) for conv in self.convs], dim=1)


        y = self.decoder(self.dropout(encoding))

        return y
#
epochs = 1
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
kernel_sizes, nums_channels = [3, 4, 5], [100, 100,100]
model = CNN(kernel_sizes, nums_channels)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

cross_entropy_loss = nn.CrossEntropyLoss()
num_batches = len(train_iter)

def evaluate_accuracy(data_iter, model):
    model.eval()
    acc_sum, n = 0.0, 0
    for i, batch in enumerate(data_iter):
        X = batch.text.data.transpose(0, 1).to(device)
        y = batch.label.data.to(device)
        acc_sum += (model(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    model.train()
    return acc_sum / n

for epoch in range(epochs):
    train_l_batch, train_acc_batch, n = 0.0, 0.0, 0

    for i, batch in enumerate(train_iter):
        model.train()
        X = batch.text.data.transpose(0, 1).to(device)
        # X = batch.text.data.to(device)
        # print(X.size())
        y = batch.label.data.to(device)
        y_hat = model(X)


        loss = cross_entropy_loss(y_hat,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_l_batch = loss.item()
        train_acc_batch = (y_hat.argmax(dim=1) == y).sum().item() / y.shape[0]
        if i % 10 == 0:
            print('epoch %d,%d /%d, loss %.4f, train acc %.3f'
                  % (epoch + 1, i, num_batches, train_l_batch, train_acc_batch,
                     ))
    evel_acc = evaluate_accuracy(val_iter,model)
    print('epoch %d, evel_acc %.3f'
          % (epoch + 1, evel_acc,
             ))


def predict_rnn_pytorch(sentence, model, device,char_to_idx):
    sentence = tokenizer(sentence)
    output = [char_to_idx[word] for word in sentence]  # output会记录prefix加上输出

    X = torch.tensor(output, device=device)
    # print(X.size())
    X = X.view(1,-1)

    predict = model(X).argmax(dim=1)

    return predict

predict = predict_rnn_pytorch('就是后悔买了1.2的爬坡有时停车再走油门小了没力气还有就是夏天开空调力气明显不够，',model,device, TEXT.vocab.stoi)
print(predict.item())
print(TEXT.vocab.stoi)