from torchtext import data
import pandas as pd


# from torchtext.vocab import Vectors
# from torch.nn import init
#
# tokenize = lambda x: x.split()
# # fix_length指定了每条文本的长度，截断补长
# TEXT = data.Field(sequential=True, fix_length=200)
# LABEL = data.Field(sequential=False, use_vocab=False)


# train_data = pd.read_csv('test.index',header=None,usecols=[1])
# train_data.to_csv('test.csv',header=['text'],index=None)
def tokenizer(text):
    return list(text)


TEXT = data.Field(sequential=True, tokenize=tokenizer)

tmp_data = data.TabularDataset(path='test.csv', format='csv', fields=[('text', TEXT)])
# print(tmp_data)
# train_iterator = data.BucketIterator.splits(
#     tmp_data,
#     batch_size=64,
#     sort_key=lambda x: len(x.text),
#     device='cpu')
# TEXT.build_vocab(tmp_data)
# for i in tmp_data:
#
#     print(i.text)
print(tmp_data[0])
print(tmp_data[0].__dict__.keys())
print(tmp_data[0].text[:100])

# -----------------------------------------------
# <torchtext.data.example.Example object at 0x7f994eb505c0>
# dict_keys(['text'])
# ['投', '资', '者', '对', '城', '投', '债', '券', '风', '险', '表', '现', '出', '的', '恐', '慌']
# -----------------------------------------------
TEXT.build_vocab(tmp_data)
vocab = TEXT.vocab
print(len(vocab))
print(vocab.stoi)
print(vocab.itos)

from torchtext.data import Iterator, BucketIterator

#
train_iter, val_iter = BucketIterator.splits((tmp_data, tmp_data),
                                             # 我们把Iterator希望抽取的Dataset传递进去
                                             batch_sizes=(25, 25),
                                             device=-1,
                                             # 如果要用GPU，这里指定GPU的编号
                                             sort_key=lambda x: len(x.text),
                                             # BucketIterator 依据什么对数据分组
                                             sort_within_batch=False,
                                             repeat=False)
# repeat设置为False，因为我们想要包装这个迭代器层。
test_iter = Iterator(tmp_data, batch_size=1,
                     device=-1,
                     sort=False,
                     sort_within_batch=False,
                     repeat=False)

for data_batch in train_iter:
    print('--------------')
    x = data_batch.text
    print(x.size())

