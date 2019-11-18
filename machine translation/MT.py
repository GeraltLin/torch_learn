from torchtext.data import Field, BucketIterator
from torchtext.datasets import TranslationDataset

Lang1 = Field(init_token='<sos>', eos_token='<eos>')
Lang2 = Field(init_token='<sos>', eos_token='<eos>')

train = TranslationDataset(path='../Datasets/MT_data/',
                           exts=('eng-fra.train.en', 'eng-fra.train.fr'),
                        fields=[('Lang1', Lang1),('Lang2', Lang2)])

train_iter, val_iter, test_iter = BucketIterator.splits((train, train, train),
                                                            batch_size=1, repeat=False)
Lang1.build_vocab(train)
Lang2.build_vocab(train)

for i, train_batch in enumerate(train_iter):
    print('Lang1  : \n', [Lang1.vocab.itos[x] for x in train_batch.Lang1.data])
    print('Lang2 : \n', [Lang2.vocab.itos[x] for x in train_batch.Lang2.data])