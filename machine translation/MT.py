import spacy
import torchtext
from torchtext.data import Field, BucketIterator, TabularDataset
en = spacy.load('en_core_web_sm')

def tokenize_en(sentence):
    return [tok.text for tok in en.tokenizer(sentence)]