
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torch.optim as optim
import torch.autograd as autograd
import torchtext.vocab as torchvocab
from torch.autograd import Variable
import tqdm
import os
import time
import re
import pandas as pd
import string
import gensim
import time
import random
import snowballstemmer
import collections
from collections import Counter
from nltk.corpus import stopwords
from itertools import chain
from sklearn.metrics import accuracy_score

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors


# In[2]:


def clean_text(text):
    
    ## Remove puncuation
    text = text.translate(string.punctuation)
    
    ## Convert words to lower case and split them
    text = text.lower().split()
    
    # Remove stop words
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops and len(w) >= 3]
    
    text = " ".join(text)
    ## Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    ## Stemming
    text = text.split()
    stemmer = snowballstemmer.stemmer('english')
    stemmed_words = [stemmer.stemWord(word) for word in text]
    text = " ".join(stemmed_words)
    print(text)
    return text


# In[3]:


def readIMDB(path, seg='train'):
    pos_or_neg = ['pos', 'neg']
    data = []
    for label in pos_or_neg:
        files = os.listdir(os.path.join(path, seg, label))
        for file in files:
            with open(os.path.join(path, seg, label, file), 'r', encoding='utf8') as rf:
                review = rf.read().replace('\n', '')
                if label == 'pos':
                    data.append([review, 1])
                elif label == 'neg':
                    data.append([review, 0])
    return data


# In[3]:


root = './aclImdb'
train_data = readIMDB(root)
test_data = readIMDB(root, 'test')


# In[4]:


def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]

train_tokenized = []
test_tokenized = []
for review, score in train_data:
    train_tokenized.append(tokenizer(review))
for review, score in test_data:
    test_tokenized.append(tokenizer(review))


# In[5]:



vocab = set(chain(*train_tokenized))
vocab_size = len(vocab)


# In[6]:


# 输入文件
glove_file = datapath('/home/cuixw/lkx/glove.6B.100d.txt')
# 输出文件
tmp_file = get_tmpfile("/home/cuixw/lkx/wv.6B.100d.txt")

# call glove2word2vec script
# default way (through CLI): python -m gensim.scripts.glove2word2vec --input <glove_file> --output <w2v_file>

# 开始转换
from gensim.scripts.glove2word2vec import glove2word2vec
glove2word2vec(glove_file, tmp_file)

# 加载转化后的文件
wvmodel = KeyedVectors.load_word2vec_format(tmp_file)


# In[7]:


word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
word_to_idx['<unk>'] = 0
idx_to_word = {i+1: word for i, word in enumerate(vocab)}
idx_to_word[0] = '<unk>'


# In[8]:


def encode_samples(tokenized_samples, vocab):
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


# In[9]:


def pad_samples(features, maxlen=500, PAD=0):
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while(len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features


# In[10]:


train_features = torch.tensor(pad_samples(encode_samples(train_tokenized, vocab)))
train_labels = torch.tensor([score for _, score in train_data])
test_features = torch.tensor(pad_samples(encode_samples(test_tokenized, vocab)))
test_labels = torch.tensor([score for _, score in test_data])


# In[13]:


class SentimentNet(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, use_gpu, **kwargs):
        super(SentimentNet, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.use_gpu = use_gpu
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                               num_layers=num_layers, bidirectional=self.bidirectional,
                               dropout=0)
        if self.bidirectional:
            self.decoder = nn.Linear(num_hiddens * 4, labels)
        else:
            self.decoder = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        states, hidden = self.encoder(embeddings.permute([1, 0, 2]))
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.decoder(encoding)
        return outputs


# In[16]:


num_epochs = 5
embed_size = 100
num_hiddens = 100
num_layers = 2
bidirectional = True
batch_size = 64
labels = 2
lr = 0.8
device = torch.device('cuda:0')
use_gpu = True

weight = torch.zeros(vocab_size+1, embed_size)

for i in range(len(wvmodel.index2word)):
    try:
        index = word_to_idx[wvmodel.index2word[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(wvmodel.get_vector(
        idx_to_word[word_to_idx[wvmodel.index2word[i]]]))


# In[17]:



net = SentimentNet(vocab_size=(vocab_size+1), embed_size=embed_size,
                   num_hiddens=num_hiddens, num_layers=num_layers,
                   bidirectional=bidirectional, weight=weight,
                   labels=labels, use_gpu=use_gpu)
net.to(device)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)


# In[18]:


train_set = torch.utils.data.TensorDataset(train_features, train_labels)
test_set = torch.utils.data.TensorDataset(test_features, test_labels)

train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=True)
test_iter = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                        shuffle=False)


# In[20]:


num_epochs = 20


# In[ ]:


for epoch in range(num_epochs):
    start = time.time()
    train_loss, test_losses = 0, 0
    train_acc, test_acc = 0, 0
    n, m = 0, 0
    for feature, label in train_iter:
        n += 1
        net.zero_grad()
        feature = Variable(feature.cuda())
        label = Variable(label.cuda())
        score = net(feature)
        loss = loss_function(score, label)
        loss.backward()
        optimizer.step()
        train_acc += accuracy_score(torch.argmax(score.cpu().data,
                                                 dim=1), label.cpu())
        train_loss += loss
    with torch.no_grad():
        for test_feature, test_label in test_iter:
            m += 1
            test_feature = test_feature.cuda()
            test_label = test_label.cuda()
            test_score = net(test_feature)
            test_loss = loss_function(test_score, test_label)
            test_acc += accuracy_score(torch.argmax(test_score.cpu().data,
                                                    dim=1), test_label.cpu())
            test_losses += test_loss
    end = time.time()
    runtime = end - start
    print('epoch: %d, train loss: %.4f, train acc: %.2f, test loss: %.4f, test acc: %.2f, time: %.2f' %
          (epoch, train_loss.data / n, train_acc / n, test_losses.data / m, test_acc / m, runtime))


# In[ ]:




