import os
import re
import numpy as np
import tensorflow as tf
import sys
import time
from sklearn.metrics import f1_score
import random

import json
import string
import pandas as pd
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
import argparse
from numpy import random, vstack, save, zeros
from gensim.models import Word2Vec
# import logging
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm, tree
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


word2vec_min_count = 5


def clearup(document):
    # document = document.translate(string.punctuation)
    numbers = re.search('[0-9]+', document)
    document = re.sub('\(\d+.\d+\)|\d-\d|\d', '', document) \
        .replace('.', '').replace(',', '').replace(',', '').replace(':', '').replace('~', '') \
        .replace('!', '').replace('@', '').replace('#', '').replace('$', '').replace('/', '') \
        .replace('%', '').replace('(', '').replace(')', '').replace('?', '') \
        .replace('—', '').replace(';', '').replace('&quot', '').replace('&lt', '') \
        .replace('^', '').replace('"', '').replace('{', '').replace('}', '').replace('\\', '').replace('+', '') \
        .replace('&gt', '').replace('&apos', '').replace('*', '').strip().lower().split()
    # return re.sub('[l]+', ' ', str(document)).strip()
    return document



def size(alist):
    return len(alist)

def prep_data_HiSAN(documents):
    t = Tokenizer()
    docs = list(filter(None, documents))
    print("Size of the documents in prep_data {}".format(len(documents)))
    t.fit_on_texts(docs)
    return t.word_index

def word2Vec(docs,word_index):

    # train word2vec
    sentences = docs
    model = Word2Vec(sentences, min_count=word2vec_min_count, size=300, workers=4, iter=10)
    # save all word embeddings to matrix
    vocab_size = len(model.wv.vocab)
    print(vocab_size)
    vocab = zeros((vocab_size + 1, 300))
    word2idx = {}
    for i,(key,val) in enumerate(model.wv.vocab.items()):
        if key in word_index:
            word2idx[key] = i+1
            vocab[i+1, :] = model[key]

    # add additional word embedding for unknown words
    unk = len(vocab)
    vocab = vstack((vocab, random.rand(1, 300) - 0.5))

    # normalize embeddings
    vocab -= vocab.mean()
    vocab /= (vocab.std() * 2.5)
    vocab[0, :] = 0
    max_len = 20000

    # convert words to indices
    text_idx = zeros((len(sentences), max_len))
    for i, sent in enumerate(sentences):
        idx = [word2idx[word] if word in word2idx else unk for word in sent][:max_len]
        l = len(idx)
        text_idx[i, :l] = idx

    # save data
    return text_idx,word2idx,vocab

root="/home/gdwang/Downloads/Case Presentation 1 Data/"
datasets=["train","test","validation"]
methods=["if else","paper","our"]
padded_label = []
padded_documents = []
input_file_name=[]

path_train=root+"Train_Textual/"
path_test=root+"Test_Intuitive/"
path_valid=root+"Validation/"
paths = [path_train, path_test, path_valid]

for path in paths:
    files= os.listdir(path)
    for file in files:
        input_file_name.append(file)
        position = path + file
        if path!= path_valid :
            if file[0]=="Y":
                padded_label.append(1)
            else:
                padded_label.append(0)
        else:
            padded_label.append(0)
        with open(position, "r") as f:
            data = f.read()
            temp = re.split("[;.:] ", data)
            length = len(temp)
            i = 0
            while i < length:
                temp[i] = temp[i].replace("\n", " ")
                if len(temp[i]) < 20:
                    temp.remove(temp[i])
                    i = i - 1
                    length = length -1
                i = i + 1
            temp = ".".join(temp)
            doc = clearup(temp)
            padded_documents.append(doc)



word_index = prep_data_HiSAN(padded_documents)
processed_padded_documents, word2idx, vocab = word2Vec(padded_documents,word_index)

# print(vocab)
# print(type(vocab))
# print(len(vocab[0]))
# print(len(vocab))
train_x = processed_padded_documents[:400]
test_x = processed_padded_documents[400:800]
val_x = processed_padded_documents[800:]
y_train = padded_label[:400]
y_test = padded_label[400:800]
y_val = padded_label[800:]

# print(len(val_x[0]))
# print(len(val_x[1]))

ans_file_name = input_file_name[800:].copy()

SVM = svm.SVC(C=1.0, kernel='linear',  gamma='auto')


SVM.fit(train_x,y_train)
predictions_SVM = SVM.predict(test_x)
print(predictions_SVM)
print("SVM Accuracy Score -> ",accuracy_score(predictions_SVM, y_test)*100)

# clf = RandomForestClassifier(n_estimators=200)
# clf.fit(train_x, y_train)
# predictions_clf = clf.predict(test_x)
# print("clf Accuracy Score -> ",accuracy_score(predictions_clf, y_test)*100)

final_ans = SVM.predict(val_x)
print(final_ans)
output_file_name="ans3.csv"
dict = {'Filename':ans_file_name,'Obesity':final_ans}
df = pd.DataFrame(dict) 
df.to_csv(output_file_name,index=None)

