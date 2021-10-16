from numpy import random, zeros
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from nltk.stem.snowball import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle
import nltk
import os
import re
import sys

def preprocess_text(sen):
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sen)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    
    stops = nltk.corpus.stopwords.words('english')

    for word in sentence.split(' '):
        if word in stops:
            sentence = sentence.replace(' '+ word + ' ', ' ')

    return sentence.lower()

def word2Vec_embedding_model():

    txts = []
    datas = []
    sentences = []
    root = "C:/Users/User/Desktop/數位醫學/HW1/Case Presentation 1 Data/"
    paths, tokenPath = route(root)

    for path in paths:
        files = os.listdir(path)
        for file in files:
            with open(path+'/'+file, 'r') as f:
                data = preprocess_text(f.read())
                txts.append(data.split(" "))
                datas.append(data)

    word2vec_model = Word2Vec(txts, vector_size=500, window=3, min_count=1, workers=4)
    word2vec_model.save('model/word2vec.model')
    #print(word2vec_model.wv['depression'])
    #result = word2vec_model.wv.similarity("obese", "obesity")
    #print(result)

    #saveTokenizer(datas)
    #print(txt2Embedding(txts[0], word2vec_model))

def saveTokenizer(path, datas):
    token = Tokenizer()
    token.fit_on_texts(datas)
    with open(path, 'wb') as handle:
        pickle.dump(token, handle, protocol=pickle.HIGHEST_PROTOCOL)

#embedding matrix (per data, row: word, column: embedding vector)
def txt2Embedding(data, model):

    embeddings = []
    for word in data:
        embeddings.append(model.wv[word])
    return embeddings
        
def route(root):
    print("section select? 1.yes, 2.no")
    section=int(input())
    
    path = []
    
    if section==1: 
        path.append(root+"train_preprocessing/")
        path.append(root+"test_preprocessing/")
        path.append(root+"validation_preprocessing/")
        tokenPath = root + 'tokenizer/' + 'tokenizer_section.pickle'
    else:
        path.append(root+"Train_Textual/")
        path.append(root+"Test_Intuitive/")
        path.append(root+"Validation/")
        tokenPath = root + 'tokenizer/' +'tokenizer_no_section.pickle'
    return path, tokenPath

def main():

    """with open(path, 'rb') as handle:
        token = pickle.load(handle)

    seq = token.texts_to_sequences(inputData)
    seq = pad_sequences(seq)
    seq = token.sequences_to_texts(seq)"""

    root = "C:/Users/User/Desktop/數位醫學/HW1/Case Presentation 1 Data/"
    paths, tokenPath = route(root)

    print("which dataset?1.train 2.test 3.validation")
    dataset = int(input())
    path = paths[dataset-1]

    txts = []
    datas = []

    files = os.listdir(path)
    for file in files:
        with open(path+'/'+file, 'r') as f:
            data = preprocess_text(f.read())
            txts.append(data.split(" "))
            datas.append(data)

    model = Word2Vec.load('./model/word2vec.model')

    embeddings_all = []
    for data in txts:
        embeddings_all.append(txt2Embedding(data, model))

main()