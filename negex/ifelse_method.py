import os
from sklearn.metrics import f1_score,confusion_matrix
import re
import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from match_pattern import *
from wrapper import *
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('wordnet')


#Set your data root here!!!
root="/home/gdwang/Downloads/Case Presentation 1 Data/"

datasets=["train","test","validation"]
method="if else"
txts = []
labels = []
predicts = []
input_file_name=[]

#Choose dataset and method
print("which dataset? 1.train, 2.test, 3.validation")
choose=int(input())
dataset=datasets[choose-1]
print("section select? 1.yes, 2.no")
section=int(input())
print("use negex? 1.yes, 2.no")
negex=int(input())

if section == 1:
    if dataset == 'train':
        path=root+"train_preprocessing/"
    elif dataset == 'test':
        path=root+"test_preprocessing/"
    elif dataset == 'validation':
        path=root+"validation_preprocessing/"
else:
    if dataset == 'train':
        path=root+"Train_Textual/"
    elif dataset == 'test':
        path=root+"Test_Intuitive/"
    elif dataset == 'validation':
        path=root+"Validation/"


#Read data
files= os.listdir(path)
for file in files:
    input_file_name.append(file)
    position = path + file
    if dataset!="validation":
        if file[0]=="Y":
            labels.append(1)
        else:
            labels.append(0)
    with open(position, "r") as f:
        data = f.read()
        if negex==1:txts.append(nltk.sent_tokenize(data))
        else: txts.append(data.split(" "))
print("data size:",len(txts))

#If else method
if method=="if else":
    targets=["obese", "obesity","overweight","fat"]
    if negex==1:
        i=0
        keys = []
        keywords = []
        for data in txts:
            #find which sentence index contain keywords
            key = []
            keyword = []
            break_flag = 0
            for i, sentence in enumerate(data):
                data[i] = sentence.replace('\n', ' ')

                for target in targets:
                    if (sentence.find(target) != -1): 
                        key.append(i)
                        keyword.append(target)
                        break

            keys.append(key)
            keywords.append(keyword)
        #Construct negex output in negex_output.txt
        predicts = negexFormatting(root, txts, keys, keywords, labels)
    else:
        for data in txts:
            find=False
            for words in data:
                for target in targets:
                    if words.lower()==target:
                        find=True
            if find==True:
                predicts.append(1)
            else:
                predicts.append(0)


#Paper method
#if method=="paper":
    
#Calculate F1 score
if dataset!="validation":
    f1=f1_score(labels,predicts)   
    print("f1 score:",f1)
    tn, fp, fn, tp = confusion_matrix(labels, predicts).ravel()
    print("tn:",tn,"fp:", fp,"fn:", fn,"tp:", tp)

#Save output
output_file_name="ifelse_ans.csv"
dict = {'Filename':input_file_name,'Obesity':predicts}
df = pd.DataFrame(dict) 
df.to_csv(output_file_name,index=None)