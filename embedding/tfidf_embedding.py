import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,confusion_matrix
from sklearn.svm import SVC
import pandas as pd
import re

#Set your data root here!!!
root="/home/yychang/Desktop/Digital Medicine/Case Presentation 1 Data/"

#Choose dataset
print("section select? 1.yes, 2.no")
section=int(input())

txts=[]
train_txts=[]
test_txts=[]
validation_txts=[]
input_file_name=[]
labels=[]

#Set train folder
if section==1: path=root+"train_preprocessing/"
else:
    path=root+"Train_Textual/"

#Read train data
files= os.listdir(path)
for file in files:
    position = path + file
    if file[0]=="Y":
        labels.append(1)
    else:
        labels.append(0)
    with open(position, "r") as f:
        data = f.read()
        txts.append(str(data.split(" ")))
        train_txts.append(str(data.split(" ")))
print("data size:",len(txts))

#Set test folder
if section==1: path=root+"test_preprocessing/"
else:
    path=root+"Test_Intuitive/"

#Read test data
files= os.listdir(path)
for file in files:
    position = path + file
    with open(position, "r") as f:
        data = f.read()
        txts.append(str(data.split(" ")))
        test_txts.append(str(data.split(" ")))
print("data size:",len(txts))


#Set validation folder
if section==1: path=root+"validation_preprocessing/"
else:
    path=root+"Validation/"

#Read validation data
files= os.listdir(path)
for file in files:
    input_file_name.append(file)
    position = path + file
    with open(position, "r") as f:
        data = f.read()
        txts.append(str(data.split(" ")))
        validation_txts.append(str(data.split(" ")))
print("data size:",len(txts))

tfidf_vectorizer=TfidfVectorizer(stop_words="english",min_df=0.01)
tfidf_vectorizer.fit(txts)
train_tfidf = tfidf_vectorizer.transform(train_txts).toarray()
print ( train_tfidf )
print ( train_tfidf.shape )

#svm=SVC(kernel="linear",probability=True)
#svm.fit(train_tfidf,labels)
LR=LogisticRegression(random_state=0)
LR.fit(train_tfidf,labels)

print(len(validation_txts))
validation_tfidf = tfidf_vectorizer.transform(validation_txts).toarray()
print ( validation_tfidf )
print ( validation_tfidf.shape)

train_predicts=LR.predict(train_tfidf)
f1=f1_score(labels,train_predicts)   
print("f1 score:",f1)
tn, fp, fn, tp = confusion_matrix(labels, train_predicts).ravel()
print("tn:",tn,"fp:", fp,"fn:", fn,"tp:", tp)

#predicts=svm.predict(test_tfidf)
predicts=LR.predict(validation_tfidf)
print(predicts)

#Save output
output_file_name="predicts.csv"
dict = {'Filename':input_file_name,'Obesity':predicts}
df = pd.DataFrame(dict) 
df.to_csv(output_file_name,index=None)
