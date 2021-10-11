import os
from sklearn.metrics import f1_score,confusion_matrix
import re
import pandas as pd
import csv
import numpy as np
import nltk
from match_pattern import *
from wrapper import *

nltk.download('punkt')

#sections = ["Diagnosis:","Past or Present History of Illness:","Social/Family History:","Physical or Laboratory Examination:","Medication/Disposition:","Other"]
#count_sections=[0,0,0,0,0,0,0]
datasets=["train","test","validation"]
txts = []
labels = []
predicts = []

all_sections = []
dif_sections = []
count_sections = []
input_file_name=[]

#Choose dataset
print("which dataset? 1.train, 2.test, 3.validation")
choose=int(input())
dataset=datasets[choose-1]

#Read data
root = "C:/Users/User/Desktop/數位醫學/HW1/Case Presentation 1 Data/"
path = root + "Train_Textual"
if(dataset=="test"):path = root + "Test_Intuitive"
elif(dataset=="validation"):path = root + "Validation"
files= os.listdir(path)
for file in files:
    input_file_name.append(file)
    position = path+'/'+ file
    if dataset!="validation":
        if file[0]=="Y":
            labels.append(1)
        else:
            labels.append(0)
    with open(position, "r") as f:
        data = f.read()
        find_pattern = re.compile(r'\n[A-Za-z ]+:', re.I)
        match_result = find_pattern.findall(data)
        all_sections.append(match_result)
        #txts.append(data.split(" "))
        txts.append(nltk.sent_tokenize(data))
print("data size:",len(txts))

#Count diffirent sections
for sections in all_sections:
    for section in sections:
        if section.lower() not in dif_sections:
            dif_sections.append(section.lower())
            count_sections.append(1)
        else:
            index=dif_sections.index(section.lower())
            count_sections[index]+=1
all_sections=[]
for i,section in enumerate(dif_sections):
    a=section.replace("\n","")
    b=a.replace(":","")
    all_sections.append(b)
#print(all_sections)
#print(count_sections)

#If else method
i=0
targets=["obese", "obesity", "overweight"]
keys = []
keywords = []

for data in txts:

    """find=False
    for words in data:
        for target in targets:
            if words.lower()==target:
                find=True
    if find==True:
        predicts.append(1)
    else:
        predicts.append(0)"""

    #find which sentence index contain keywords
    key = []
    keyword = []
    for i, sentence in enumerate(data):
        data[i] = sentence.replace('\n', ' ')

        for target in targets:
            if (sentence.find(target) != -1): 
                key.append(i)
                keyword.append(target)
                break
              
    keys.append(key)
    keywords.append(keyword)

#construct negex output in negex_output.txt
predicts = negexFormatting(root, txts, keys, keywords, labels)


#Calculate F1 score
if dataset!="validation":
    f1=f1_score(labels,predicts)   
    print("f1 score:",f1)
    tn, fp, fn, tp = confusion_matrix(labels, predicts).ravel()
    print("tn:",tn,"fp:", fp,"fn:", fn,"tp:", tp)

#Save output
output_file_name=dataset+".csv"
dict = {'Filename':input_file_name,'Obesity':predicts}
df = pd.DataFrame(dict) 
df.to_csv(output_file_name,index=None)