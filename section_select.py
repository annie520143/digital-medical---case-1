import os
from sklearn.metrics import f1_score,confusion_matrix
import re
import pandas as pd
import csv
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))

    s1, s2 = add_space(s1), add_space(s2)
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    numerator = np.sum(np.min(vectors, axis=0))
    denominator = np.sum(np.max(vectors, axis=0))
    return 1.0 * numerator / denominator

#Set your data root here!!!
root="/home/yychang/Desktop/Digital Medicine/Case Presentation 1 Data/"

targ_sections = ["diagnosis","past history of illness","present history of illness","physical examination","physical exam upon transfer to the gms service","disposition"]#"medication disposition" #laboratory
threshold = 0.63
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

#Set folder
preprocessing_path=root+dataset+"_preprocessing/"
path=root+"Train_Textual/"
if(dataset=="test"):path=root+"Test_Intuitive/"
elif(dataset=="validation"):path=root+"Validation/"

#Read data
files= os.listdir(path)
for file in files:
    input_file_name.append(file)
    position = path+file
    with open(position, "r") as f:
        data = f.read()
        find_pattern = re.compile(r'[\n][A-Z ]+:')
        match_result = find_pattern.findall(data)
        all_sections.append(match_result)
        txts.append(data.lower())#.lower()
print("data size:",len(txts))

#Count total sections
count=0
for sections in all_sections:
        count+=len(sections)
print("all section size",count)

#Lower sections
for i in range(len(all_sections)):
    for j in range(len(all_sections[i])):
        all_sections[i][j]=all_sections[i][j].lower()
        all_sections[i][j]=all_sections[i][j].replace("\n","")

#Count diffirent sections
for sections in all_sections:
    for section in sections:
        if section not in dif_sections:
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
print("different section size",len(all_sections))
#print(all_sections)

#Count similarity
final_sections=[]
for pred_section in all_sections:
    for targ_section in targ_sections:
        score=jaccard_similarity(pred_section,targ_section)
        if score>threshold and pred_section!="admission pe" and pred_section!="admission ekg" and pred_section!="last pertinent tests at time of discharge" and pred_section!="transfer medications to the gms service":
            final_sections.append(pred_section)
print("final sections size",len(final_sections))
print(final_sections)

#Paper method
print("#####################")
print("####preprocessing####")
print("#####################")
if not os.path.isdir(preprocessing_path):
    os.mkdir(preprocessing_path)
for num,txt in enumerate(txts):
    txt_file_name=input_file_name[num]
    f=open(preprocessing_path+txt_file_name,"w")
    find_pattern = re.compile(r'[\n ][A-Za-z ]+:', re.I)
    match_result = find_pattern.findall(txt)
    for i in range(len(match_result)):
        match_result[i]=match_result[i].replace("\n","")
        match_result[i]=match_result[i].replace(":","")
    for section in final_sections:
        if section in match_result:
            txt_index=txt.find(section)
            section_txt=""
            section_index=match_result.index(section)
            txt_next_index=txt.index(match_result[section_index+1])
            for i in range(txt_index+len(section)+2,txt_next_index):
                section_txt+=txt[i]
            section_txt=section_txt.replace("\n"," ")
            f.write(section_txt)
            f.write("\n")
    f.close()

