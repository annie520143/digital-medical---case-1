# digital-medical---case-1
### Introduction
2008年的i2b2 obesity chanllege，旨意在於從電子病歷中，分辨此患者有無肥胖的症狀。此repository嘗試不同的embedding及model，試圖完成此項作業。
 * Embedding : word2vec、tfidf
 * Model ： Logistic Regression、SVM、HiSAN
 * Other method : if/else + negex

### File Description
* `/embedding/`
  * `tfidf_embedding.py`: Run **tfidf** embedding and **Logistic Regression** and output F1 score
  * `word2vec_embedding.py` : **word2vec** embedding and output to .npy form 
* `/model/` : pretrained model for word2vec embedding 
* `/negex/` : **negex** + **if/else** method
* `/preprocessing/` : some preprocessing functions
* `HiSAN.py` : **word2vec** + **HiSAN**
* `SVM.py` : **word2vec** + **SVM**
* `section_select.py` : select section with Jaccard Similarity, and output the selected data in txt format

### Requirements
see requirements.txt

### Implementation
> place data(Case Presentation 1 Data folder) in project root directory
> change root in file to the location of Case Presentation 1 Data folder

1. HiSAN : our DL model
 ```
  python HiSAN.py --lemma n --stop n
 ```
 * parameter
   * lemma : Do you want to lemmatize input data? n for no, y for yes
   * stop :  Do you want to remove stopwords for input data? n for no, y for yes
  
2. SVM : our ML model, is also the best performance model
 ```
  python SVM.py --lemma n --stop n
 ```
 
 3. if-else method
 ```
 cd negex
 python ifelse.py
 ```

### Reference
* [A text mining approach to the prediction of disease status from clinical discharge summaries](https://pubmed.ncbi.nlm.nih.gov/19390098/)
* [A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries](https://www.sciencedirect.com/science/article/pii/S1532046401910299)
* [Classifying cancer pathology reports with hierarchical self-attention networks](https://www.sciencedirect.com/science/article/pii/S0933365719303562)
* [Limitations of Transformers on Clinical Text Classification](https://pubmed.ncbi.nlm.nih.gov/33635801/)
