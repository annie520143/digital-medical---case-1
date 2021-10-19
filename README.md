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
* `/preprocessing/` : 
* `HiSAN.py` : **word2vec** + **HiSAN**
* `SVM.py` : **word2vec** + **SVM**
* `section_select.py` : select section with Jaccard Similarity, and output the selected data in txt format

### Requirements
### Implementation
### Result
### Reference
* [A text mining approach to the prediction of disease status from clinical discharge summaries](https://pubmed.ncbi.nlm.nih.gov/19390098/)
* [A Simple Algorithm for Identifying Negated Findings and Diseases in Discharge Summaries](https://www.sciencedirect.com/science/article/pii/S1532046401910299)
* [Classifying cancer pathology reports with hierarchical self-attention networks](https://www.sciencedirect.com/science/article/pii/S0933365719303562)
