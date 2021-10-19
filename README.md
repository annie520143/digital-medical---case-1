# digital-medical---case-1
### Introduction
2008年的i2b2 obesity chanllege，旨意在於從電子病歷中，分辨此患者有無肥胖的症狀。此repository嘗試不同的embedding及model，試圖完成此項作業。
 * Embedding : word2vec、tfidf
 * model ： Logistic Regression、SVM、HiSAN
 * other method : if/else + negex

### File Description
* `/embedding/`
  * `tfidf_embedding.py`: Run **tfidf** embedding and **Logistic Regression** and output F1 score
  * `word2vec_embedding.py` : **word2vec** embedding and output to .npy form 
* `/model/` : pretrained model for word2vec embedding 
* `/negex/` : **negex** + **if/else** method
* `/preprocessing/` : 
* `HiSAN.py` : **word2vec** + **HiSAN**
* `SVM.py` : **word2vec** + **SVM**
* `section_select.py` : select section with Jaccard Similarity

### Requirements
### Implementation
### Result
### Reference
