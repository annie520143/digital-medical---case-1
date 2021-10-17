import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

def removeStopwords(sentence):

    stops = nltk.corpus.stopwords.words('english')
    stops.append('mg')
    for i,word in enumerate(sentence):
        if word in stops:
            sentence[i] = ''
    return list(filter(lambda w: w != '', sentence))

#def normalize(embedding):

def lemmatize(sentence):

    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')

    wnl = WordNetLemmatizer()
    taggedSentence = nltk.pos_tag(sentence)
    for i,word in enumerate(sentence):
        originWord = wnl.lemmatize(word, wordnetPos(taggedSentence[i]) or wordnet.NOUN)
        sentence[i] = originWord
    return sentence

def wordnetPos(tags):
    
    tag = tags[1]
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

