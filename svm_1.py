
import sys
import csv
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold

imdbpath = "./trainingdata/train_svm.csv"
#testpath = "./trainingdata/test.txt"

punct = ['.',',', '-']
stops = punct + stopwords.words('english')

#Percent of variance allowed: Ignore all features that are the same (whether 1 or 0) in p*100% of the samples.
p = .9

def read_data(path):
   
    data_file = open(path, 'r')
    reviews = list(csv.reader(data_file))
    sentences = []
    scores = []
    for line in reviews:
        sentence = line[0]
        score = line[1]
        sentences.append(sentence)
        scores.append(score)
    print(sentences)    
    return sentences, scores



def create_vocab(sentences):
    text = ' '.join(sentences)
    tokens = nltk.word_tokenize(text)

    tokens_filtered = [x.lower() for x in tokens if not x in stops]
    
    freq_dist = nltk.FreqDist(tokens_filtered)
    return freq_dist.keys()
    
def transform_sentence(sentence, vocab):
    tokens = [x.lower() for x in nltk.word_tokenize(sentence) if not x in stops]
    fdist = nltk.FreqDist(tokens)
    features = [fdist[x] for x in vocab]
    return features

def predict(sentence, vocab, clf):
    print(clf.predict([transform_sentence(sentence, vocab)]))


sentences, scores = read_data(imdbpath)
vocab = create_vocab(sentences)
X = [transform_sentence(x, vocab) for x in sentences]

#Features are boolean and thus Bernoulli RVs, so variance is given by p*(1-p)
#var = VarianceThreshold(threshold = (p * (1 - p)))
#X = var.fit_transform(X) #Not a good method of feature selection for this data. Variance is already sufficiently controlled by removing stopwords; also, words that vary little may still be significant.

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X, scores)

predict("It was fun!", vocab, clf)
predict("Horrible movie.", vocab, clf)
predict("It was only a small step up from their usual fiascos.", vocab, clf)
predict("I loved it only a little less than I love chocolate", vocab, clf)
predict("It's not my most favorite, but it's my least favorite.", vocab, clf)
