
import sys
import glob, os
import numpy as np
import nltk
import csv
from collections import Counter
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as mp
from arrays import *  

imdbpath = "./trainingdata/train_file.txt"
testpath = "./trainingdata/test_file.txt"

prediction = []
actual = []
punct = ['.',',', '-']
stops = punct + stopwords.words('english')

#Percent of variance allowed: Ignore all features that are the same (whether 1 or 0) in p*100% of the samples.
p = .9

def read_data(path):
    data_file = open(path,'r')
    sentences = []
    scores = []
    for line in data_file:
        sentence, score = line.split("      ",1)
        sentences.append(sentence)
        scores.append(score.strip())
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
    answer = clf.predict([transform_sentence(sentence, vocab)])
    answer = answer.astype(int)
    prediction.append(answer[0])
    return answer[0]

    

sentences, scores = read_data(imdbpath)
vocab = create_vocab(sentences)
X = [transform_sentence(x, vocab) for x in sentences]
clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X, scores)


test_data = open(testpath,'r')
 
for line in test_data:
        sentence, score = line.split("       ",1)
        predict(sentence,vocab,clf)
        actual.append(int(score))


print(accuracy_score(actual,prediction))

prediction = []

print(" SVM ")

i=0
n=0
xaxis = ["BUSS","ENT","LIFE","SPORTS","TECH"]
yaxis = []
width = 1/1.5
ind = [0, 1, 2, 3, 4]

print("DNA ")
os.chdir("../news_output/DNA")
for file in glob.glob("*.txt"):
    print(file)
    actual = []
    with open(file,'r') as txtinput:
        with open(file.replace('.txt','')+"DnaSvm.csv",'w') as txtoutput:
          writer = csv.writer(txtoutput, delimiter=',')
          j=0
          for line in txtinput:
             actual.append(DNA[n][j])   
             x = line+","+str(predict(line,vocab,clf))+","+str(DNA[n][j])
             print(x)
             print(str(n)+str(j))
             writer.writerow(x.split(","))
             j+=1
          txtoutput.close()
    print(actual)
    print(prediction)
    n+=1
    print(accuracy_score(actual,prediction))       
    final= dict(Counter(prediction))
    ans = (final[1])/(final[1]+final[-1])*100
    print ('Objective Percentage\t'+str(ans))
    yaxis.append(ans)
    prediction = []
    i=i+1

mp.bar(ind, yaxis, width, color="white",align ='center')
mp.xticks(ind,xaxis)
mp.ylabel('Objective Percentage')
mp.title('DNA India')
for a,b in zip(ind, yaxis):
    mp.text(a-0.25, b-2.4, str(round(b,2)),size=14)
font = {'family' : 'normal',
        'size'   : 15}
mp.rc('font', **font)     
mp.show()  
    
 
print("THE TIMES OF INDIA")

yaxis = []
i=0
n=0
os.chdir("../TOI")
for file in glob.glob("*.txt"):
    print(file)
    actual = []
    with open(file,'r') as txtinput:
        with open(file.replace('.txt','')+"ToiSvm.csv",'w') as txtoutput:
            writer = csv.writer(txtoutput, delimiter=',')
            j=0
            for line in txtinput:
               actual.append(TOI[n][j])  
               x = line+","+str(predict(line,vocab,clf))+","+str(TOI[n][j])
               writer.writerow(x.split(","))
               j+=1
            txtoutput.close()
    #print(actual)
    #print(prediction)     
    print(accuracy_score(actual,prediction))        
    final= dict(Counter(prediction))
    print(final)
    ans = (final[1])/(final[1]+final[-1])*100
    print ('Objective Percentage\t'+str(ans))
    yaxis.append(ans)     
    prediction = []
    i=i+1
    n+=1
mp.bar(ind, yaxis, width, color="white",align ='center')
mp.xticks(ind,xaxis)
mp.ylabel('Objective Percentage')
mp.title('THE TIMES OF INDIA')
for a,b in zip(ind, yaxis):
    mp.text(a-0.25, b-2.4, str(round(b,2)),size =14)
font = {'family' : 'normal',
        'size'   : 15}

mp.rc('font', **font)    
mp.show()


