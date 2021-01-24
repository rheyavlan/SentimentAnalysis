from nltk.corpus import stopwords
from collections import Counter
import csv
import re
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as mp
from sklearn import metrics
import string
from arrays import *  
print("NAIVE BAYES\n")

after_stop_word = open("naive_bayes_preprocess.txt",'w')
stop_words = set(stopwords.words('english'))

punctuations = list(string.punctuation)

# Read in the training data.
with open("train_set_subjective.csv", 'r') as file:
  reviews = list(csv.reader(file))

def stop_word_removal(text):
   tokenize_text = word_tokenize(text)
   words = [w for w in tokenize_text if not w in stop_words]
   words_remove_punctuations = [w for w in words if not w in punctuations]
   after_stop_word.write(" ".join(words_remove_punctuations))
   return words_remove_punctuations

def get_text(reviews, score):
  # Join together the text in the reviews for a particular tone.
  # We lowercase to avoid "Not" and "not" being seen as different words, for example.
  return " ".join([r[0].lower() for r in reviews if r[1] == str(score)])

def count_text(words):
  # Count up the occurence of each word.
  return Counter(words)

def get_y_count(score):
  # Compute the count of each classification occuring in the data.
  return len([r for r in reviews if r[1] == str(score)])

def make_decision(text, make_class_prediction):
    # Compute the subjective and objective probabilities.
    subjective_prediction = make_class_prediction(text, subjective_counts, prob_subjective, subjective_review_count)
    objective_prediction = make_class_prediction(text, objective_counts, prob_objective, objective_review_count)

    # We assign a classification based on which probability is greater.
    if subjective_prediction > objective_prediction:
      return -1
    return 1

def make_class_prediction(text, counts, class_prob, class_count):
  prediction = 1
  text_counts = count_text(stop_word_removal(text))
  for word in text_counts:
      # For every word in the text, we get the number of times that word occured in the reviews for a given class, add 1 to smooth the value, and divide by the total number of words in the class (plus the class_count to also smooth the denominator).
      # Smoothing ensures that we don't multiply the prediction by 0 if the word didn't exist in the training data.
      # We also smooth the denominator counts to keep things even.
      prediction *=  text_counts.get(word) * ((counts.get(word, 0) + 1) / (sum(counts.values()) + class_count))
  # Now we multiply by the probability of the class existing in the documents.
  return prediction * class_prob


subjective_text = get_text(reviews,-1)
objective_text = get_text(reviews, 1)

# Generate word counts for objective tone.
objective_counts = count_text(stop_word_removal(objective_text))

# Generate word counts for subjective tone.
subjective_counts = count_text(stop_word_removal(subjective_text))

# We need these counts to use for smoothing when computing the prediction.
objective_review_count = get_y_count(1)
subjective_review_count = get_y_count(-1)

# These are the class probabilities (we saw them in the formula as P(y)).
prob_objective = objective_review_count / len(reviews)
prob_subjective = subjective_review_count / len(reviews)

# As you can see, we can now generate probabilities for which class a given review is part of.
# The probabilities themselves aren't very useful -- we make our classification decision based on which value is greater.

with open("test_data_subjective.csv", 'r') as file:
    test = list(csv.reader(file))

predictions = [make_decision(r[0], make_class_prediction) for r in test]
actual = [int(r[1]) for r in test]

#Make a new file with predictions
with open('test_data_subjective.csv','r') as csvinput:
    with open('output.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput, lineterminator='\n')
        reader = csv.reader(csvinput)

        all = []
        row = ["Sample", "Actual","Prediction"]
        all.append(row)
        row = next(reader)
        row.append(predictions[0])
        all.append(row)
        n=0
        for row in reader:
            row.append(predictions[n])
            n+=1
            all.append(row)

        writer.writerows(all)





# Generate the roc curve using scikits-learn.
fpr, tpr, thresholds = metrics.roc_curve(actual, predictions, pos_label=1)

# Measure the area under the curve.  The closer to 1, the "better" the predictions.
print("AUC of the predictions: {0}".format(metrics.auc(fpr, tpr)))



txtprediction = []
print("Naive Bayes")
print("THE DNA")



i=0
n=0
xaxis = ["BUSS","ENT","LIFE","SPORTS","TECH"]
yaxis = []
width = 1/1.5
ind = [0, 1, 2, 3, 4]
import glob, os
os.chdir("../news_output/DNA")
for file in glob.glob("*.txt"):
    print(file)
    actual = []
    with open(file,'r') as txtinput:
       with open(file.replace('.txt','')+"DnaNaive.csv",'w') as txtoutput:
         writer = csv.writer(txtoutput, delimiter=',')
         j=0
         for line in txtinput:
              actual.append(DNA[n][j])    
              x = line+","+str(make_decision(line, make_class_prediction))+","+str(DNA[n][j])
              writer.writerow(x.split(","))
              #print(line)
              #print(make_decision(line, make_class_prediction))
              txtprediction.extend([make_decision(line, make_class_prediction)])
              j+=1
         txtoutput.close()
         
    print((accuracy_score(actual,txtprediction)*100))
    n+=1
    final= dict(Counter(txtprediction))
    print(final)
    ans = (final[1])/(final[1]+final[-1])*100
    print ('Objective Percentage\t'+str(ans))
    yaxis.append(ans)
    txtprediction = []
    i=i+1

mp.bar(ind, yaxis, width, color="white",align ='center')
mp.xticks(ind,xaxis)
mp.ylabel('Objective Percentage')
mp.title('DNA India')
for a,b in zip(ind, yaxis):
    mp.text(a-0.3, b-2.2, str(round(b,2)),size = 13)
    
mp.show()
mp.savefig("dna.png")
print("THE TIMES OF INDIA")
yaxis = []
os.chdir("../TOI")
i=0
n=0
for file in glob.glob("*.txt"):
    print(file)
    actual = []
    with open(file,'r') as txtinput:
       with open(file.replace('.txt','')+"ToiNaive.csv",'w') as txtoutput:
          writer = csv.writer(txtoutput, delimiter=',')
          j=0
          for line in txtinput:
             x = line+","+str(make_decision(line, make_class_prediction))+","+str(TOI[n][j])
             actual.append(TOI[n][j])
             writer.writerow(x.split(","))
             txtprediction.extend([make_decision(line, make_class_prediction)])
             j+=1
          txtoutput.close()
    #print(actual)
    #print(txtprediction)
    print(accuracy_score(actual,txtprediction))      
    final= dict(Counter(txtprediction))
    print(final)
    ans = (final[1])/(final[1]+final[-1])*100
    print ('Objective Percentage\t'+str(ans))
    yaxis.append(ans)     
    txtprediction = []
    i=i+1
    n+=1
mp.bar(ind, yaxis, width, color="white",align ='center')
mp.xticks(ind,xaxis)
mp.ylabel('Objective Percentage')
mp.title('THE TIMES OF INDIA')
for a,b in zip(ind, yaxis):
    mp.text(a-0.3, b-2.2, str(round(b,2)),size=13)
   
mp.show()
mp.savefig("toi.png")
after_stop_word.close()
