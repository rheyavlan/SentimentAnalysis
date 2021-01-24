"""
classifier.py
==========
"""
from collections import Counter
import matplotlib.pyplot as mp
import os
import re
import sys
import json
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass
from glob import glob
from sklearn.externals import joblib
from sklearn.datasets.base import Bunch
from collections import OrderedDict
import scraper

def thetimesofindia(soup):
    body = soup.find_all('div', id_='Normal')
    if body:
        return extract_text_from_p(body)
    return None

def thedna(soup):
    body = soup.find_all('div', id_='Normal')
    if body:
        return extract_text_from_p(body)
    return None

CATEGORIES = OrderedDict(
        [
        ['business', [[thetimesofindia, '1898055.cms'],
                       [thedna, 'business']]],
         ['entertainment', [[thedna, 'entertainment'],
                            [thetimesofindia, '1081479906.cms']
                            ]],
         ['lifestyle', [[thedna, 'life'],
                            [thetimesofindia, '3908999.cms']]],
         ['sports', [[thetimesofindia, '4719148.cms'],
                     [thedna, 'sport'],
                     [thedna, 'football']
                     ]],
         ['technology', [[thedna, 'entertainment_and_arts'],
                            [thetimesofindia, '5880659.cms']]]
        
        ])

def get_data_TOI(data_path):
    """ Get training data from the articles folder. """
    all_data = []
    print(CATEGORIES.keys())
    for path in glob(os.path.join(data_path, 'toi.json')):
        with open(path, 'r') as jsonfile:
            data = json.loads(jsonfile.read())
            for article in data.get('articles'):
                all_data.extend([scraper.clean(article['content'])])
    jsonfile.close()
    return Bunch(categories=CATEGORIES.keys(),
                 values=None,
                 data=all_data)


def get_data_DNA(data_path):
    """ Get training data from the articles folder. """
    all_data = []

    for path in glob(os.path.join(data_path, 'dna.json')):
        with open(path, 'r') as jsonfile:
            data = json.loads(jsonfile.read())
            for article in data.get('articles'):
                all_data.extend([scraper.clean(article['content'])])

    return Bunch(categories=scraper.CATEGORIES.keys(),
                 values=None,
                 data=all_data)

def main(path):

    # Get the latest .pkl
    files = glob(os.path.join(path, '*.pkl'))
    filename = max(files, key=lambda x: int(re.sub(r'\D', '', x)))

    # Check if the models exists
    if not filename:
        print("No models found in %s" % path)
        sys.exit(1)

    # Load the models using the already generated .pkl file
    model = joblib.load(filename)
    data_toi = get_data_TOI('input')
    data_dna = get_data_DNA('input')
    data_weighted_toi = model['vectorizer'].transform(data_toi.data)
    data_weighted_toi = model['feature_selection'].transform(data_weighted_toi)
    prediction_toi = model['clf'].predict(data_weighted_toi)

    
    data_weighted_dna = model['vectorizer'].transform(data_dna.data)
    data_weighted_dna = model['feature_selection'].transform(data_weighted_dna)
    prediction_dna = model['clf'].predict(data_weighted_dna)

    print("TOI")
    labels = ["business","entertainment","lifestyle","sports","technology"]
    colors = ["white","white","white","white","white"]
    print(prediction_toi)
    final_toi= dict(Counter(prediction_toi))
    values = []
    values.append(final_toi[0])
    values.append(final_toi[1])
    values.append(final_toi[2])
    values.append(final_toi[3])
    values.append(final_toi[4])
    mp.figure(0)
    mp.pie(values, labels=labels,colors=colors,autopct='%.2f')
    mp.title("The Times Of India")
    print("DNA")
    print(prediction_dna)
    final_dna= dict(Counter(prediction_dna))
    values = []
    values.append(final_dna[0])
    values.append(final_dna[1])
    values.append(final_dna[2])
    values.append(final_dna[3])
    values.append(final_dna[4])
    mp.figure(1)
    mp.pie(values, labels=labels,colors=colors,autopct='%.2f')
    mp.title("DNA India")
    os.chdir("./news_output/TOI")
    #print(list(scraper.CATEGORIES.keys())[prediction])
    mp.show()
    for text, prediction in zip(data_toi.data, prediction_toi):
         for x in CATEGORIES.keys():
              with open(x+".txt",'a') as f:
                    if(list(CATEGORIES.keys())[prediction] is x):
                            f.write(text)
                            print(list(CATEGORIES.keys())[prediction].ljust(15, ' '), text[:100], '...')
                                 
    os.chdir("../DNA")
    for text, prediction in zip(data_dna.data, prediction_dna):
         for x in CATEGORIES.keys():
              with open(x+".txt",'a') as f:
                    if(list(CATEGORIES.keys())[prediction] is x):
                            f.write(text)
                            print(list(CATEGORIES.keys())[prediction].ljust(15, ' '), text[:100], '...')

   
if __name__ == '__main__':
    main('training')
