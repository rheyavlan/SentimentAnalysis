"""
scraper.py
"""

import os
import sys
import re
import json
import random
from collections import OrderedDict

import requests
from bs4 import BeautifulSoup


#
# Newspapers parsers
#

def extract_text_from_p(body):
    return clean(' '.join([t.get_text() for t in body.find_all('p')]))

def thetimesofindia(soup):
    body = soup.body.find('div', attrs={'class': 'Normal'})
    if body:
        return body.text
    return None

def thedna(soup):
    body = soup.body.find('div', {'class':'body-text'})
    if body:
        return extract_text_from_p(body)
    return None
#
# Categories we'll use to classify
#

CATEGORIES = OrderedDict(
        [
        ['business', [[thetimesofindia, '1898055.cms'],
                       [thedna, 'money']]],
        #,['education', [[thetimesofindia, '913168846.cms'], [thedna, 'education']]],
         ['health', [[thedna, 'health'],
                            [thetimesofindia, '3908999.cms']]],
         #['science', [[thedna, 'sci-tech'],[thetimesofindia, '-2128672765.cms']]],
         ['technology', [[thedna, 'scitech'],
                            [thetimesofindia, '5880659.cms']]],
         ['entertainment', [[thedna, 'entertainment'],
                            [thetimesofindia, '1081479906.cms']
                            ]],
         ['sports', [[thetimesofindia, '4719148.cms'],
                     [thedna, 'sport']
                     ]]
        ])

#
# RSS for every newspaper
#
     
RSS = {thetimesofindia: 'http://timesofindia.indiatimes.com/rssfeeds/{0}',
       thedna:'http://www.dnaindia.com/feeds/{0}.xml'}
       

def clean(text):
    text = text.replace(',','')
    #text = re.sub(r'\W', ' ', text)
    #text = re.sub(r'\s+', ' ', text)
    return text


def main(path):

    # Create destination directory if it doesn't exist:
    if not os.path.exists(path):
        os.mkdir(path)
    non_bmp_map = dict.fromkeys(range(0x10000, sys.maxunicode + 1), 0xfffd)

    # Get a json of articles for every category.
    for category, sources in CATEGORIES.items():

        content = []
        
        for parser, source_category in sources:
            print(parser)
            print(source_category)
            # Get the RSS
            link = RSS.get(parser).format(source_category)
            print(link)
            print("=" * 50)

            feed = requests.get(link, timeout=20)
            if feed.status_code != 200:
                continue

            # Loop all over the news and parse each one using
            # the appropiate parser.
              
            for url in BeautifulSoup(feed.content).find_all('guid'):
                try:
                    print(url.text)
                    article = requests.get(url.text, timeout=40)
                except Exception:
                    continue
                #print(article)
                soup = BeautifulSoup(article.content,'html.parser')
                #print(soup)
                #businesspage1 = urlopen(x)
                 #                data1 = businesspage1.read()
                  #               soup1 = BeautifulSoup(data1,'html.parser')
                #body = soup.body.find('div', attrs={'class': 'Normal'})
                                #print(body)
                body = parser(soup)  
                if body:
                    content.append(body)
                    #print(content.translate(non_bmp_map))
        random.shuffle(content)

        # Save all the articles shuffled as json
        with open('articles/{0}.json'.format(category), 'w') as output:
            output.write(json.dumps(content))

if __name__ == '__main__':
    main('articles')
