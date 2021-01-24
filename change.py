import os
import re
import json
import random
from collections import OrderedDict

import requests
from bs4 import BeautifulSoup


def timesofindia(soup):
    print(soup)
    body = soup.find_all('div', class_='story')
    if body:
        return extract_text_from_p(body)
    return None


def hindu(soup):
    body = soup.find_all('div', id_='articleText')
    if body:
        return extract_text_from_p(body)
    return None

CATEGORIES = OrderedDict(
        [['business', [ [timesofindia, 'business'],
                        [hindu,'business']]],
         ['health', [[timesofindia, 'health'],
                     [hindu, 'UKHealthNews']]],
         ['technology', [[timesofindia, 'technology'],
                         [hindu, 'technologyNews']]],
         ['entertainment', [[hindu, 'entertainment_and_arts'],
                            [timesofindia, 'culture']]],
         ['sports', [[timesofindia, 'sport'],
                     [timesofindia, 'football'],
                     [hindu, 'UKSportsNews']]]])
# Save all the articles shuffled as json
for category, sources in CATEGORIES.items():
      with open('articles/'+category+'.txt',mode='r', errors='ignore') as input:
           content = input.read()
           with open('articles/{0}.json'.format(category), 'w') as output:
             output.write(json.dumps(content))
