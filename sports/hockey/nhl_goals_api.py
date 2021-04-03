# -*- coding: utf-8 -*-
"""
Created on Thu May 21 10:51:24 2020

@author: Brendan Non-Admin
"""

import pandas as pd
import requests
import urllib.parse as up
import json
import copy
import os
import time
from tqdm import tqdm

url = ('https://api.nhle.com/stats/rest/en/skater/summary'
       '?isAggregate=false'
       '&isGame=false'
       '&sort=%5B%7B%22property%22:%22goals%22,%22direction%22:%22DESC%22%7D%5D'
       '&start=0'
       '&limit=100'
       '&factCayenneExp=gamesPlayed%3E=1'
       '&cayenneExp=gameTypeId=2%20and%20'
           'seasonId%3C=20192020%20and%20seasonId%3E=20192020'
      )
url_base, qstr = url.split('?')

response = requests.get(url)
data = json.loads(response.text)

def urlencode_wrapper(qstrobj):
    return (up.urlencode(qstrobj, quote_via=up.quote)
        .replace('%3D', '=')
        .replace('%3A', ':')
        .replace('%2C', ',')
        .replace('%5B%27', '')
        .replace('%27%5D', '')
    )

assert(qstr == urlencode_wrapper(up.parse_qs(qstr)))

qstrobj = up.parse_qs(qstr)
qstrobj['cayenneExp'][0] = qstrobj['cayenneExp'][0].replace('20192020', '{year}')

def generate_qstrobj(qstrobj, start, year):
    qstrobj = copy.deepcopy(qstrobj)
    qstrobj['start'] = start
    qstrobj['cayenneExp'][0] = qstrobj['cayenneExp'][0].format(year=year)
    return urlencode_wrapper(qstrobj)

years = [str(x) + str(x+1) for x in range(1917, 2020, 1)]
frames = []

def get_total_players_in_year(year):
    url = url_base + '?' + generate_qstrobj(qstrobj, 0, year)
    response = requests.get(url)
    data = json.loads(response.text)
    total_players_in_year = data['total']
    return total_players_in_year

for year in tqdm(years):
    total_per_year = get_total_players_in_year(year)
    
    start = 0
    while start < total_per_year:
        url = url_base + '?' + generate_qstrobj(qstrobj, start, year)
        response = requests.get(url)
        data = json.loads(response.text)
        frames.append(pd.DataFrame.from_records(data['data']))
        
        start = start + 100
        
        time.sleep(3)

df = pd.concat(frames)

filename = os.path.join('data', 'nhl_player_data_{}-{}.csv'.format(years[0][:4], int(years[-1][:4]) + 1))
df.to_csv(filename, index=False)

year = years[-1]
start = 801
url = url_base + '?' + generate_qstrobj(qstrobj, start, year)
response = requests.get(url)
data = json.loads(response.text)
data['total']
len(data['data'])
frames.append(pd.DataFrame.from_records(data['data']))




