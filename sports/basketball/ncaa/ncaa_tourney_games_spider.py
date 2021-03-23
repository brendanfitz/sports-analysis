# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:47:39 2021

@author: Brendan Non-Admin
"""

from urllib.request import urlopen
from lxml import etree
from time import sleep
import csv

FILENAME = 'data/ncaa_tournament_data.csv'
START_YEAR = 1985
END_YEAR = 2019
COLUMN_NAMES = [
    'year',
    'region',
    'round',
    'team1_seed',
    'team1_name',
    'team1_score',
    'team2_seed',
    'team2_name',
    'team2_score',
]

def main():
    with open(FILENAME, 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f, delimiter=',')
    
        csvwriter.writerow(COLUMN_NAMES)
        
        years = range(START_YEAR, END_YEAR+1)
        for year in years:
            url = f'https://www.sports-reference.com/cbb/postseason/{year}-ncaa.html'
            response = urlopen(url)
            
            htmlparser = etree.HTMLParser()
            tree = etree.parse(response, htmlparser)
            
            brackets = tree.xpath('//div[@id="brackets"]/div')
            
            for bracket in brackets:
                rounds = bracket.xpath('.//div[@class="round"]')[:-1]
                
                region = bracket.get('id')
                round_n = 5 if region == 'national' else 1
                
                for round_ in rounds:
                    games = round_.xpath('./div')
                    for game in games:
                        game_data = list()
                        game_data.extend([year, region, round_n])
                        teams = game.xpath('./div')
                        for team in teams:
                            seed = team.xpath('span')[0].text
                            team_name, score = [x.text for x in team.xpath('.//a')]
                            game_data.extend([seed, team_name, score])
                        csvwriter.writerow(game_data)
                    round_n += 1
            sleep(3)

if __name__ == '__main__':
    main()