# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 16:47:39 2021

@author: Brendan Non-Admin
"""

from urllib.request import urlopen
from lxml import etree
from time import sleep
import csv

filename = 'data/ncaa_tournament_data.csv'
with open(filename, 'w', newline='', encoding='utf-8') as f:
    csvwriter = csv.writer(f, delimiter=',')
    
    COLUMN_NAMES = ['year', 'region', 'round', 'team1_rank', 'team1_name', 'team1_score', 'team2_rank', 'team2_name', 'team2_score']
    csvwriter.writerow(COLUMN_NAMES)
    
    START_YEAR = 1985
    END_YEAR = 2019
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
        
    
"""
analysis
"""
import pandas as pd
import numpy as np
from sports.basketball.ncaa import BayesBetaBinomial
import matplotlib.pyplot as plt

df = pd.read_csv(filename)

df.head()

assert(63 * len(years) == df.shape[0]) # games check
assert(df.loc[df.team1_score == df.team2_score, ].empty) # no ties

year = 1985
region = 'national'
mask = (df.loc[:, 'year'] == year) & (df.loc[:, 'region'] == region)
df.loc[mask, ]


def rank_matchup(row):
    ranks = sorted(
        [
            str(row['team1_rank']).zfill(2), 
            str(row['team2_rank']).zfill(2)
        ]
    )
    
    return 'v'.join(ranks)

df.loc[:, 'rank_matchup'] = df.apply(rank_matchup, axis=1)

df.loc[:, 'winner'] = np.where(df.team1_score > df.team2_score, 'team1', 'team2')

df.head()

def rank_win(row):
    if row['team1_rank'] == row['team2_rank']:
        return 'equal rank'
    
    if (
            (
                row['winner']  == 'team1' 
            and row['team1_rank'] < row['team2_rank']
            )
        or            
            (
                row['winner']  == 'team2' 
            and row['team2_rank'] < row['team1_rank']
            )
        ):
        return 'higher ranked win'
    
    return 'lower ranked win'
    

df.loc[: , 'rank_win'] = df.apply(rank_win, axis=1)

df.groupby(['rank_matchup', 'rank_win']).count()


equal_ranks_mask = df.loc[:, 'team1_rank'] == df.loc[:, 'team2_rank']
df_ranking_wins = (df.loc[~equal_ranks_mask, ['rank_matchup', 'rank_win']]
    .value_counts()
    .unstack('rank_win')
    .fillna(0)
)

df_ranking_wins.to_excel('data/ncaa_ranking_wins.xlsx')

df.head()

a_prior = 16
b_prior = 1

mask = df.loc[:, 'round'] == 1
first_round_matchups = (df.loc[mask, 'rank_matchup']
    .drop_duplicates()
    .sort_values()
    .tolist()
)

for matchup in first_round_matchups:
    df_matchup = df.loc[df.rank_matchup == matchup, ]
    
    higher_seed, lower_seed = list(map(int, matchup.split('v')))
    
    model = BayesBetaBinomial(matchup, a_prior=lower_seed, b_prior=higher_seed)
    
    x = sum(df_matchup.rank_win == 'higher ranked win')
    n = df_matchup.shape[0]
    model.update(x, n)
    
    title = f'{higher_seed} vs {lower_seed} Seed Matchups - Bayesian Posterior Distribution'
    ax = model.plot_posterior(title)
    plt.show()
