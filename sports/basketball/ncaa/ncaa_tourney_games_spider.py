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
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv(filename)

df.head()

assert(63 * len(years) == df.shape[0]) # games check
assert(df.loc[df.team1_score == df.team2_score, ].empty) # no ties

year = 1985
region = 'national'
mask = (df.loc[:, 'year'] == year) & (df.loc[:, 'region'] == region)
df.loc[mask, ]


row = df.iloc[1]

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

from scipy.stats import beta

a_prior = 16
b_prior = 1

def bayesian_update_beta_binom(a_prior, b_prior, x, n, credible_interval_quantile=0.95):
    a_posterior = a_prior + x
    b_posterior = b_prior + n - x
    
    posterior_dist = beta(a_posterior, b_posterior)

    l = (1-credible_interval_quantile)/2
    u = 1 - l    
    credible_interval = posterior_dist.ppf((l, u))
    
    return posterior_dist, credible_interval

mask = df.loc[:, 'round'] == 1
first_round_matchups = (df.loc[mask, 'rank_matchup']
    .drop_duplicates()
    .sort_values()
    .tolist()
)

plt.style.use('ggplot')
plt.style.use('fivethirtyeight')

def plot_beta_binom_posterior(posterior_dist, credible_interval, matchup, ax=None):
    ls = "solid"
    
    fig, ax = plt.subplots()
    
    x = np.arange(0, 1.01, 0.001)
    y = posterior_dist.pdf(x)
    ax.plot(x, y, lw=2, ls=ls, color='navy', alpha=0.5)
    
    yrange_min, yrange_max = ax.get_ylim()
    yrange = yrange_max - yrange_min
    
    credible_interval_x = np.arange(*credible_interval, 0.001)
    credible_interval_y = posterior_dist.pdf(credible_interval_x)
    ax.fill_between(credible_interval_x, 0, credible_interval_y, color='lightskyblue', alpha=0.5)
    
    l, u = credible_interval
    
    higher_seed, lower_seed = tuple(map(int, matchup.split('v')))
    title = f'{higher_seed} vs {lower_seed} Seed Matchups - Bayesian Posterior Distribution'
    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Higher Ranked Team Win Probability", fontsize=10)
    
    ymin = (0 - yrange_min) / yrange
    
    posterior_mean = posterior_dist.mean()
    ymax = (posterior_dist.pdf(posterior_mean) - yrange_min) / yrange
    ax.axvline(x=posterior_dist.mean(), ymin=ymin, ymax=ymax, lw=2, alpha=0.5, color='navy', ls="dashed")
    
    ymax = (posterior_dist.pdf(l) - yrange_min) / yrange
    ax.axvline(x=l, ymin=ymin, ymax=ymax, lw=2, alpha=0.5, color='navy', ls="dashed")
    
    ymax = (posterior_dist.pdf(u) - yrange_min) / yrange
    ax.axvline(x=u, ymin=ymin, ymax=ymax, lw=2, alpha=0.5, color='navy', ls="dashed")
    
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_ticklabels([])

    s = f"Lower Bound: {l:>6.1%}\nUpper Bound: {u:>6.1%}"
    text_y = ax.get_ylim()[1] * 0.8
    ax.text(x=0.05, y=text_y, s=s, fontsize=8)
    
    s = f"Posterior Mean: {posterior_mean:>6.1%}"
    ax.text(x=posterior_mean, y=-0.15, s=s, fontsize=7.5, ha="center", va="top")
    
    return ax

for matchup in first_round_matchups:
    df_matchup = df.loc[df.rank_matchup == matchup, ]
    
    x = sum(df_matchup.rank_win == 'higher ranked win')
    n = df_matchup.shape[0]
    a_prior, b_prior = list(map(int, matchup.split('v')))[::-1]
    posterior_dist, credible_interval = bayesian_update_beta_binom(a_prior, b_prior, x, n)
    
    ax = plot_beta_binom_posterior(posterior_dist, credible_interval, matchup)
    plt.show()
