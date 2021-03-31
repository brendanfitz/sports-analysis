# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:23:09 2021

@author: Brendan Non-Admin
"""

import pandas as pd
import numpy as np
from sports.basketball.ncaa import BayesBetaBinomial
import matplotlib.pyplot as plt
from sports.basketball.ncaa import utils
from sports.basketball.ncaa.ncaa_tourney_games_spider import FILENAME

df = (pd.read_csv(FILENAME)
    .rename(columns=lambda x: x.replace('team1', 'home').replace('team2', 'away'))
    .rename(columns={'home_name': 'home_team', 'away_name': 'away_team'})
)

utils.data_checks(df)

utils.print_round(df, year=1990, round_=3)

df.loc[:, ]

df = df.assign(
    seed_matchup=lambda x: x.apply(utils.seed_matchup, axis=1),
    winner=lambda x: np.where(x.home_score > x.away_score, 'home_team', 'away_team'),
    seed_result=lambda x: x.apply(utils.seed_result, axis=1),
)

df.groupby(['seed_matchup', 'seed_result']).count()

equal_seeds_mask = df.loc[:, 'home_seed'] == df.loc[:, 'home_seed']
df_seed_stats = (df.loc[~equal_seeds_mask, ['seed_matchup', 'seed_result']]
    .value_counts()
    .unstack('seed_result')
    .fillna(0)
)

df_seed_stats.to_excel('data/ncaa_seed_stats.xlsx')

mask = df.loc[:, 'round'] == 1
first_round_matchups = (df.loc[mask, 'seed_matchup']
    .drop_duplicates()
    .sort_values()
    .tolist()
)

models = dict()

for matchup in first_round_matchups:
    df_matchup = df.loc[df.seed_matchup == matchup, ]
    
    higher_seed, lower_seed = list(map(int, matchup.split('v')))
    
    model = BayesBetaBinomial(matchup, a_prior=lower_seed, b_prior=higher_seed)
    
    x = sum(df_matchup.seed_result == 'higher_seed_win')
    n = df_matchup.shape[0]
    model.update(x, n)
    
    title = f'{higher_seed} vs {lower_seed} Seed Matchups - Bayesian Posterior Distribution'
    ax = model.plot_posterior(title)
    plt.savefig(f'data/{matchup}_posterior_distribution.png')
    plt.show()
    
    models[matchup] = model
    
g = utils.RidgePlot(models)
plt.savefig('data/NCAA First Round Posterior Distribution Ridgeplot.png')
plt.show()


df_first_rounds = utils.create_df_first_rounds(df, models)

filepath = "data/first_round_matchups.html"
(df_first_rounds.apply(utils.numeric_col_formatter)
    .drop(['total_games'], axis=1)
    .rename(columns=lambda x: x.replace('_', ' ').title())
    .pipe(utils.to_html, filepath)
)

filepath = "data/ncaa-data.html"
(df.head()
    .pipe(utils.to_html, filepath)
)


