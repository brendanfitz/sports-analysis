# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 17:23:09 2021

@author: Brendan Non-Admin
"""

import pandas as pd
import numpy as np
from sports.basketball.ncaa import BayesBetaBinomial
import matplotlib.pyplot as plt
from sports.basketball.ncaa.ncaa_tourney_games_spider import START_YEAR, END_YEAR, FILENAME

df = pd.read_csv(FILENAME)

df.head()

assert(63 * (END_YEAR - START_YEAR + 1) == df.shape[0]) # check games total
assert(df.loc[df.team1_score == df.team2_score, ].empty) # no ties

year = 1985
region = 'national'
mask = (df.loc[:, 'year'] == year) & (df.loc[:, 'region'] == region)
df.loc[mask, ]


def seed_matchup(row):
    seeds = sorted(
        [
            str(row['team1_seed']).zfill(2), 
            str(row['team2_seed']).zfill(2)
        ]
    )
    
    return 'v'.join(seeds)

df.loc[:, 'seed_matchup'] = df.apply(seed_matchup, axis=1)

df.loc[:, 'winner'] = np.where(df.team1_score > df.team2_score, 'team1', 'team2')

df.head()

def seed_result(row):
    if row['team1_seed'] == row['team2_seed']:
        return 'equal_seeds'
    
    if (
            (
                row['winner']  == 'team1' 
            and row['team1_seed'] < row['team2_seed']
            )
        or            
            (
                row['winner']  == 'team2' 
            and row['team2_seed'] < row['team1_seed']
            )
        ):
        return 'higher_seed_win'
    
    return 'lower_seed_win'
    

df.loc[: , 'seed_result'] = df.apply(seed_result, axis=1)

df.groupby(['seed_matchup', 'seed_result']).count()


equal_seeds_mask = df.loc[:, 'team1_seed'] == df.loc[:, 'team2_seed']
df_seed_stats = (df.loc[~equal_seeds_mask, ['seed_matchup', 'seed_result']]
    .value_counts()
    .unstack('seed_result')
    .fillna(0)
)

df_seed_stats.to_excel('data/ncaa_seed_stats.xlsx')

df.head()

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
    plt.show()
    
    models[matchup] = model

mask = df.loc[:, 'round'] == 1
df_first_rounds = (df.loc[mask, ['seed_matchup', 'seed_result']]
    .value_counts()
    .unstack('seed_result')
    .fillna(0)
    .rename(columns=lambda x: x.replace(' ', '_') + 's')
    .assign(
        total_games=lambda x: x.sum(axis=1),
        higher_seed_win_pct=lambda x: x.higher_seed_wins.div(x.total_games),
        lower_seed_win_pct=lambda x: x.lower_seed_wins.div(x.total_games),
        # higher_seed_win_pct_prior_mean=lambda x: x.index.map(lambda x: models[x].prior_dist.mean()),
        higher_seed_win_pct_posterior_mean=lambda x: x.index.map(lambda x: models[x].posterior_dist.mean()),
        higher_seed_win_pct_lower_bound=lambda x: x.index.map(lambda x: models[x].credible_interval()[0]),
        higher_seed_win_pct_upper_bound=lambda x: x.index.map(lambda x: models[x].credible_interval()[1]),
        
    )
)

df_first_rounds
