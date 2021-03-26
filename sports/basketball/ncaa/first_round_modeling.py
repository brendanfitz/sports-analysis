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
from sports.basketball.ncaa import utils

df = (pd.read_csv(FILENAME)
    .rename(columns=lambda x: x.replace('team1', 'home').replace('team2', 'away'))
    .rename(columns={'home_name': 'home_team', 'away_name': 'away_team'})
)

assert(63 * (END_YEAR - START_YEAR + 1) == df.shape[0]) # check games total
assert(df.loc[df.home_score == df.away_score, ].empty) # no ties

year = 1985
region = 'national'
mask = (df.loc[:, 'year'] == year) & (df.loc[:, 'region'] == region)
df.loc[mask, ]

df.loc[:, 'seed_matchup'] = df.apply(utils.seed_matchup, axis=1)

df.loc[:, 'winner'] = np.where(df.home_score > df.away_score, 'home_team', 'away_team')

df.loc[: , 'seed_result'] = df.apply(utils.seed_result, axis=1)

df.groupby(['seed_matchup', 'seed_result']).count()

equal_seeds_mask = df.loc[:, 'home_seed'] == df.loc[:, 'home_seed']
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
    
g = utils.RidgePlot(models)
plt.show()

mask = df.loc[:, 'round'] == 1
df_first_rounds = (df.loc[mask, ['seed_matchup', 'seed_result']]
    .value_counts()
    .unstack('seed_result')
    .fillna(0)
    .rename(columns=lambda x: x.replace(' ', '_') + 's')
    .assign(
        total_games=lambda x: x.sum(axis=1),
        higher_seed_win_pct=lambda x: x.higher_seed_wins.div(x.total_games),
        # lower_seed_win_pct=lambda x: x.lower_seed_wins.div(x.total_games),
        # higher_seed_win_pct_prior_mean=lambda x: x.index.map(lambda x: models[x].prior_dist.mean()),
        higher_seed_win_pct_posterior_mean=lambda x: x.index.map(lambda x: models[x].posterior_dist.mean()),
        higher_seed_win_pct_lower_bound=lambda x: x.index.map(lambda x: models[x].credible_interval()[0]),
        higher_seed_win_pct_upper_bound=lambda x: x.index.map(lambda x: models[x].credible_interval()[1]),
        
    )
)


html = (df_first_rounds.apply(utils.numeric_col_formatter)
    .drop(['total_games'], axis=1)
    .rename(columns=lambda x: x.replace('_', ' ').title())
    .to_html(classes="table", border=0, justify="left", index_names=False)
    .replace('class="dataframe ', 'class="')
)
filename = "data/first_round_matchups.html"
with open(filename, 'w') as f:
    f.write(html)
