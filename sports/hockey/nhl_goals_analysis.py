# -*- coding: utf-8 -*-
"""
Created on Thu May 21 12:55:12 2020

@author: Brendan Non-Admin
"""

import os
import numpy as np
import math
import pandas as pd

filename = os.path.join('data', 'nhl_player_data_1990-2020.csv')
df = pd.read_csv(filename)

df.loc[:, 'year_season_start'] = df.seasonId.astype(str).str[:4].astype('int64')
df.loc[:, 'year_season_end'] = df.seasonId.astype(str).str[4:].astype('int64')

subsetcols = [
    'skaterFullName', 'seasonId', 'goals', 'gamesPlayed', 'positionCode', 
    'penaltyMinutes', 'plusMinus', 'shootingPct', 'shots', 'teamAbbrevs',
    'year_season_start', 'year_season_end',
]
df = (df
    .loc[:, subsetcols]
    .drop_duplicates()
)

subset = ['skaterFullName', 'year_season_start']
mask = df.duplicated(subset=subset, keep=False)
(df
 .loc[mask, subsetcols]
 .sort_values(subset[::-1])
 .to_csv('duplicates.csv')
)

subset = ['skaterFullName', 'year_season_start']
mask = ~df.duplicated(subset=subset, keep='first')
df = (df
 .loc[mask, subsetcols]
)

df = (df
    .set_index(['skaterFullName', 'year_season_start'], drop=False, verify_integrity=True)
    .sort_index()
)

df.index = df.index.rename([x + '_idx' for x in list(df.index.names)])

df.loc[:, 'season_number'] = (df.groupby(['skaterFullName'])
    ['year_season_start'].rank(method="first")
)

df.loc[:, 'years_played'] = (df.groupby('skaterFullName')
    ['season_number'].transform('max')
)


import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

x = (df.loc[:, ['skaterFullName', 'years_played']]
     .drop_duplicates()
     .years_played
    )
sns.distplot(x)

data = df.groupby('season_number', as_index=False)['goals'].sum()
sns.lineplot('season_number', 'goals', data=data)

df.groupby('positionCode').goals.mean()
df = df.assign(goals_per_game=lambda x: x.goals.div(x.gamesPlayed))
df.groupby('positionCode').goals_per_game.mean() * 82

def goals_per_player_by_season(input_df):
    """
    Parameters
    ----------
    input_df : dataframe with goals and season_number columns

    Returns
    -------
    df : dataframe grouped by goals and season number
    """
    df = input_df.copy()
    df = (df.groupby('season_number')['goals'].mean()
          .reset_index(name='goals_per_player')
         )
    return df

df.groupby('year_season_start')['goals'].mean()

def plot_goals_per_player_by_season(df, years_played, ax=None):
    label = "All Players" if years_played is None else "{} Years Played".format(years_played)
    if years_played:
        mask = df.loc[:, 'years_played'] == years_played
        df = df.loc[mask, :]
    data = df.pipe(goals_per_player_by_season)
    if ax:
        return sns.lineplot('season_number', 'goals_per_player', label=label, data=data, ax=ax)
    return sns.lineplot('season_number', 'goals_per_player', label=label, data=data)
        
""" Run all at once for same plot """
max_seasons = 15
for i in range(2, max_seasons+1, 2):
    ax = plot_goals_per_player_by_season(df, i)
sns.lineplot('season_number', 'goals_per_player', label="All", data=df.loc[df.years_played <= 15, ].pipe(goals_per_player_by_season))
plt.xticks(np.arange(1, max_seasons+6, step=1))
ax.set(title="Goals by Season Number and Years Played", xlabel="Season Number", ylabel="Average Goals Per Player")
plt.legend()
plt.show()


df.season_number.value_counts()
sns.countplot(df.season_number)

ovi = df.loc[df.skaterFullName == 'Alex Ovechkin', ]
ax = plot_goals_per_player_by_season(ovi, None)
plt.xticks(np.arange(1, df.years_played.max(), step=1))
plt.legend()
plt.show()

mask = df.loc[:, 'years_played'] >= 5
stats = (df
 .loc[mask, ].groupby('positionCode')
 .goals.agg(['mean', 'std', 'median'])
 .T
 .loc[:, ['L', 'C', 'R', 'D',]]
)
quantiles = (df
 .loc[mask, ].groupby('positionCode')
 .goals.quantile([0.25, 0.75,])
 .unstack('positionCode')
)
pd.concat([stats, quantiles]).round(2)

(df.groupby('positionCode')
 .goals
 .describe()
 .round(2)
 .T
)

x = 'shootingPct'
y = 'goals'
mask = (df.year_season_start == df.year_season_start.max()) & (df.shootingPct < 0.4)
data = df.loc[mask, [x, y]]
sns.scatterplot(x=x, y=y, data=data)

import statsmodels.formula.api as smf
from statsmodel_results_html import results_as_html

additional_cols = ['season_number', 'gamesPlayed', 'positionCode']

def create_lags(input_df, metric, n=5, include_metric_name_in_columns=False):
    df = input_df.copy()
    frames = [df.groupby('skaterFullName')[metric].shift(i) for i in range(n+1)]
    if include_metric_name_in_columns:
        keys = [metric] + [metric + '_L%s' % i for i in range(1, n+1)]
    else:
        keys = ['y'] + ['L%s' % i for i in range(1, n+1)]
    df = (pd.concat(frames, axis=1, keys=keys)
        .dropna()
    )
    return df

def create_X(n, additional_cols):
    """
    Using this to avoid excessive dropping of na's
    """
    X =     
    
    X = X.join(df.loc[:, additional_cols])
    return X

mask = df.loc[:, 'years_played'] >= 5
df = df.loc[mask, :]

X = create_X(3, additional_cols)    

mod = smf.ols('y ~ L1 + L2 + L3', data=X)
results = mod.fit()
results.summary()
results_as_html(results, 'model1.html', 'AR(3) Model Results')

X = create_X(5, additional_cols)

model = smf.ols('y ~ L1 + L2 + L3 + L4 + L5', data=X)
results = model.fit()
results.summary()
results_as_html(results, 'model2.html', 'AR(5) Model Results')

l5_str = 'L1 + L2 + L3 + L4 + L5'
formula = 'y ~ {} + season_number'.format(l5_str)
mod = smf.ols(formula, data=X)
results = mod.fit()
results.summary()
results_as_html(results, 'model3.html', 'AR(5) w/ Season Number Model Results')

l5_str = 'L1 + L2 + L3 + L4 + L5'
formula = 'y ~ {} + season_number + gamesPlayed'.format(l5_str)
mod = smf.ols(formula, data=X)
results = mod.fit()
results.summary()
results_as_html(results, 'model4.html', 'AR(5) w/ Season Number & Games Played Model Results')

X.loc[:, 'season_number_squared'] = X.season_number.pow(2)
formula = ('y ~ {} + season_number + season_number_squared + '
           'gamesPlayed').format(l5_str)
mod = smf.ols(formula, data=X)
results = mod.fit()
results.summary()
results_as_html(results, 'model5.html', 'AR(5) w/ Season Number Squared & Games Played Model Results')
X = X.drop('season_number_squared', axis=1) # Ignore due to multi-collinearity

formula = ('y ~ {} + season_number + gamesPlayed + C(positionCode)'
           .format(l5_str))
mod = smf.ols(formula, data=X) 
results = mod.fit()
results.summary()
results_as_html(results, 'model6.html', 'AR(5) w/ Season Number, Games Played & Position Code Model Results')

position_code_map = {'D': 'D', 'C': 'F', 'R': 'F', 'L': 'F'}
X.loc[:, 'positionCode'] = X.loc[:, 'positionCode'].map(position_code_map)

formula = ('y ~ {} + season_number + gamesPlayed + C(positionCode)'
           .format(l5_str))
mod = smf.ols(formula, data=X)
results = mod.fit()
results.summary()
results_as_html(results, 'model7.html', 'AR(5) w/ Season Number, Games Played & Position Code (D vs F) Model Results')

print("Root MSE (Total): {:0.2f}".format(math.sqrt(results.mse_total)))

y_pred = results.predict(X).rename('predicted_goals')

cols = [
    "skaterFullName",
    "season_number", 
    "year_season_start", 
    "year_season_end", 
    "gamesPlayed", 
    "years_played", 
    "positionCode", 
    "L1", 
    "L2", 
    "L3", 
    "L4", 
    "L5", 
    "goals", 
    "predicted_goals",
    "mae",
]

filename = os.path.join('data', 'nhl_player_goal_predictions_1990-2019.xlsx')
(df.join(y_pred)
    .join(X.drop(['y'] + additional_cols, axis=1))
    .assign(mae=lambda x: (x.goals - x.predicted_goals).abs())
    .loc[:, cols]
    .to_excel(filename, index=False)
)

model_filename = os.path.join('data', 'nhl_goals_regression_model.pkl')
results.save(model_filename)

from statsmodels.regression.linear_model import OLSResults

nhl_goals_mod = OLSResults.load(model_filename)

row = pd.DataFrame({
        'L1': 17,
        'L2': 20,
        'L3': 18,
        'L4': 18,
        'L5': 18,
        'season_number': 1,
        'gamesPlayed': 50,
        'positionCode': 'D'
    }, index=[0]
)
results.predict(row)[0]

"""
Diagnostics
"""
residuals = (X.loc[:, 'y'] - results.fittedvalues).rename('residuals')
sns.scatterplot(X.loc[:, 'y'], residuals, hue=X.positionCode, alpha=0.4)

import scipy as sp

fig, ax = plt.subplots(figsize=(6,2.5))
_, (__, ___, r) = sp.stats.probplot(residuals, plot=ax, fit=True)
r**2

sns.distplot(residuals)

X = pd.concat([X, residuals], axis=1)

X.loc[X.y > 40, ]

"""
Additional lag test
"""

lagged_metrics = ['penaltyMinutes', 'plusMinus', 'shootingPct', 'shots',]
lagged_dfs = [create_lags(df, x, 1, include_metric_name_in_columns=True).drop(x, axis=1) for x in lagged_metrics]

for x in lagged_dfs:
    df = df.join(x)
    
additional_cols = [
    'season_number', 'gamesPlayed', 'positionCode',
    'penaltyMinutes_L1', 'plusMinus_L1', 'shootingPct_L1', 'shots_L1'
]
X = create_X(5, additional_cols)
position_code_map = {'D': 'D', 'C': 'F', 'R': 'F', 'L': 'F'}
X.loc[:, 'positionCode'] = X.loc[:, 'positionCode'].map(position_code_map)

formula = ('y ~ {} + season_number + gamesPlayed + C(positionCode) + penaltyMinutes_L1 + plusMinus_L1 + shootingPct_L1 + shots_L1'
           .format(l5_str))
mod = smf.ols(formula, data=X)
results = mod.fit()
results.summary()

formula = ('y ~ {} + season_number + gamesPlayed + C(positionCode) + penaltyMinutes_L1'
           .format(l5_str))
mod = smf.ols(formula, data=X)
results = mod.fit()
results.summary()


"""
10 Years Played
"""

df = df.loc[df.years_played == 10, ]
additional_columns = ['season_number',
 'gamesPlayed',
 'positionCode',]
X = create_X(5, additional_cols)
position_code_map = {'D': 'D', 'C': 'F', 'R': 'F', 'L': 'F'}
X.loc[:, 'positionCode'] = X.loc[:, 'positionCode'].map(position_code_map)
X.loc[:, 'season_number_squared'] = X.season_number.pow(2)

formula = ('y ~ L1 + L2 + L3 + L4 + L5 + season_number + season_number_squared + gamesPlayed + C(positionCode)'
           .format(l5_str))
mod = smf.ols(formula, data=X)
results = mod.fit()
results.summary()

(X.loc[:, ['season_number', 'season_number_squared']]
 .drop_duplicates()
 .sort_values(by='season_number')
)


"""
We're only predicting players after they've had 5 years of experience.
This makes the season_number coefficient negative because after 5 years, a player is usually on the second half of their 
career, meaning they are on the downward side of the U-shape relationship between season number and goals.
"""


""" Output Mark Messier """
columns = [
    'skaterFullName',
    'year_season_start',
    'season_number',
    'positionCode', 
    'goals',
    'gamesPlayed',
]
mask = df.loc[:, 'skaterFullName'] == 'Alex Ovechkin'
filename = os.path.join('data', 'alexander_ovechkin.csv')
(df.loc[mask, columns]
 .sort_values('year_season_start')
 .to_csv(filename, index=False)
)


"""
Top goal scorers
"""
top_goal_scorers = (df.groupby('skaterFullName').goals.sum()
                    .nlargest(50)
                    .index
                    .tolist()
                    )
columns = [
    'skaterFullName',
    'year_season_start',
    'season_number',
    'goals',
    'gamesPlayed',
]
mask = df.skaterFullName.isin(top_goal_scorers)
filename = os.path.join('data', 'top_50_goal_scorers.csv')
(df.loc[mask, columns]
 .to_csv(filename)
)