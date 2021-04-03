# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 17:43:10 2020

@author: Brendan Non-Admin

background: Damian Lillard claims it was easier to play in the bubble than normal
Let's run a t-test of his points in and out of the bubble to see if there was a statistically significant uptick in points
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t
from hypothesis_testers.basketball.nba import SingleMeanHypothesisTester

filename = 'data/Damian Lillard 2019-20 Game Log _ Basketball-Reference.com.html'

tables = pd.read_html(filename)

for table in tables:
    print(table.columns[:4])

data = (pd.concat(tables[-2:], axis=0)
    .query('Rk != "Rk"')
)
data.to_csv('data/damian_lillard_2020_stats.csv')

data.columns

data.iloc[:, 0:5].head()

data.columns.tolist()

subset = ['Rk', 'G', 'Date', 'Opp', 'PTS', 'AST', 'STL', 'TRB']
mask = data.loc[:, 'GS'] == '1'
data.loc[~mask, subset]

df  = (data.loc[mask, subset]
    .astype({
        'Rk': 'int32', 
        'G': 'int32', 
        'Date': 'datetime64',
        'PTS': 'int32',
        'AST': 'int32',
        'STL': 'int32',
        'TRB': 'int32',
        
    })
    .assign(
        bubble=lambda x: (x.Date >= '2020-04-01').astype('int32'),
        playoffs=lambda x: (x.Date >= '2020-08-17').astype('int32'),
        bubble_reg_szn=lambda x: x.bubble * x.playoffs,
        Bubble=lambda x: df.bubble.map({1: 'In', 0: 'Out'}),
        Playoffs=lambda x: df.playoffs.map({1: 'Playoffs', 0: 'Regular Season'}),
    )
)
df.to_csv('data/damian_lillard.csv', index=False)

df.groupby('Bubble')[['PTS', 'AST', 'STL', 'TRB']].mean()

dims = ['Playoffs', 'Bubble']
measures = ['PTS', 'AST', 'STL', 'TRB']
html = (df.groupby(dims)[measures].mean()
    .applymap('{:,.1f}'.format)
    .to_html(classes="table", border=0, justify="left")
    .replace('class="dataframe ', 'class="')
)

fout = 'data/lillard_summary_stats.html'
with open(fout, 'w') as f:
    f.write(html)
    
tester = SingleMeanHypothesisTester(df, 'bubble', 'PTS')
tester.run_hypotheis_test()
tester.conf_int()
tester.test_visual(title="Damian Lillard's Regular Season vs Bubble Hypothesis Test")
plt.show()

# simulate under full season conditions
tester.run_hypotheis_test(n1=82, n2=82)

mask = df.loc[:, 'playoffs'] == 0
df_reg_szn = df.loc[mask, :]
tester = SingleMeanHypothesisTester(df_reg_szn, 'bubble', 'PTS')
tester.run_hypotheis_test()
tester.conf_int()
tester.y_bar_diff, tester.se

html = (tester.create_test_stats_df()
    .rename(index={
        'Sample 1': 'Sample 1 (Pre-Bubble)',
        'Sample 2': 'Sample 2 (In-Bubble)'
    })
    .to_html(classes="table", border=0, justify="left")
    .replace('class="dataframe ', 'class="')
)
fout = 'data/lillard_first_htest.html'
with open(fout, 'w') as f:
    f.write(html)

tester.test_visual(title="Damian Lillard's Regular Season vs Bubble Hypothesis Test")
# simulate under full season conditions
pvalue, sims = tester.simulation()
pvalue

ax = sns.histplot(sims)
ax.vlines(x=tester.y_bar_diff, ymin=0, ymax=100, color='red', linestyle='dashed')
ax.set(title="Damian Lillard's Regular Season vs Bubble Hypothesis Test\nSimulation Method")

title = "Distribution of Lillard's Points Scored\n2019-2020 Regular Season"
data = df_reg_szn.assign(Bubble=lambda x: x.bubble.map({1: "Yes", 0: "No"}))
ax = sns.histplot(data, x='PTS', hue='Bubble')
ax.set(title=title, xlabel="Points", ylabel="Game Counts")
regszn_pts_avg, bubble_pts_avg = df_reg_szn.groupby('bubble').PTS.mean()
plt.show()

mask = (df.bubble == 0) & (df.playoffs == 0)
s = df.loc[mask, 'PTS'].rolling(8).mean()
s[s > 37].count()


mask = (df.bubble == 1) & (df.playoffs == 0)
df.loc[mask, ].shape
tester.t_stat
