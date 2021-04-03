# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 10:16:20 2020

@author: Brendan Non-Admin

source_url: https://stats.nba.com/players/bio/?Season=2019-20&SeasonType=Regular%20Season

background: I downloaded html file to use since testing the API directly didn't work

"""

import re
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import scipy.stats.stats
plt.style.use('ggplot')

filename = 'NBA.com_Stats _ Players Bios.html'
filepath = path.join('data', filename)

with open(filepath, 'r') as f:
    df = (pd.read_html(f)[0]
        .set_index('Player')
    )
    
# sanity check for players per team (should be around 15)
players_estimate = len(df) / 30
print(f"{players_estimate:0,.2f}")

##############################################################################
# check all heights match pattern
##############################################################################
pat = re.compile(r'\d-\d{1,2}')

mask = ~df.Height.str.match(pat)
assert(df.loc[mask, 'Height'].empty)

df.loc[:, 'Height (Inches)'] = (df.Height
    .str.split('-')
    .apply(lambda x: int(x[0]) * 12 + int(x[1]))
)

ax = df.plot.scatter(x='Height (Inches)', y='Weight')
ax.set(title='NBA Player BioMetrics')
plt.show()

##############################################################################
# scaling
##############################################################################
from sklearn.preprocessing import StandardScaler

height_scaler = StandardScaler()
x = df.loc[:, ['Height (Inches)']]
df.loc[:, 'height_scaled'] = height_scaler.fit_transform(x)

weight_scaler = StandardScaler()
x = df.loc[:, ['Weight']]
df.loc[:, 'weight_scaled'] = weight_scaler.fit_transform(x)

df.plot.scatter(x='height_scaled', y='weight_scaled')

##############################################################################
# dbscan
##############################################################################
from sklearn.cluster import DBSCAN

X = df.loc[:, ['height_scaled', 'weight_scaled']]

db = DBSCAN().fit(X)

df.loc[:, 'DBSCAN Results'] = (pd.Categorical(db.labels_)
    .rename_categories({-1: 'Outlier', 0: 'Core Data'})
)

mask = df.loc[:, 'DBSCAN Results'] == 'Outlier'
df.loc[mask, ['Height', 'Weight']]

plot_kwargs = dict(
    x='Height (Inches)',
    y='Weight', 
    hue='DBSCAN Results',
    data=df,
    palette=sns.color_palette("RdBu", n_colors=2)[::-1]
)
ax = sns.scatterplot(**plot_kwargs)
ax.set(title='NBA Player BioMetrics')
plt.show()

##############################################################################
# Isolation Forests
##############################################################################
from sklearn.ensemble import IsolationForest

X = df.loc[:, ['height_scaled', 'weight_scaled']]

clf = IsolationForest(contamination=0.01).fit(X)

df.loc[:, 'IsolationForest Results'] = (pd.Categorical(clf.predict(X))
    .rename_categories({-1: 'Outlier', 1: 'Core Data'})
)

mask = df.loc[:, 'IsolationForest Results'] == 'Outlier'
df.loc[mask, ['Height', 'Weight']]

mask = df.loc[:, 'IsolationForest Results'] == 'Outlier'
df.loc[mask, ['Height', 'Weight']]

plot_kwargs = dict(
    x='Height (Inches)', y='Weight', hue='IsolationForest Results', data=df,
    palette=sns.color_palette("RdBu", n_colors=2)[::-1]
)
ax = sns.scatterplot(**plot_kwargs)
ax.set(title='NBA Player BioMetrics')
plt.show()

##############################################################################
# PCA with z-score
##############################################################################
from sklearn.decomposition import PCA

columns = ['height_scaled', 'weight_scaled']
X = df.loc[:, columns]

pca = PCA(n_components=1)
pca.fit(X)

X_trans = pca.transform(X)

df.loc[:, 'Height-Weight PCA'] = X_trans

a = df.loc[:, 'Height-Weight PCA']
z_scores = stats.zscore(a)

# 1.960 is 95% confidence interval for standard normal distribution
# 2.576 is 99% confidence interval for standard normal distribution
z_score_outliers = (np.abs(z_scores) >= 1.960) * 1

df.loc[:, 'PCA Z-Score Results'] = (pd.Categorical(z_score_outliers)
    .rename_categories({1: 'Outlier', 0: 'Core Data'})
)

bool(0)

mask = df.loc[:, 'PCA Z-Score Results'] == 'Outlier'
sorted(df.loc[mask, :].index.tolist())

plot_kwargs = dict(
    x='Height (Inches)', y='Weight', hue='PCA Z-Score Results', data=df,
    palette=sns.color_palette("RdBu", n_colors=2)[::-1]
)
ax = sns.scatterplot(**plot_kwargs)
ax.set(title='NBA Player BioMetrics')
plt.show()

##############################################################################
# Write to file
##############################################################################

columns = [
    'Team',
    'Age',
    'Height',
    'Weight',
    'College',
    'Country',
    'Draft Year',
    'Draft Round',
    'Draft Number', 
    'Height (Inches)',
    'DBSCAN Results',
    'IsolationForest Results', 
    'Height-Weight PCA', 
    'PCA Z-Score Results'
]
filename = path.join('data', 'nba_biometrics_analysis.csv')
df.to_csv(filename)