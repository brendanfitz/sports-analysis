# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter


def seed_result(row):
    if row['home_seed'] == row['away_seed']:
        return 'equal_seeds'
    
    if (
            (
                row['winner']  == 'home_team' 
            and row['home_seed'] < row['away_seed']
            )
        or            
            (
                row['winner']  == 'away_team' 
            and row['away_seed'] < row['home_seed']
            )
        ):
        return 'higher_seed_win'
    
    return 'lower_seed_win'

def seed_matchup(row):
    seeds = sorted(
        [
            str(row['home_seed']).zfill(2), 
            str(row['away_seed']).zfill(2)
        ]
    )
    
    return 'v'.join(seeds)


def numeric_col_formatter(column):
    if column.dtype == np.float64:
        return column.map('{:,.1%}'.format)
    
    return column

def RidgePlot(models):
    x = np.arange(0, 1.01, 0.001)
    dists = { k: v.posterior_dist.pdf(x) for k, v in models.items() }
    
    
    df_dists = (pd.DataFrame(dists)
        .assign(x=x)
        .melt(id_vars='x', var_name='matchup', value_name='pdf')
        .sort_values(['matchup', 'x'])
    )
    
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    
    
    def fill_between(x, y, color, label):
        ax = plt.gca()
        ax.fill_between(x=x, y1=0, y2=y, color=color)
    
    
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df_dists, row="matchup", hue="matchup", aspect=15, height=.5, palette=pal, sharey=False)
    g.map(sns.lineplot, 'x', 'pdf', clip_on=False)
    
    g.map(label, "x")
    
    for ax in g.axes[:, 0]:
        # ax.set_xlim(0, 1.05)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1))
    
    g.map(fill_between, "x", "pdf")
    
    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)
        
    g.set_titles("")
    g.set_xlabels("")
    g.set_ylabels("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    font_kwargs = dict(fontname='verdana', fontsize=14, fontstyle='italic', fontweight='bold')
    g.fig.suptitle("Bayesian Posterior Distributions of Higher Seed Win Probability", **font_kwargs)
    
    return g