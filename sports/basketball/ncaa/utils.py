# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from sports.basketball.ncaa.ncaa_tourney_games_spider import START_YEAR, END_YEAR
from bs4 import BeautifulSoup

def data_checks(df):
    # check games total
    if not(63 * (END_YEAR - START_YEAR + 1) == df.shape[0]):
        return {"result": "failed", "test": "games total"}
    
    # no ties
    if not(df.loc[df.home_score == df.away_score, ].empty):
        return {"result": "failed", "test": "no ties"}
    
    return {"result": "passed"}

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
    g = sns.FacetGrid(df_dists, row="matchup", hue="matchup", aspect=15, height=.6, palette=pal, sharey=False)
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

def print_round(df, year, round_):
    mask = (df.loc[:, 'year'] == year) & (df.loc[:, 'round'] == round_)
    columns = ['region', 'home_seed', 'home_team', 'home_score', 'away_seed', 'away_team', 'away_score', 'winner']
    g = df.loc[mask, columns].groupby('region')
    
    for region, results in g:
        print(region.center(80)+ '\n' + '*' * 80)
        for idx, row in results.iterrows():
            if row['winner'] == 'home_team':
                winning_team = row['home_team']
                winning_score = row['home_score']
                winning_seed = row['home_seed']
                losing_team = row['away_team']
                losing_score = row['away_score']
                losing_seed = row['away_seed']
            else:
                winning_team = row['away_team']
                winning_score = row['away_score']
                winning_seed = row['away_seed']
                losing_team = row['home_team']
                losing_score = row['home_score'] 
                losing_seed = row['home_seed']
            
            winning_team_and_seed = f"{winning_team} ({winning_seed})"
            losing_team_and_seed = f"{losing_team} ({losing_seed})"
            print(f"{winning_team_and_seed:30} {winning_score:3} | {losing_team_and_seed:30} {losing_score:3} | FINAL")
        print('\n\n')
        


def create_df_first_rounds(df, models):
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
            confidence_interval_band=lambda x: x.higher_seed_win_pct_upper_bound - x.higher_seed_win_pct_lower_bound,
            
        )
    )
    
    df_first_rounds.columns.name = None
    
    return df_first_rounds

    
def to_html(df, filepath):
    html = df.to_html(justify="left")
    
    soup = BeautifulSoup(html, 'html.parser')
    
    table = soup.find('table')
    table['class'] = 'table table-striped border border-secondary'
    del table.attrs['border']
    
    thead = soup.find('thead')
    thead['class'] = 'thead-dark'
    
    trs = thead.find_all('tr')
    
    if len(trs) > 1:
        tr1, tr2 = trs
        tr1.find('th').string = tr2.find('th').string.replace('_', ' ').title()
        tr2.decompose()
    
    html = soup.prettify()
    
    with open(filepath, 'w') as f:
        f.write(html)