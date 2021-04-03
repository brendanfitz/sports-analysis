#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:23:27 2020
@author: BFitzpatrick
"""

import os
import json
import numpy as np
import pandas as pd
import requests
import datetime as dt
from bs4 import BeautifulSoup

def main():
    url = 'https://www.hockey-reference.com/leagues/NHL_2020_games.html'
    response = requests.get(url)
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = str(soup.find('table', {'id': 'games'}))
    df = pd.read_html(table)[0]
     
    colmap = {
        'Date': 'date',
        'Visitor': 'away_team',
        'G': 'away_goals',
        'Home': 'home_team',
        'G.1': 'home_goals',
        'Unnamed: 5': 'extra_time',
        'Att.': 'attendance',
        'LOG': 'game_length',
        'Notes': 'notes',
    }
    df = (df.rename(columns=colmap)
        .assign(date=lambda x: pd.to_datetime(x.date))
        .pipe(filter_unplayed_games)
        .set_index('date', append=True)
        .rename_axis(["game_id", "date"])
        .assign(home_win=lambda x: (x.home_goals > x.away_goals),
                away_win=lambda x: (x.home_goals < x.away_goals))
    )
        
    df_teams = create_df_teams(df)
    
    df_full = create_df_full(df_teams)
    
    df_wildcard = create_df_wildcard(df_full)
    
    df_full = (df_full.join(df_wildcard)
        .assign(wildcard=lambda x: x.wildcard.fillna('No'))
    )
    
    records = df_full_to_records(df_full)
        
    ds = dt.datetime.today().strftime('%y%m%d')
    teams_fout = os.path.join(
        'data',
        f"nhl_results_{ds}.json",
    )
    with open(teams_fout, 'w') as f:
        json.dump(records, f, indent=4)
       
    print("NHL Point Data Scrape Successful!"
          f"\nSee files located at {teams_fout}")

   
def points_calc(win, extra_time):
    extra_time_loss = ~win & ~extra_time.isnull()
    loss_points = np.where(extra_time_loss, 1, 0)
    points = np.where(win, 2, loss_points)
    return points

def filter_unplayed_games(input_df):
    df = input_df.copy()
    today = dt.date.today().isoformat()
    mask = df.date < today
    return df.loc[mask, ]

def create_df_teams(df):
    home_cols = ['home_team', 'home_win', 'extra_time']
    away_cols = ['away_team', 'away_win', 'extra_time']
    
    frames = list()
    for cols in [home_cols, away_cols]:
        df_team = (df.loc[:, cols]
            .rename(columns={
                'home_team': 'team',
                'away_team': 'team',
                'home_win': 'win',
                'away_win': 'win',
            })
            .set_index('team', append=True)
        )
        frames.append(df_team)
    
    df_teams = (pd.concat(frames)
        .assign(points=lambda x: points_calc(x.win, x.extra_time))
        .sort_values(by=['team', 'date'])
        .assign(team_game_id=lambda x: x.groupby('team').cumcount() + 1,
                total_points=lambda x: x.groupby('team')['points'].cumsum())
    )
    return df_teams

def create_index_from_interpolation(df_teams):
    season_start = df_teams.index.get_level_values(1).min()
    season_end = df_teams.index.get_level_values(1).max()
    dates = pd.date_range(season_start, season_end, freq='D').to_list()
    teams = df_teams.index.get_level_values(2).unique().to_list()
    iterables = [dates, teams]
    index = pd.MultiIndex.from_product(iterables, names=['date', 'team'])
    return index

def create_df_full(df_teams):    
    index = create_index_from_interpolation(df_teams)
    df_teams_temp = (df_teams
        .reset_index(level=0)
        .rename(columns={'team_game_id': 'games_played'})
        .loc[:, ['games_played', 'total_points']]
        .rename(columns={'total_points': 'points'})
    )
    df_merged = pd.DataFrame(index=index).merge(df_teams_temp, how='left', left_index=True, right_index=True)
    df_merged[['games_played', 'points']] = df_merged.groupby(['team']).ffill().fillna(0)
    return df_merged

def df_full_to_records(df_full):
    records = list()
    for date in df_full.index.get_level_values(level=0).unique().to_list():
        record = {
            'date': date.strftime('%Y-%m-%d'),
            'teams': df_full.xs(date).reset_index().to_dict(orient='records')
        }
        records.append(record)
    return records


def create_df_wildcard(df_full):
    df = df_full.copy()
    team_data = pd.read_csv(os.path.join('data', 'nhl_team_data.csv')).set_index('team')
    df = df.join(team_data.drop('color', axis=1), how='left', on='team')
    df = df.sort_values(by=['date', 'division', 'points'], ascending=[True, True, False])
    
    df['division_rank'] = (df.sort_values(by=['date', 'division', 'points'], ascending=[True, True, False])
     .groupby(['date', 'division']).points.rank(method='first', ascending=False)
    )
    
    mask = df.division_rank > 3
    df_wildcard = df.loc[mask, ].copy()
    df_wildcard = df_wildcard.sort_values(by=['date', 'conference', 'points'], ascending=[True, True, False])
    df_wildcard['conference_rank'] = (df_wildcard.sort_values(by=['date', 'conference', 'points'], ascending=[True, True, False])
     .groupby(['date', 'conference']).points.rank(method='first', ascending=False)
    )
    
    mask = df_wildcard.conference_rank == 2
    df_wildcard = (df_wildcard.loc[mask, ]
     .drop(['games_played', 'division', 'division_rank', 'conference_rank', 'points'], axis=1)
     .rename(columns={'conference': 'wildcard'})
    )
    return df_wildcard
    

if __name__ == '__main__':
    main()
