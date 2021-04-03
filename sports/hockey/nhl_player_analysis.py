# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 11:56:22 2020

@author: Brendan Non-Admin
"""

import os
import json
import pandas as pd

filename = os.path.join('data', 'nhl_player_data_1917-2020.csv')
data = pd.read_csv(filename)

(data.assign(
    decade=lambda x: x.year_season_start.apply(lambda x: 10 * round(x/10))
    )
  .groupby('decade').year_season_start.count()
)

def pre_proc(input_df):
    df = input_df.copy()
    
    df = df.sort_values(['skaterFullName', 'seasonId'])
    
    df.loc[:, 'year_season_start'] = df.seasonId.astype(str).str[:4].astype('int64')
    df.loc[:, 'year_season_end'] = df.seasonId.astype(str).str[4:].astype('int64')
    
    df.loc[:, 'season_number'] = (df
        .sort_values(['skaterFullName', 'seasonId'])
        .groupby(['skaterFullName'])
        ['year_season_start'].rank(method="first")
    )
    
    df.loc[:, 'years_played'] = (df.groupby('skaterFullName')
        ['season_number'].transform('max')
    )
    
    df = df.sort_values(['skaterFullName', 'season_number'])

    return df

data = data.pipe(pre_proc)

def all_time_ranking(metric):
    return (metric.sort_values(ascending=False)
            .rank(method="dense", ascending=False)
           )

def create_df_players(input_df):
    df = input_df.copy()
    df = (df
       .groupby('skaterFullName')
       .agg(
           points=('points', 'sum'),
           goals=('goals', 'sum'),
           assists=('assists', 'sum'),
           gamesPlayed=('gamesPlayed', 'sum'),
           rookie_year=('year_season_start', 'min'),
           retirement_year=('year_season_end', 'max'),
           teams=('teamAbbrevs', lambda x: ', '.join(x.unique()))
       )
       .assign(
           points_ranking=lambda x: all_time_ranking(x.points),
           goals_ranking=lambda x: all_time_ranking(x.goals),
           assists_ranking=lambda x: all_time_ranking(x.assists),
           gamesPlayed_ranking=lambda x: all_time_ranking(x.gamesPlayed),
           years_played=lambda x: x.retirement_year - x.rookie_year
       )
    )
    return df

df_players = create_df_players(data)

player_name = 'Mark Messier'
mask = data.skaterFullName == player_name
data.loc[mask, 'playerId'].unique()

player_name = 'Mark Messier'
mask = data.skaterFullName == player_name
columns = [
    'skaterFullName',
    'year_season_start',
    'teamAbbrevs',
    'positionCode',
    'gamesPlayed',
    'goals', 
    'assists',
    'faceoffWinPct', 
    'gameWinningGoals',
    'otGoals',
    'penaltyMinutes',
    'plusMinus',
    'points',
    'ppGoals',
    'ppPoints',
    'shGoals',
    'shPoints', 
    'shootingPct',
    'shots',
    'timeOnIcePerGame',
]
df = data.loc[mask, columns]

(df
 .loc[:, ['skaterFullName', 'goals']]
 .assign(
     goals_lag1Y=lambda x: x.goals.shift(1),
     goals_lag2Y=lambda x: x.goals.shift(2),
     goals_lag3Y=lambda x: x.goals.shift(3),
     goals_lag4Y=lambda x: x.goals.shift(4),
     goals_lag5Y=lambda x: x.goals.shift(5)
 )
 .to_excel('mark_messier_goals_lagged.xlsx')
)


def create_df_yearly_avgs(input_df):
    df = (input_df
        .copy()
        .groupby('year_season_start')
        .agg(
            league_goals_average=('goals', 'mean'),
            league_assists_average=('assists', 'mean'),
       )
    )
    
    return df

df_yearly_avgs = create_df_yearly_avgs(data)

def calc_goals_and_assists_by_year(input_df, to_dict=False):
    df = input_df.copy()
    
    df = (df.set_index('year_season_start')
     .loc[:, ['goals', 'assists']]
     .join(df_yearly_avgs)
    )
    
    if to_dict:
        df = df.reset_index().to_dict(orient='records')
    
    return df

goals_and_assists_by_year = df.pipe(calc_goals_and_assists_by_year, True)

def calc_games_by_team(input_df, to_dict=False):
    df = input_df.copy()
    df = df.groupby('teamAbbrevs').gamesPlayed.sum()
    
    if to_dict:
        df = df.to_dict()
        
    return df

games_by_team = df.pipe(calc_games_by_team, True)

def calc_career_data(df_players, player_name, to_dict=False):
    player_data = df_players.loc[player_name, ]
    
    if to_dict:
        player_data_dict = player_data.to_dict()
        player_data = {'skaterFullName': player_name}
        player_data.update(player_data_dict)
        return player_data
    
    return player_data

career_data = df_players.pipe(calc_career_data, 'Mark Messier', True)

def create_photo_url(player_name):
    mask = data.skaterFullName == player_name
    player_id = data.loc[mask, 'playerId'].unique()[0]
    
    url_str = 'https://nhl.bamcontent.com/images/headshots/current/168x168/{}.jpg'
    
    return url_str.format(player_id)

photo_url = create_photo_url(player_name)

player_data = {
    'career_data': career_data,
    'goals_and_assists_by_year': goals_and_assists_by_year,
    'games_by_team': games_by_team,
    'photo_url': photo_url
}

json.dumps(player_data)