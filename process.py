# process.py

import pandas as pd
import numpy as np

def load_and_process_data(matches_path, ball_by_ball_path):
    """Loads raw data and returns a clean DataFrame ready for modeling."""
    
    ball_by_ball_df = pd.read_csv(ball_by_ball_path)
    matches_df = pd.read_csv(matches_path)

    # Calculate first innings total runs for each match
    total_runs_df = ball_by_ball_df.groupby(['ID', 'innings']).sum(numeric_only=True)['total_run'].reset_index()
    first_innings_runs = total_runs_df[total_runs_df['innings'] == 1]
    first_innings_runs.rename(columns={'total_run': 'target_score'}, inplace=True)

    # Merge match data with the first innings score to get a target
    match_df = matches_df.merge(first_innings_runs[['ID', 'target_score']], on='ID')

    # Standardize team names
    match_df['Team1'] = match_df['Team1'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['Team2'] = match_df['Team2'].str.replace('Delhi Daredevils', 'Delhi Capitals')
    match_df['Team1'] = match_df['Team1'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')
    match_df['Team2'] = match_df['Team2'].str.replace('Deccan Chargers', 'Sunrisers Hyderabad')

    # Merge the ball-by-ball data with the cleaned match data for the second innings
    second_innings_df = match_df.merge(ball_by_ball_df[ball_by_ball_df['innings'] == 2], on='ID')

    # Feature Engineering
    second_innings_df['current_score'] = second_innings_df.groupby('ID').cumsum(numeric_only=True)['total_run']
    second_innings_df['runs_left'] = second_innings_df['target_score'] - second_innings_df['current_score']
    second_innings_df['balls_left'] = 120 - (second_innings_df['overs'] * 6 + second_innings_df['ballnumber'])
    second_innings_df['wickets_fallen'] = second_innings_df.groupby('ID').cumsum(numeric_only=True)['isWicketDelivery']
    second_innings_df['wickets_left'] = 10 - second_innings_df['wickets_fallen']
    second_innings_df['overs_completed'] = second_innings_df['overs'] + (second_innings_df['ballnumber'] / 6)
    second_innings_df['crr'] = (second_innings_df['current_score'] * 6) / (second_innings_df['overs_completed'] * 6)
    second_innings_df['rrr'] = np.where(second_innings_df['balls_left'] == 0, 999, (second_innings_df['runs_left'] * 6) / second_innings_df['balls_left'])
    second_innings_df.replace([np.inf, -np.inf], 999, inplace=True)
    second_innings_df['crr'].fillna(0, inplace=True)

    # Determine bowling team and result
    second_innings_df['BowlingTeam'] = second_innings_df.apply(lambda row: row['Team2'] if row['BattingTeam'] == row['Team1'] else row['Team1'], axis=1)
    second_innings_df['result'] = np.where(second_innings_df['BattingTeam'] == second_innings_df['WinningTeam'], 1, 0)
    
    return second_innings_df.dropna()