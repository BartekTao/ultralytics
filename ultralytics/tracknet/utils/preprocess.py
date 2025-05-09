import numpy as np
import pandas as pd


def preprocess_csv(csv_file):
    # Read the ball_trajectory csv file
    ball_trajectory_df = pd.read_csv(csv_file)
    ball_trajectory_df['nX'] = ball_trajectory_df['X'].shift(-1).fillna(ball_trajectory_df['X'])
    ball_trajectory_df['nY'] = ball_trajectory_df['Y'].shift(-1).fillna(ball_trajectory_df['Y'])

    if 'Event' in ball_trajectory_df.columns:
        ball_trajectory_df['hit'] = ((ball_trajectory_df['Event'] == 1) | (ball_trajectory_df['Event'] == 2)).astype(int)
    else:
        ball_trajectory_df['hit'] = 0

    drop_columns = ['Fast', 'Event', 'Z', 'Shot', 'player_X', 'player_Y', 'prev_hit', 'next_hit', 'Timestamp']
    
    ball_trajectory_df = ball_trajectory_df.drop(drop_columns, axis=1, errors='ignore')

    return ball_trajectory_df