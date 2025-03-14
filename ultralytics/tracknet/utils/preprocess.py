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

    #ball_trajectory_df['prev_hit'] = ball_trajectory_df['hit'].shift(fill_value=0)
    #ball_trajectory_df['next_hit'] = ball_trajectory_df['hit'].shift(-1, fill_value=0)
    #ball_trajectory_df['hit'] = ball_trajectory_df[['hit', 'prev_hit', 'next_hit']].max(axis=1)
    if 'Visibility' in ball_trajectory_df.columns:
        visibility = ball_trajectory_df['Visibility'].values
        x_vals = ball_trajectory_df['X'].values
        y_vals = ball_trajectory_df['Y'].values
        
        movement_threshold_3 = 3.0  # 判定靜止的最大首尾位移（pixel）
        movement_threshold_5 = 5.0  # 判定靜止的最大首尾位移（pixel）
        
        # 找出所有 Visibility==1 的索引
        vis_indices = np.where(visibility == 1)[0]
        if vis_indices.size > 0:
            # (a) 第一段：從第一個出現球的索引開始，向後找出連續區段
            for i in range(len(vis_indices)-1):
                dx = x_vals[vis_indices[i]] - x_vals[vis_indices[i+1]]
                dy = y_vals[vis_indices[i]] - y_vals[vis_indices[i+1]]
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= movement_threshold_5:
                    ball_trajectory_df.loc[vis_indices[i]:vis_indices[i+1], 'Visibility'] = 0
                else:
                    break

            # (b) 最後一段：從最後一個出現球的索引開始，向前找出連續區段
            # 處理最後一段：從最後一個出現的索引往回找
            for i in range(len(vis_indices)-1, 0, -1):
                dx = x_vals[vis_indices[i]] - x_vals[vis_indices[i-1]]
                dy = y_vals[vis_indices[i]] - y_vals[vis_indices[i-1]]
                dist = np.sqrt(dx**2 + dy**2)
                if dist <= movement_threshold_5:
                    ball_trajectory_df.loc[vis_indices[i-1]:vis_indices[i], 'Visibility'] = 0
                else:
                    break

    else:
        print("Warning: 'Visibility' column not found in CSV.")

    drop_columns = ['Fast', 'Event', 'Z', 'Shot', 'player_X', 'player_Y', 'prev_hit', 'next_hit', 'Timestamp']
    
    ball_trajectory_df = ball_trajectory_df.drop(drop_columns, axis=1, errors='ignore')

    return ball_trajectory_df