"""
NBA fantasy predictions using trained SAINT model.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Tuple
import logging

from src.models.saint import SAINT
from draftkings import get_draft_groups, clean_name
from nba_stats import get_players
from prediction_utils import optimize_lineup

def predict_game_stats(model_path, game_data, device= None):
    """
    Generate game statistics predictions using trained model.
    
    Args:
        model_path: Path to saved model checkpoint
        game_data: DataFrame containing player and game information
        device: Computing device to use
    
    Returns:
        DataFrame with predicted statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model with config from checkpoint
    model = SAINT(
        categories=checkpoint['cat_dims'],
        num_continuous=len(checkpoint['con_idxs'])
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare input data
    cat_data = []
    for idx in checkpoint['cat_idxs']:
        col_data = game_data.iloc[:, idx].astype('category').cat.codes.values
        cat_data.append(col_data)

    x_categorical = torch.tensor(np.stack(cat_data, axis=1), dtype=torch.long)
    x_continuous = torch.tensor(game_data.iloc[:, checkpoint['con_idxs']].values, dtype=torch.float32)
    
    # Normalize continuous features
    if 'train_mean' in checkpoint and 'train_std' in checkpoint:
        x_continuous = (x_continuous - checkpoint['train_mean']) / checkpoint['train_std']

    # Add batch dimension and move to device
    x_categorical = x_categorical.unsqueeze(0).to(device)
    x_continuous = x_continuous.unsqueeze(0).to(device)

    # Get predictions
    with torch.no_grad():
        predictions = model(x_categorical, x_continuous)
        predictions = predictions.squeeze(0).cpu().numpy()
        
    # Unscale predictions if scales exist
    if 'y_scales' in checkpoint:
        predictions = predictions * checkpoint['y_scales']

    # Convert to DataFrame
    stats_columns = ['points', 'threePointersMade', 'steals', 'blocks', 
                    'turnovers', 'rebounds', 'assists']
    predictions_df = pd.DataFrame(predictions, columns=stats_columns)
    
    return predictions_df.clip(lower=0)

def matchup(game_id, df_all, out_players, m_players= 6):
    """
    Create matchup data for a specific game.
    
    Args:
        game_id: Game identifier (e.g., 'BOS @ LAL')
        df_all: DataFrame with all player data
        out_players: List of players who are out
        m_players: Number of players to select per team
    """
    df_all = df_all[~df_all['name'].isin(out_players)]
    team_id_home, team_id_away = game_id.split(' @ ')
    
    team_home = df_all[df_all['team_id'] == team_id_home].copy()
    team_home.loc[:, 'home?'] = 1
    team_away = df_all[df_all['team_id'] == team_id_away]
    
    team_home = team_home.nlargest(m_players, 'minutes')
    team_away = team_away.nlargest(m_players, 'minutes')
    
    both_teams = pd.concat([team_home, team_away]).sample(frac=1).reset_index(drop=True)
    return both_teams

def player_points(player):
    """Calculate DraftKings points for a player."""
    stat_cols = ['points', 'rebounds', 'assists', 'blocks', 'steals']
    double_double = int(sum(player[stat_cols] >= 10) >= 2)
    triple_double = int(sum(player[stat_cols] >= 10) >= 3)
    
    cols = ['points', 'threePointersMade', 'rebounds', 'assists', 'steals', 'blocks', 'turnovers']
    weights = [1, 0.5, 1.25, 1.5, 2, 2, -0.5]
    points = (player[cols] * weights).sum() + 1.5 * double_double + 3 * triple_double
    
    player['Player Points'] = points
    return player

def predict_matchup(model_path, game_id, df_all, 
                   out_players, m_players, 
                   n_games=1):
    """
    Predict player performance for a specific matchup.
    
    Args:
        model_path: Path to trained model
        game_id: Game identifier
        df_all: DataFrame with all player data
        out_players: List of players who are out
        m_players: Number of players to select per team
        n_games: Number of recent games to consider
    """
    # Get updated player data
    df_all = get_players(n_games=n_games)
    
    # Process matchup
    game = matchup(game_id, df_all, out_players, m_players)
    game['Player Points'] = 0
    
    # Make predictions
    cols_input = ['points', 'threePointersMade', 'steals', 'blocks', 'turnovers', 'assists', 
                  'rebounds', 'weight', 'height', 'age', 'years_in_league', 'home?', 
                  'draft_round', 'position']
    game_data = game.loc[:, cols_input]
    
    predictions = predict_game_stats(model_path, game_data)
    game.loc[:, predictions.columns] = predictions
    
    # Calculate player points
    game = game.apply(player_points, axis=1)
    return game.sort_values('Player Points', ascending=False)

def automate_predictions(model_path, n_games, m_players= 6):
    """
    Generate predictions for all draft groups.
    
    Args:
        model_path: Path to trained model
        n_games: Number of recent games to consider
        m_players: Number of players to select per team
        
    Returns:
        Tuple containing:
        - Dictionary of optimized results
        - Dictionary of draft group data
        - Dictionary of game predictions
        - Total predictions DataFrame
    """
    logging.info("Getting player data...")
    df_all = get_players(n_games=n_games)
    
    logging.info("Forming draft groups...")
    group_list = get_draft_groups()
    
    # Initialize storage
    out_players = []
    game_ids = set()
    draft_group_dic = {}
    results_dic = {}
    game_dic = {}

    # Process draft groups
    for group in group_list:
        draft_group = str(group['Draft Group'][0])
        draft_group_dic[draft_group] = []
        game_ids.update(group['Game'].unique())
        out_players.extend(group.loc[group['Status'] == 'OUT', 'Name'])

    out_players = set(out_players)
    
    logging.info("Making predictions...")
    for game_id in game_ids:
        logging.info(f"Predicting {game_id}...")
        result = predict_matchup(model_path, game_id, df_all, out_players, m_players)
        game_dic[game_id] = result
    
    # Process results for each draft group
    for group in group_list:
        draft_group = str(group['Draft Group'][0])
        games = group['Game'].unique()
        
        # Combine predictions
        total_df = pd.concat([game_dic[game_id][['name', 'Player Points']] 
                            for game_id in games], ignore_index=True)
        
        # Match with draft group
        total_df['Clean Name'] = total_df['name'].apply(clean_name)
        group['Clean Name'] = group['Name'].apply(clean_name)
        
        merge_df = pd.merge(total_df, group, on='Clean Name', how='inner')
        final_df = merge_df[['Name', 'Position', 'Salary', 'Player Points']]
        
        # Calculate value and store results
        final_df['Value'] = final_df['Player Points'] / final_df['Salary'] * 1000
        draft_group_dic[draft_group] = final_df
        
        # Optimize lineup if valid predictions exist
        final_df = final_df[final_df['Player Points'] != 0].copy()
        if not final_df.empty:
            result = optimize_team_ilp(final_df)
            results_dic[draft_group] = result
    
    logging.info("Predictions completed")
    return draft_group_dic

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = config['paths']['best_model']
    results = automate_predictions(model_path)