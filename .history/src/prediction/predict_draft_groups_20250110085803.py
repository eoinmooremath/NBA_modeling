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
import sys
import os 
from datetime import datetime

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


from src.models.saint import SAINT
from draft_kings import get_draft_groups, clean_name
from nba_stats import get_players
from prediction_utils import optimize_lineup #, predict_game_stats

# def predict_game_stats(model_path, game_data, device= None):
#     """
#     Generate game statistics predictions using trained model.
    
#     Args:
#         model_path: Path to saved model checkpoint
#         game_data: DataFrame containing player and game information
#         device: Computing device to use
    
#     Returns:
#         DataFrame with predicted statistics
#     """
#     if device is None:
#         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
#     # Load checkpoint
#     checkpoint = torch.load(model_path, map_location=, weights_only=True)
    
#     # Initialize model with config from checkpoint
#     model = SAINT(
#         categories=checkpoint['cat_dims'],
#         num_continuous=len(checkpoint['con_idxs'])
#     ).to(device)
    
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()

#     # Prepare input data
#     cat_data = []
#     for idx in checkpoint['cat_idxs']:
#         col_data = game_data.iloc[:, idx].astype('category').cat.codes.values
#         cat_data.append(col_data)

#     x_categorical = torch.tensor(np.stack(cat_data, axis=1), dtype=torch.long)
#     x_continuous = torch.tensor(game_data.iloc[:, checkpoint['con_idxs']].values, dtype=torch.float32)
    
#     # Normalize continuous features
#     if 'train_mean' in checkpoint and 'train_std' in checkpoint:
#         x_continuous = (x_continuous - checkpoint['train_mean']) / checkpoint['train_std']

#     # Add batch dimension and move to device
#     x_categorical = x_categorical.unsqueeze(0).to(device)
#     x_continuous = x_continuous.unsqueeze(0).to(device)

#     # Get predictions
#     with torch.no_grad():
#         predictions = model(x_categorical, x_continuous)
#         predictions = predictions.squeeze(0).cpu().numpy()
        
#     # Unscale predictions if scales exist
#     if 'y_scales' in checkpoint:
#         predictions = predictions * checkpoint['y_scales']

#     # Convert to DataFrame
#     stats_columns = ['points', 'threePointersMade', 'steals', 'blocks', 
#                     'turnovers', 'rebounds', 'assists']
#     predictions_df = pd.DataFrame(predictions, columns=stats_columns)
    
#     return predictions_df.clip(lower=0)


def predict_game_stats(model_path, game_data, config, device= None):
    """
    Generate game statistics predictions using trained model.
    
    Args:
        model_path: Path to saved model checkpoint
        game_data: DataFrame containing player and game information
        config: Configuration dictionary
        device: Computing device to use
    
    Returns:
        DataFrame with predicted statistics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Get model config
    saint_config = config['model']['saint']
    
    # Initialize model with config parameters
    model = SAINT(
        categories=checkpoint['cat_dims'],
        num_continuous=len(checkpoint['con_idxs']),
        dim=saint_config['dim'],
        depth=saint_config['depth'],
        num_heads=saint_config['num_heads'],
        output_dim=saint_config['output_dim'],
        num_players=saint_config['num_players'],
        head_dim=saint_config['head_dim'],
        mlp_hidden_factors=saint_config['mlp_hidden_factors'],
        continuous_embedding_type=saint_config['continuous_embedding_type'],
        # Use inference dropout rates (all 0)
        attention_dropout=saint_config['inference_dropout']['attention'],
        ffn_dropout=saint_config['inference_dropout']['ffn'],
        mlp_dropout=saint_config['inference_dropout']['mlp']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    try:
        # Process input data
        categorical_columns = ['home?', 'draft_round', 'position']
        numeric_columns = [col for col in game_data.columns if col not in categorical_columns]

        # Process categorical features
        cat_data = []
        cat_dims = checkpoint.get('cat_dims', None)
        
        if cat_dims is not None:
            for col, dim in zip(categorical_columns, cat_dims):
                col_data = game_data[col].astype('category').cat.codes.values
                col_data = np.clip(col_data, 0, dim - 1)
                cat_data.append(col_data)
        else:
            # Fallback processing if cat_dims not available
            for col in categorical_columns:
                col_data = game_data[col].astype('category').cat.codes.values
                cat_data.append(col_data)

        x_categorical = np.stack(cat_data, axis=1)
        x_categorical = torch.tensor(x_categorical, dtype=torch.long)

        # Process continuous features with normalization
        x_continuous = game_data[numeric_columns].values.astype(np.float32)
        
        # Apply normalization if parameters exist
        train_mean = checkpoint.get('train_mean')
        train_std = checkpoint.get('train_std')
        y_scales = checkpoint.get('y_scales')
        if y_scales is None:
            raise ValueError("No y_scales found in checkpoint")
        
        if train_mean is not None and train_std is not None:
            # Convert to numpy if they're tensors
            if torch.is_tensor(train_mean):
                train_mean = train_mean.cpu().numpy()
            if torch.is_tensor(train_std):
                train_std = train_std.cpu().numpy()
            
            # Apply normalization
            x_continuous = (x_continuous - train_mean) / (train_std + 1e-8)  # Add epsilon for stability

        x_continuous = torch.tensor(x_continuous, dtype=torch.float32)

        # Add batch dimension and move to device
        x_categorical = x_categorical.unsqueeze(0).to(device)
        x_continuous = x_continuous.unsqueeze(0).to(device)

        # Get predictions
        with torch.no_grad():
            predictions = model(x_categorical, x_continuous)
            predictions = predictions.squeeze(0)
            
            # Move to CPU and convert to numpy
            predictions = predictions.cpu().numpy()
            predictions = predictions * y_scales
        # if hasattr(train_ds, 'unscale_predictions'):
        #     predictions = train_ds.unscale_predictions(predictions)
        
        # try:
        #     if isinstance(predictions, torch.Tensor):
        #         predictions = predictions.cpu().numpy()
        #     if isinstance(y_scales, torch.Tensor):
        #         y_scales = y_scales.cpu().numpy()
        #         print('yo')
        #         predictions = predictions * y_scales
        # except Exception as e:
        #     print(f"Error unscaling predictions: {str(e)}")
        #     raise
        
        

        # scrint(f'y_scales are {y_scales}')
        # Convert to DataFrame
        stats_columns = ['points', 'threePointersMade', 'steals', 'blocks', 
                        'turnovers', 'rebounds', 'assists']
        predictions_df = pd.DataFrame(predictions, columns=stats_columns)

        # Ensure non-negative values
        predictions_df = predictions_df.clip(lower=0)
        # print(f'y scales are {y_scales}')
        return predictions_df

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        print(f"Available keys in checkpoint: {checkpoint.keys()}")
        raise




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

def predict_matchup(model_path, game_id, df_all, out_players, m_players, config, n_games=1):
    """
    Predict player performance for a specific matchup.
    
    Args:
        model_path: Path to trained model
        game_id: Game identifier
        df_all: DataFrame with all player data
        out_players: List of players who are out
        m_players: Number of players to select per team
        config: Configuration dictionary
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
    
    predictions = predict_game_stats(model_path, game_data, config)
    game.loc[:, predictions.columns] = predictions
    
    # Calculate player points
    game = game.apply(player_points, axis=1)
    return game.sort_values('Player Points', ascending=False)

def automate_predictions(model_path, config, n_games, m_players=6):
    """
    Generate predictions for all draft groups.
    
    Args:
        model_path: Path to trained model
        config: Configuration dictionary
        n_games: Number of recent games to consider
        m_players: Number of players to select per team
        
    Returns:
        Dictionary of draft group data with predictions
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
        result = predict_matchup(model_path, game_id, df_all, out_players, m_players, config, n_games)
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
        final_df = merge_df[['Name', 'Position', 'Salary', 'Player Points']].copy()
        
        # Calculate value and store results
        final_df['Value'] = final_df['Player Points'] / final_df['Salary'] * 1000
        draft_group_dic[draft_group] = final_df
        
        # Optimize lineup if valid predictions exist
        final_df = final_df[final_df['Player Points'] != 0].copy()
        if not final_df.empty:
            result = optimize_lineup(final_df)
            results_dic[draft_group] = result
    
    logging.info("Predictions completed")
    return draft_group_dic

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = config['paths']['models']['best_model']
    results = automate_predictions(
        model_path=model_path,
        config=config,
        n_games=config['prediction']['recent_games_window']
    )
    
    # Create output directory if it doesn't exist
    output_dir = Path(config['paths']['predictions']['output'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save and display results for each draft group
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = output_dir / f'predictions_{timestamp}.txt'
    
    print("\n=== DraftKings NBA Predictions ===")
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using model: {model_path}")
    print(f"Looking at last {config['prediction']['recent_games_window']} games\n")
    
    with open(output_file, 'w') as f:
        f.write("=== DraftKings NBA Predictions ===\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Using model: {model_path}\n")
        f.write(f"Looking at last {config['prediction']['recent_games_window']} games\n\n")
        
        for draft_group, df in results.items():
            header = f"\nDraft Group: {draft_group}"
            print(header)
            print("="* len(header))
            
            # Sort by predicted points
            df_sorted = df.sort_values('Player Points', ascending=False)
            
            # Format for display
            display_df = df_sorted.copy()
            display_df['Player Points'] = display_df['Player Points'].round(2)
            display_df['Value'] = display_df['Value'].round(2)
            
            # Display to console
            print(display_df.to_string(index=False))
            print(f"\nTotal Salary Used: ${display_df['Salary'].sum():,}")
            print(f"Projected Points: {display_df['Player Points'].sum():.2f}")
            print("\n" + "-"*80 + "\n")
            
            # Write to file
            f.write(header + "\n")
            f.write("="* len(header) + "\n")
            f.write(display_df.to_string(index=False) + "\n")
            f.write(f"\nTotal Salary Used: ${display_df['Salary'].sum():,}\n")
            f.write(f"Projected Points: {display_df['Player Points'].sum():.2f}\n")
            f.write("\n" + "-"*80 + "\n\n")
    
    print(f"\nResults have been saved to: {output_file}")