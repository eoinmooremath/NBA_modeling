"""
Utilities for making NBA fantasy predictions with trained SAINT model.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from pulp import *

def prepare_game_data(game_df, config):
    """Prepare game data for model prediction."""
    cols_input = [
        'points', 'threePointersMade', 'steals', 'blocks', 'turnovers',
        'assists', 'rebounds', 'weight', 'height', 'age', 'years_in_league',
        'home?', 'draft_round', 'position'
    ]
    return game_df.loc[:, cols_input]

def predict_game_stats(model, game_data, device):
    """
    Predict player statistics for a game.
    
    Args:
        model: Trained SAINT model
        game_data: Prepared game data DataFrame
        device: Computing device
    
    Returns:
        DataFrame with predicted stats
    """
    model.eval()
    with torch.no_grad():
        # Convert data to model input format
        # [Your existing prediction code]
        pass

def calculate_player_points(stats_df):
    """
    Calculate DraftKings fantasy points from player statistics.
    
    Points scoring:
    - Points: 1 pt
    - 3-pt made: 0.5 pt
    - Rebounds: 1.25 pts
    - Assists: 1.5 pts
    - Steals: 2 pts
    - Blocks: 2 pts
    - Turnovers: -0.5 pts
    - Double-double: 1.5 pts
    - Triple-double: 3 pts
    """
    # Base points
    points = (
        stats_df['points'] * 1.0 +
        stats_df['threePointersMade'] * 0.5 +
        stats_df['rebounds'] * 1.25 +
        stats_df['assists'] * 1.5 +
        stats_df['steals'] * 2.0 +
        stats_df['blocks'] * 2.0 +
        stats_df['turnovers'] * -0.5
    )
    
    # Bonus points for double/triple doubles
    categories = ['points', 'rebounds', 'assists', 'blocks', 'steals']
    doubles = (stats_df[categories] >= 10).sum(axis=1)
    
    double_double_bonus = (doubles >= 2) * 1.5
    triple_double_bonus = (doubles >= 3) * 3.0
    
    return points + double_double_bonus + triple_double_bonus

def optimize_lineup(predictions_df, salary_cap=50000):
    """
    Optimize DraftKings lineup based on predictions.
    
    Args:
        predictions_df: DataFrame with predictions and salaries
        salary_cap: Maximum total salary allowed
    
    Returns:
        DataFrame with optimized lineup
    """

    
    # Setup optimization problem
    prob = LpProblem("FantasyTeamOptimization", LpMaximize)
    
    # Create binary decision variables for each position slot
    # We'll create separate variables for each position slot to ensure uniqueness
    positions = ['PG', 'SG', 'SF', 'PF', 'C', 'G', 'F', 'UTIL']
    player_vars = {}
    for pos in positions:
        player_vars[pos] = LpVariable.dicts(f"select_{pos}",
                                          ((i) for i in draft_group.index),
                                          cat='Binary')
    
    # Objective: Maximize total fantasy points
    prob += lpSum([player_vars[pos][i] * draft_group.loc[i, 'Player Points']
                   for pos in positions
                   for i in draft_group.index])
    
    # Constraint 1: Salary cap across all positions
    prob += lpSum([player_vars[pos][i] * draft_group.loc[i, 'Salary']
                   for pos in positions
                   for i in draft_group.index]) <= 50000
    
    # Get indices for each position type
    pg_indices = draft_group[draft_group['Position'].str.lower().str.contains('pg')].index
    sg_indices = draft_group[draft_group['Position'].str.lower().str.contains('sg')].index
    sf_indices = draft_group[draft_group['Position'].str.lower().str.contains('sf')].index
    pf_indices = draft_group[draft_group['Position'].str.lower().str.contains('pf')].index
    c_indices = draft_group[draft_group['Position'].str.lower().str.contains('c')].index
    g_indices = draft_group[draft_group['Position'].str.lower().str.contains('g')].index
    f_indices = draft_group[draft_group['Position'].str.lower().str.contains('f')].index
    all_indices = draft_group.index
    
    # Constraint 2: Position eligibility constraints
    # PG slot can only be filled by PG
    for i in draft_group.index:
        if i not in pg_indices:
            prob += player_vars['PG'][i] == 0
            
    # SG slot can only be filled by SG
    for i in draft_group.index:
        if i not in sg_indices:
            prob += player_vars['SG'][i] == 0
            
    # SF slot can only be filled by SF
    for i in draft_group.index:
        if i not in sf_indices:
            prob += player_vars['SF'][i] == 0
            
    # PF slot can only be filled by PF
    for i in draft_group.index:
        if i not in pf_indices:
            prob += player_vars['PF'][i] == 0
            
    # C slot can only be filled by C
    for i in draft_group.index:
        if i not in c_indices:
            prob += player_vars['C'][i] == 0
            
    # G slot can only be filled by guards
    for i in draft_group.index:
        if i not in g_indices:
            prob += player_vars['G'][i] == 0
            
    # F slot can only be filled by forwards
    for i in draft_group.index:
        if i not in f_indices:
            prob += player_vars['F'][i] == 0
    
    # Constraint 3: Exactly one player per position slot
    for pos in positions:
        prob += lpSum(player_vars[pos][i] for i in draft_group.index) == 1
    
    # Constraint 4: Each player can only be selected once across all positions
    for i in draft_group.index:
        prob += lpSum(player_vars[pos][i] for pos in positions) <= 1
    
    # Solve the problem
    # print("Starting optimization...")
    prob.solve(PULP_CBC_CMD(msg=False))
    
    # Get solution
    selected_indices = []
    selected_positions = []
    total_points = 0
    total_salary = 0
    
    # Collect selected players by position slot
    for pos in positions:
        for i in draft_group.index:
            if value(player_vars[pos][i]) > 0.5:
                selected_indices.append(i)
                selected_positions.append(pos)
                total_points += draft_group.loc[i, 'Player Points']
                total_salary += draft_group.loc[i, 'Salary']
        
    # Get the selected players dataframe
    selected_df = draft_group.loc[selected_indices].copy()
    # Add the position slot information
    selected_df['Roster_Slot'] = selected_positions
    
    # Sort by our position order
    position_order = {pos: i for i, pos in enumerate(positions)}
    selected_df = selected_df.sort_values('Roster_Slot', 
                                        key=lambda x: x.map(position_order))
    
    if len(selected_df) != 8:
        print("\nWARNING: Solution does not contain exactly 8 players!")
        
    return selected_df


def predict_game_stats(model_path, game_data, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get model configuration
        model_config = checkpoint.get('model_config', {})
        if not model_config:
            # If model_config isn't found, try to reconstruct from saved parameters
            model_config = {
                'categories': checkpoint.get('cat_dims', []),
                'num_continuous': checkpoint.get('num_continuous', len([col for col in game_data.columns 
                                               if col not in ['home?', 'draft_round', 'position']])),
                'dim': 32,  # Default values
                'depth': 6,
                'heads': 8,
                'output_dim': 7,  # 7 statistics to predict
                'num_players': 12,  # Default number of players
                'dim_head': 64,
                'mlp_hidden_mults': (4, 2),
                'cont_embeddings': 'MLP',
                'attn_dropout': 0.0,  # Set to 0 for inference
                'ff_dropout': 0.0,
                'mlp_dropout': 0.0
            }
            print("Warning: Using default model configuration")

        # Initialize model
        model = SAINT(
            categories=model_config['categories'],
            num_continuous=model_config['num_continuous'],
            dim=model_config['dim'],
            depth=model_config['depth'],
            heads=model_config['heads'],
            output_dim=model_config['output_dim'],
            num_players=model_config['num_players'],
            dim_head=model_config['dim_head'],
            mlp_hidden_mults=model_config['mlp_hidden_mults'],
            cont_embeddings=model_config['cont_embeddings'],
            attn_dropout=0.0,  # Set to 0 for inference
            ff_dropout=0.0,
            mlp_dropout=0.0
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

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

