"""
Utilities for making NBA fantasy predictions with trained SAINT model.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from pulp import LpProblem, LpVariable, lpSum, value

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

