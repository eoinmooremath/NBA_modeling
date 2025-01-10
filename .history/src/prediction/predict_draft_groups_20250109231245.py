"""
Main script for running NBA fantasy predictions with trained model.
"""

import torch
import pandas as pd
from pathlib import Path
import yaml
import logging

from .prediction_utils import (
    prepare_game_data,
    predict_game_stats,
    calculate_player_points,
    optimize_lineup,
    load_model,
    validate_predictions
)
from .draftkings import get_draft_groups
from .nba_stats import get_player_stats

def setup_logger():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    """Run NBA fantasy predictions."""
    setup_logger()
    logger = logging.getLogger(__name__)
    
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        model = load_model(config['paths']['best_model'], device)
        logger.info("Model loaded successfully")
        
        # Get current NBA player stats
        player_stats = get_player_stats()
        logger.info("Player stats retrieved")
        
        # Get DraftKings contests and lineups
        draft_groups = get_draft_groups()
        logger.info(f"Retrieved {len(draft_groups)} draft groups")
        
        results = {}
        for group in draft_groups:
            logger.info(f"Processing draft group: {group['id']}")
            
            # Prepare game data
            game_data = prepare_game_data(player_stats, config)
            
            # Make predictions
            predictions = predict_game_stats(model, game_data, device)
            validate_predictions(predictions)
            
            # Calculate fantasy points
            fantasy_points = calculate_player_points(predictions)
            
            # Optimize lineup
            lineup = optimize_lineup(
                pd.concat([predictions, fantasy_points], axis=1),
                salary_cap=config['draftkings']['max_salary']
            )
            
            results[group['id']] = lineup
            logger.info(f"Completed predictions for group {group['id']}")
        
        # Save results
        output_path = Path(config['paths']['predictions'])
        output_path.mkdir(parents=True, exist_ok=True)
        
        for group_id, lineup in results.items():
            lineup.to_csv(output_path / f'lineup_{group_id}.csv')
        
        logger.info("Predictions completed successfully")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()