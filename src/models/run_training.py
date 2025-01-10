"""
Main training script for NBA player performance prediction model.
"""

import os
import sys
# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

import pickle
import torch
import numpy as np
import yaml
from pathlib import Path
from datetime import datetime

from src.data.data_prep import GameDataset, custom_train_test_split, load_and_process_data
from src.models.saint import SAINT
from src.models.training_utils import train_model

torch.cuda.empty_cache()


def calculate_balanced_weights(y_data):
    """Calculate weights using multiple statistics."""
    means = np.abs(y_data).mean(axis=0)
    stds = np.std(y_data, axis=0)
    p95 = np.percentile(np.abs(y_data), 95, axis=0)
    
    weights_from_means = means.max() / means
    weights_from_stds = stds.max() / stds
    weights_from_p95 = p95.max() / p95
    
    combined_weights = np.power(
        weights_from_means * weights_from_stds * weights_from_p95,
        1/3
    )
    
    return torch.tensor(combined_weights / combined_weights.mean(), dtype=torch.float32)

def evaluate_model(model, valid_dataset, device, batch_size=360):
    """Evaluate model on validation dataset."""
    model.eval()
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for x_categ, x_cont, y_true in valid_loader:
            x_categ = x_categ.to(device)
            x_cont = x_cont.to(device)
            y_true = y_true.to(device)
            
            y_pred = model(x_categ, x_cont)
            
            all_predictions.append(y_pred.cpu())
            all_targets.append(y_true.cpu())
    
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    if hasattr(valid_dataset, 'unscale_predictions'):
        all_predictions = torch.tensor(valid_dataset.unscale_predictions(all_predictions))
        all_targets = torch.tensor(valid_dataset.unscale_predictions(all_targets))
    
    errors = (all_predictions - all_targets).flatten()
    metrics = {
        'mean_error': float(torch.mean(errors)),
        'mae': float(torch.mean(torch.abs(errors))),
        'rmse': float(torch.sqrt(torch.mean(errors**2)))
    }
    
    return metrics

def main():
    # Load configuration
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up CUDA if available
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        torch.cuda.empty_cache()
        print("CUDA optimizations enabled")
        print(f"GPU Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # Load and process data
    print("\nLoading and processing data...")
    data_path = config['paths']['data']['processed'] + '/box_scores.pkl'
    num_players = config['model']['num_players']

    X, y_data, threes_games = load_and_process_data(data_path, num_players)
    
    split_data = custom_train_test_split(
        X, y_data,
        datasplit=config['model']['training']['splits'],
        num_players=config['model']['num_players']
    )
    cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, y_dim = split_data
    
    # Create datasets
    print("\nCreating datasets...")
    train_ds = GameDataset(X_train, y_train, cat_idxs, 
                          num_players=num_players, 
                          continuous_mean_std=(train_mean, train_std))
    valid_ds = GameDataset(X_valid, y_valid, cat_idxs, 
                          num_players=num_players, 
                          continuous_mean_std=(train_mean, train_std))
    test_ds = GameDataset(X_test, y_test, cat_idxs, 
                         num_players=num_players, 
                         continuous_mean_std=(train_mean, train_std))
    
    # Initialize model with full configuration
    print("\nInitializing model...")
    saint_config = config['model']['saint']
    model = SAINT(
        categories=cat_dims,
        num_continuous=len(con_idxs),
        dim=saint_config['dim'],
        depth=saint_config['depth'],
        num_heads=saint_config['num_heads'],
        output_dim=saint_config['output_dim'],
        num_players=num_players,
        head_dim=saint_config['head_dim'],
        mlp_hidden_factors=saint_config['mlp_hidden_factors'],
        continuous_embedding_type=saint_config['continuous_embedding_type'],
        attention_dropout=saint_config['train_dropout']['attention'],  
        ffn_dropout=saint_config['train_dropout']['ffn'],             
        mlp_dropout=saint_config['train_dropout']['mlp']             
    ).to(device)
    
    # Calculate weights
    column_weights = calculate_balanced_weights(y_train['data'])
    
    # Train model
    print("\nStarting training...")
    best_model_state, best_loss, save_path = train_model(
        model=model,
        train_dataset=train_ds,
        valid_dataset=valid_ds,
        device=device,
        column_weights=column_weights.to(device),
        save_dir=config['paths']['models']['checkpoints'],
        batch_stages=config['model']['training']['batch_stages'],
        num_epochs= config['model']['training']['num_epochs'],
        initial_learning_rate=config['model']['training']['learning_rate']['initial'],
        min_learning_rate=config['model']['training']['learning_rate']['minimum'],
        weight_decay=config['model']['training']['weight_decay'],
        gradient_clip=config['model']['training']['gradient_clip'],
        num_workers=config['model']['training']['num_workers']
    )
    
    # Save complete model state with all necessary information
    print("\nSaving final model...")
    torch.save({
        'model_state_dict': best_model_state,
        'cat_dims': cat_dims,
        'cat_idxs': cat_idxs,
        'con_idxs': con_idxs,
        'train_mean': train_mean,
        'train_std': train_std,
        'y_scales': train_ds.y_scales,  # Save the y_scales for prediction
        'config': {                     # Save the modified config structure
            'dim': saint_config['dim'],
            'depth': saint_config['depth'],
            'num_heads': saint_config['num_heads'],
            'output_dim': saint_config['output_dim'],
            'head_dim': saint_config['head_dim'],
            'mlp_hidden_factors': saint_config['mlp_hidden_factors'],
            'continuous_embedding_type': saint_config['continuous_embedding_type'],
            'train_dropout': saint_config['train_dropout'],
            'inference_dropout': saint_config['inference_dropout']
    }  # Save the full model config
    }, save_path)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = evaluate_model(model, valid_ds, device)
    
    print("\nValidation Metrics:")
    print(f"Mean Error: {metrics['mean_error']:.4f}")
    print(f"MAE: {metrics['mae']:.4f}")
    print(f"RMSE: {metrics['rmse']:.4f}")
    
    return save_path, metrics

if __name__ == "__main__":
    main()