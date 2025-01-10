"""
NBA Game Dataset Preparation Module

This module provides the GameDataset class and utilities for preparing NBA game data
for model training. It handles data loading, scaling, and batching for both categorical
and continuous features.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
import yaml
import pickle

class GameDataset(Dataset):
    """Dataset class for NBA game data handling categorical and continuous features."""
    
    def __init__(self, X, Y, cat_cols, num_players=10, continuous_mean_std=None):
        """
        Initialize dataset with categorical and continuous features.
        
        Args:
            X (dict): Dictionary containing 'data' and 'mask' arrays
            Y (dict): Dictionary containing target data
            cat_cols (list): Indices of categorical columns
            num_players (int): Number of players per game
            continuous_mean_std (tuple): (mean, std) for continuous feature normalization
        """
        super().__init__()
        
        # Load configuration
        with open('configs/config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
        
        if continuous_mean_std is not None:
            self.train_mean, self.train_std = continuous_mean_std
        
        # Store categorical dimensions
        self.cat_dims = [len(np.unique(X['data'][:, i])) for i in cat_cols]

        try:
            # Convert inputs to numpy arrays if needed
            X_data = np.asarray(X['data'])
            Y_data = np.asarray(Y['data'])
            
            # Input validation
            if len(X_data) != len(Y_data):
                raise ValueError(f"X length ({len(X_data)}) does not match Y length ({len(Y_data)})")
            
            if len(X_data) % num_players != 0:
                # Truncate to nearest complete game
                num_complete_games = len(X_data) // num_players
                samples_to_keep = num_complete_games * num_players
                X_data = X_data[:samples_to_keep]
                Y_data = Y_data[:samples_to_keep]
            
            # Get continuous column indices
            cat_cols = list(cat_cols)
            con_cols = list(set(range(X_data.shape[1])) - set(cat_cols))
            
            # Calculate number of games
            self.num_games = len(X_data) // num_players
            self.num_players = num_players
            
            try:
                # Convert and reshape categorical data
                x_cat = X_data[:, cat_cols].copy()
                self.X_categorical = torch.tensor(x_cat, dtype=torch.long)
                self.X_categorical = self.X_categorical.reshape(self.num_games, num_players, -1)
                
                # Convert and reshape continuous data
                x_cont = X_data[:, con_cols].copy()
                self.X_continuous = torch.tensor(x_cont, dtype=torch.float32)
                self.X_continuous = self.X_continuous.reshape(self.num_games, num_players, -1)
                
                # Process Y data
                self.y_scales = np.std(Y_data, axis=0)
                self.y_scales = np.where(self.y_scales < 1e-8, 1.0, self.y_scales)
                
                # Scale Y data
                scaled_y = Y_data / self.y_scales
                self.y = torch.tensor(scaled_y, dtype=torch.float32)
                self.y = self.y.reshape(self.num_games, num_players, -1)
                
            except Exception as e:
                print(f"Error during data reshaping: {str(e)}")
                raise
            
            # Apply normalization to continuous features if provided
            if continuous_mean_std is not None:
                mean = torch.tensor(mean, dtype=torch.float32)
                std = torch.tensor(std, dtype=torch.float32)
                std = torch.where(std < 1e-8, torch.ones_like(std), std)
                self.X_continuous = (self.X_continuous - mean) / std
            
            # Move tensors to pinned memory for faster GPU transfer
            if torch.cuda.is_available():
                self.X_categorical = self.X_categorical.pin_memory()
                self.X_continuous = self.X_continuous.pin_memory()
                self.y = self.y.pin_memory()
            
        except Exception as e:
            print(f"Error during dataset initialization: {str(e)}")
            raise
    
    def __len__(self):
        """Return number of games."""
        return self.num_games
    
    def __getitem__(self, idx):
        """Get data for a single game."""
        try:
            if not 0 <= idx < self.num_games:
                raise IndexError(f"Index {idx} out of bounds for dataset with {self.num_games} games")
                
            return (
                self.X_categorical[idx],
                self.X_continuous[idx],
                self.y[idx]
            )
        except Exception as e:
            print(f"Error accessing game index {idx}: {str(e)}")
            raise
    
    def unscale_predictions(self, y_pred):
        """Convert scaled predictions back to original scale."""
        try:
            if isinstance(y_pred, torch.Tensor):
                y_pred = y_pred.cpu().numpy()
            return y_pred * self.y_scales
        except Exception as e:
            print(f"Error unscaling predictions: {str(e)}")
            raise

def custom_train_test_split(X, y, datasplit=[.8, .1, .1], num_players=12, test_seed=1, train_seed=None):
    """
    Custom train-test split that maintains game integrity.
    
    Args:
        X: Input features DataFrame
        y: Target values
        datasplit: List of train/valid/test split ratios
        num_players: Number of players per game
        test_seed: Random seed for test set split
        train_seed: Random seed for train set split
    """
    X = X.copy()
    y = y.copy()
    
    # Verify input data is properly structured
    assert len(X) % num_players == 0, f"Input X length {len(X)} is not divisible by {num_players}"
    assert len(y) % num_players == 0, f"Input y length {len(y)} is not divisible by {num_players}"
    
    # First identify categorical and continuous columns
    categorical_indicator = np.array(X.dtypes=='category')
    categorical_columns = X.columns[list(np.where(categorical_indicator)[0])].tolist()
    cont_columns = list(set(X.columns.tolist()) - set(categorical_columns))

    cat_idxs = list(np.where(categorical_indicator)[0])
    con_idxs = list(set(range(len(X.columns))) - set(cat_idxs))
    
    # Convert continuous columns to float
    for col in cont_columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Split data into games first
    num_games = len(X) // num_players
    game_indices = np.arange(num_games)

    rng_state = np.random.get_state()  # Save current random state before setting it to test_seed
    np.random.seed(test_seed)
    np.random.shuffle(game_indices)
    
    # Calculate split sizes in terms of games
    test_games = int(num_games * datasplit[2])
    remaining_games = num_games - test_games
    
    # Select test indices deterministically
    test_game_idx = game_indices[-test_games:]
    remaining_game_idx = game_indices[:-test_games]
    
    if train_seed is not None:
        np.random.seed(train_seed)
    else:
        np.random.set_state(rng_state)  # Restore original random state if no train_seed
    np.random.shuffle(remaining_game_idx)

    train_ratio = datasplit[0] / (datasplit[0] + datasplit[1])
    train_games = int(remaining_games * train_ratio)
    valid_games = remaining_games - train_games
   
    # Split game indices
    # Split remaining indices into train and validation
    train_game_idx = remaining_game_idx[:train_games]
    valid_game_idx = remaining_game_idx[train_games:]
    
    # Convert game indices to player indices
    def games_to_players(game_idx):
        player_idx = []
        for g in game_idx:
            start_idx = g * num_players
            player_idx.extend(range(start_idx, start_idx + num_players))
        return np.array(player_idx)
    
    train_indices = games_to_players(train_game_idx)
    valid_indices = games_to_players(valid_game_idx)
    test_indices = games_to_players(test_game_idx)
    
    # Create nan_mask
    nan_mask = pd.DataFrame(np.ones_like(X.values), columns=X.columns)
    for col in cont_columns:
        nan_mask[col] = ~X[col].isna()
    
    # Process categorical columns
    cat_dims = []
    for col in categorical_columns:
        X[col] = X[col].astype(str)
        l_enc = LabelEncoder()
        X[col] = l_enc.fit_transform(X[col])
        cat_dims.append(len(l_enc.classes_))
        nan_mask[col] = 1
    
    # Process continuous columns
    for col in cont_columns:
        train_mean = X.loc[train_indices, col].mean()
        X[col] = X[col].fillna(train_mean)

    
    # Split data
    X_train = {'data': X.iloc[train_indices].values, 'mask': nan_mask.iloc[train_indices].values}
    X_valid = {'data': X.iloc[valid_indices].values, 'mask': nan_mask.iloc[valid_indices].values}
    X_test = {'data': X.iloc[test_indices].values, 'mask': nan_mask.iloc[test_indices].values}
    
    y_train = {'data': y[train_indices]}
    y_valid = {'data': y[valid_indices]}
    y_test = {'data': y[test_indices]}
    
    # Calculate mean and std for continuous features
    train_mean = np.array(X_train['data'][:, con_idxs], dtype=np.float32).mean(0)
    train_std = np.array(X_train['data'][:, con_idxs], dtype=np.float32).std(0)
    train_std = np.where(train_std < 1e-6, 1e-6, train_std)
    
    
    y_dim = y.shape[1] if len(y.shape) > 1 else 1
    
    return cat_dims, cat_idxs, con_idxs, X_train, y_train, X_valid, y_valid, X_test, y_test, train_mean, train_std, y_dim


def load_and_process_data(data_path, n_players=6):
    with open(data_path, 'rb') as file:
        box_scores = pickle.load(file)
    
    box_scores = [box for box in box_scores if all(len(v)==16 for v in box.values())]
    check_keys = [key for key in box_scores[0].keys() if 'detailed' in key]
    threes_games = [box for box in box_scores if not(np.all([np.all(box[key][:,4]==0) for key in check_keys]))]
    
    def process_box_scores(boxes, n_players):
        X_boxes = [np.concatenate([
            np.concatenate([box['last_1 general'][:n_players,:], box['bio'][:n_players,:]], axis=1),
            np.concatenate([box['last_1 general'][8:8+n_players,:], box['bio'][8:8+n_players,:]], axis=1)
        ], axis=0) for box in boxes]
        
        y_boxes = [np.concatenate([
            box['today general'][:n_players,:],
            box['today general'][8:8+n_players,:]
        ], axis=0) for box in boxes]
        
        return np.vstack(X_boxes), np.vstack(y_boxes).astype('float32')
    
    X_data, y_data = process_box_scores(threes_games, n_players)
    
    cols_general = ['points', 'threePointersMade', 'steals', 'blocks', 'turnovers', 'assists', 'rebounds']
    cols_bio = ['weight', 'height', 'age', 'years_in_league', 'home?', 'draft_round', 'position']
    columns = cols_general + cols_bio
    
    X = pd.DataFrame(X_data, columns=columns)
    X['draft_round'] = X['draft_round'].astype(float).clip(upper=3)
    
    categorical_cols = ['home?', 'draft_round', 'position']
    numeric_cols = [col for col in X.columns if col not in categorical_cols]
    
    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce').astype('float32')
    for col in categorical_cols:
        X[col] = X[col].astype('category')
    
    def shuffle_players(X, y_data, n_players):
        n_games = len(X) // n_players
        idx = np.random.permutation(n_players)
        for d in range(1, n_games):
            perm = np.random.permutation(np.arange(n_players*d, n_players*(d+1)))
            idx = np.concatenate([idx, perm])
        return X.iloc[idx,:], y_data[idx,:]
    
    X, y_data = shuffle_players(X, y_data, n_players)
    return X, y_data, threes_games



