"""
NBA Game Data Preprocessing

This module handles the preprocessing of NBA game data, including player statistics
calculation, data cleaning, and feature engineering for the prediction model.
"""

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import unicodedata
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import isodate
from dateutil import parser

@dataclass
class PlayerGameInfo:
    """Container for processed player game information."""
    info: Dict
    statistics: Dict

class GameDataProcessor:
    """Handles the preprocessing of NBA game data."""
    
    def __init__(self, player_bios_path: str):
        """
        Initialize the processor with player biographical data.
        
        Args:
            player_bios_path: Path to player biographical information CSV
        """
        self.player_bios = pd.read_csv(player_bios_path, index_col=0)
        
        # Default values for missing player information
        self.default_values = {
            'height': 78,  # average height
            'weight': 215,  # average weight
            'position': '',
            'draft_round': 2,
            'age': 27,
            'years_in_league': 7
        }

    def _remove_accents(self, input_str: str) -> str:
        """Remove diacritics from player names."""
        nfkd_form = unicodedata.normalize('NFD', input_str)
        return ''.join([c for c in nfkd_form if not unicodedata.combining(c)])

    def _parse_minutes(self, minutes: Union[str, float]) -> float:
        """Convert various minutes formats to float."""
        if isinstance(minutes, float):
            return max(minutes, 0)
        
        if isinstance(minutes, str):
            if ':' in minutes:
                mins, secs = minutes.split(':')
                return float(mins) + float(secs)/60
            elif len(minutes) == 0:
                return 0
            else:
                try:
                    duration = isodate.parse_duration(minutes)
                    total_seconds = duration.total_seconds()
                    return total_seconds / 60
                except:
                    return 20  # default minutes if parsing fails
        return 20

    def process_player_data(self, player_data: Dict, game: Dict, location: str) -> Optional[pd.Series]:
        """
        Process individual player data for a game.
        
        Args:
            player_data: Raw player data dictionary
            game: Game data dictionary
            location: 'home' or 'away'
            
        Returns:
            Processed player data as pandas Series
        """
        if 'statistics' not in player_data:
            print(f"Missing statistics for {player_data.get('personId')} in game {game['gameId']}")
            return None
            
        game_date = parser.isoparse(game['gameEt']).replace(tzinfo=None)
        game_date_str = game_date.strftime("%Y-%m-%d")
        
        # Determine team identifiers based on location
        is_home = location == 'home'
        us = 'homeTeam' if is_home else 'awayTeam'
        them = 'awayTeam' if is_home else 'homeTeam'
        
        # Get player bio information
        person_id = player_data['personId']
        player_bio = self._get_player_bio(player_data, game_date)
        
        # Process statistics
        statistics = self._process_statistics(player_data['statistics'])
        
        # Create game information dictionary
        info = {
            'gameId': game['gameId'],
            'teamCode': game[us]['teamTricode'],
            'date': game_date,
            'season': self._get_season(game_date),
            'home?': int(is_home),
            'win?': game[us]['score'] > game[them]['score'],
            'age': player_bio['age'],
            'years_in_league': player_bio['years_in_league'],
            'position': player_bio['position']
        }
        
        # Create final player data
        player_info = {
            'Name': f"{player_data['firstName']} {player_data['familyName']}",
            'personId': person_id,
            'height': player_bio['height'],
            'weight': player_bio['weight'],
            'draft_round': player_bio['draft_round'],
            game_date_str: {'info': info, 'statistics': statistics}
        }
        
        return pd.Series(player_info)

    def _get_player_bio(self, player_data: Dict, game_date: datetime) -> Dict:
        """Get player biographical information."""
        bio = self.default_values.copy()
        person_id = player_data['personId']
        name = self._remove_accents(f"{player_data['firstName']} {player_data['familyName']}")
        
        if person_id in self.player_bios.index:
            bio_data = self.player_bios.loc[person_id]
            bio.update(self._extract_bio_data(bio_data, game_date))
        else:
            try:
                bio_data = self.player_bios[
                    self.player_bios['display_first_last'] == name
                ].iloc[0]
                bio.update(self._extract_bio_data(bio_data, game_date))
            except (IndexError, KeyError):
                pass
                
        return bio

    def _extract_bio_data(self, bio_data: pd.Series, game_date: datetime) -> Dict:
        """Extract and process biographical data."""
        bio = {}
        
        # Process position
        if pd.notna(bio_data.get('position')):
            bio['position'] = bio_data['position']
            # Update height/weight based on position
            bio.update(self._get_position_metrics(bio_data['position']))
            
        # Process height
        if pd.notna(bio_data.get('height')):
            height = bio_data['height'].split('-')
            bio['height'] = int(height[0])*12 + int(height[1])
            
        # Process other numerical fields
        for field in ['weight', 'draft_round']:
            if pd.notna(bio_data.get(field)):
                try:
                    bio[field] = float(bio_data[field])
                except ValueError:
                    pass
                    
        # Calculate age and experience
        if pd.notna(bio_data.get('birthdate')):
            birthday = parser.isoparse(bio_data['birthdate']).replace(tzinfo=None)
            bio['age'] = (game_date - birthday).days/365.25
            
        if pd.notna(bio_data.get('from_year')):
            bio['years_in_league'] = game_date.year - int(bio_data['from_year'])
            
        return bio

    def _get_position_metrics(self, position: str) -> Dict[str, int]:
        """Get default metrics based on player position."""
        position_metrics = {
            'Forward': {'height': 80, 'weight': 225},
            'F': {'height': 80, 'weight': 225},
            'Center': {'height': 83, 'weight': 250},
            'C': {'height': 83, 'weight': 250},
            'Guard': {'height': 76, 'weight': 200},
            'G': {'height': 76, 'weight': 200},
            'Forward-Guard': {'height': 78, 'weight': 213},
            'Guard-Forward': {'height': 78, 'weight': 213},
            'F-G': {'height': 78, 'weight': 213},
            'G-F': {'height': 78, 'weight': 213},
            'Center-Forward': {'height': 82, 'weight': 238},
            'Forward-Center': {'height': 82, 'weight': 238},
            'C-F': {'height': 82, 'weight': 238},
            'F-C': {'height': 82, 'weight': 238}
        }
        return position_metrics.get(position, {'height': 78, 'weight': 215})

    def _process_statistics(self, stats: Dict) -> Dict:
        """Process and clean player game statistics."""
        # Convert minutes to float
        stats['minutes'] = self._parse_minutes(stats['minutes'])
        
        # Remove percentage statistics
        keys_to_remove = [
            'fieldGoalsPercentage',
            'threePointersPercentage',
            'freeThrowsPercentage',
            'reboundsTotal'
        ]
        
        return {k: float(v) for k, v in stats.items() if k not in keys_to_remove}

    def _get_season(self, game_date: datetime) -> int:
        """Determine NBA season for a given date."""
        return game_date.year - 1 if game_date.month <= 8 else game_date.year

    def process_game(self, game: Dict, player_bios: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process all player data for a single game.
        
        Args:
            game: Game data dictionary
            player_bios: Player biographical information
            
        Returns:
            DataFrame containing processed game data
        """
        # Check for required data
        for team in ['homeTeam', 'awayTeam']:
            if 'players' not in game[team]:
                return None
                
        home_players = game['homeTeam']['players']
        away_players = game['awayTeam']['players']
        
        if not home_players or not away_players:
            return None
            
        game_date = parser.isoparse(game['gameEt'])
        game_date = game_date.replace(tzinfo=None)
        game_date_str = game_date.strftime("%Y-%m-%d")
        
        # Process each player
        player_data = []
        for player in home_players:
            processed = self.process_player_data(player, game, 'home')
            if processed is not None:
                player_data.append(processed)
                
        for player in away_players:
            processed = self.process_player_data(player, game, 'away')
            if processed is not None:
                player_data.append(processed)
                
        # Create DataFrame
        if player_data:
            df = pd.DataFrame(player_data)
            return df.dropna(how='all')
        return None

def main():
    """Main processing function."""
    # Load data
    with open("data/raw/total_bball_games.pkl", 'rb') as f:
        games = pickle.load(f)
    
    player_bios = pd.read_csv("data/raw/common_player_info.csv", index_col=0)
    processor = GameDataProcessor(player_bios)
    
    # Process games
    processed_games = []
    for game in games:
        processed = processor.process_game(game, player_bios)
        if processed is not None:
            processed_games.append(processed)
            
    # Save processed data
    with open("data/processed/processed_games.pkl", 'wb') as f:
        pickle.dump(processed_games, f)

if __name__ == "__main__":
    main()