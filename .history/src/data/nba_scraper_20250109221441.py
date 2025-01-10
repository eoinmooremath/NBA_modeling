"""
NBA Data Scraper

This module handles the scraping of NBA game data from the official NBA website.
It provides functionality to collect game data, process it, and save it in a structured format.
"""

import requests
import json
from bs4 import BeautifulSoup
import numpy as np
from time import sleep
import pickle
import os
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path

class NbaScraper:
    """A class to handle NBA game data scraping operations."""
    
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Referer": "https://www.nba.com",
            "Upgrade-Insecure-Requests": "1",
        }
        
        # Create necessary directories
        self.data_dir = Path("data")
        self.pickled_games_dir = self.data_dir / "raw" / "pickled_games"
        self.pickled_games_dir.mkdir(parents=True, exist_ok=True)

    def make_blocks(self, games: List[int], block_length: int) -> List[List[int]]:
        """Split games into blocks of specified length."""
        blocks = []
        k = 0
        while k < len(games):
            blocks.append(games[k:k+block_length])
            k += block_length
        return blocks

    def scrape_game_data(self, game_ids: List[int]) -> Tuple[List[Dict], List[int]]:
        """
        Scrape game data for the specified game IDs.
        
        Args:
            game_ids: List of game IDs to scrape
            
        Returns:
            Tuple containing:
            - List of game data dictionaries
            - List of successfully found game IDs
        """
        block_games = []
        found_games = []
        
        for game_id in game_ids:
            print(f"Trying game {game_id}")
            url = f"https://www.nba.com/game/00{str(game_id)}"
            
            try:
                response = requests.get(url, headers=self.headers)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    script_tag = soup.find("script", type="application/json")
                    
                    if script_tag:
                        json_data = json.loads(script_tag.string)
                        if "game" in json_data['props']['pageProps']:
                            game = json_data['props']['pageProps']['game']
                            block_games.append(game)
                            found_games.append(game_id)
                            print(f"Got game {game_id} info.")
                        else:
                            print("Didn't get game data.")
                else:
                    print(f"Status code error {response.status_code} for game {game_id}.")
            except Exception as e:
                print(f"Problem with game {game_id}: {e}")
                
            sleep(np.random.rand() * 2)  # Random delay to avoid rate-limiting
            
        return block_games, found_games

    def save_block(self, block: List[Dict], block_index: int, attempt: int) -> str:
        """Save a block of game data to a pickle file."""
        block_filename = self.pickled_games_dir / f"block_{block_index}_{attempt}.pkl"
        
        with open(block_filename, "wb") as f:
            pickle.dump(block, f)
            print(f"Pickled block {block_index}_{attempt}")
            
        return str(block_filename)

    def combine_pickled_games(self, output_file: str) -> None:
        """Combine all pickled game blocks into a single file."""
        final_list = []
        
        for filename in self.pickled_games_dir.glob("*.pkl"):
            if filename.is_file():
                with open(filename, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, list):
                        final_list.extend(data)
                    else:
                        print(f"Warning: {filename} does not contain a list.")
                        
        with open(self.data_dir / "processed" / output_file, "wb") as f:
            pickle.dump(final_list, f)
            
        print(f"All games combined and saved in {output_file}")

    def run_scraping(self, games_csv_path: str, block_length: int = 50) -> None:
        """
        Run the complete scraping process.
        
        Args:
            games_csv_path: Path to CSV file containing game IDs
            block_length: Number of games to process in each block
        """
        # Load game data
        games = pd.read_csv(games_csv_path)
        unfound_games = games["game_id"].tolist()
        
        attempt = 0
        found_games = []

        # Scrape game data in blocks
        while len(unfound_games) > 0:
            blocks = self.make_blocks(unfound_games, block_length)
            for index, block in enumerate(blocks):
                print(f"Block index is {index}")
                block_games, block_found_games = self.scrape_game_data(block)
                self.save_block(block_games, index, attempt)
                found_games.extend(block_found_games)
            unfound_games = list(set(unfound_games) - set(found_games))
            attempt += 1

        # Combine all pickled game data
        self.combine_pickled_games("total_bball_games.pkl")

if __name__ == "__main__":
    scraper = NbaScraper()
    scraper.run_scraping("data/raw/games.csv")