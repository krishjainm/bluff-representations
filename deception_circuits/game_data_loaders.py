"""
Game data loaders for deception circuit research.

This module provides data loaders for game-based deception scenarios:
- Mafia (Mafiascum dataset)
- Bullshit (BS card game)
- Poker (structured game data)

These loaders provide the structure for collecting and processing game data.
Users will need to run the actual games to collect data.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import json
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MafiaGameData:
    """Structure for Mafia game data."""
    game_id: str
    players: List[str]
    roles: Dict[str, str]  # player -> role (mafia, town, etc.)
    rounds: List[Dict]  # List of round data
    model_was_mafia: bool
    model_won: bool
    deception_instances: List[Dict]  # Instances where model was deceptive
    final_outcome: str


@dataclass
class BullshitGameData:
    """Structure for Bullshit card game data."""
    game_id: str
    players: List[str]
    cards_played: List[Dict]  # Cards played in each turn
    bluffs: List[Dict]  # Bluff instances
    model_bluffed: bool
    model_won: bool
    deception_instances: List[Dict]


@dataclass
class PokerGameData:
    """Structure for poker game data."""
    game_id: str
    players: List[str]
    hands: Dict[str, List[str]]  # player -> cards
    betting_rounds: List[Dict]
    model_hand: List[str]
    model_bluffed: bool
    model_won: bool
    deception_instances: List[Dict]


class MafiaDataLoader:
    """
    Data loader for Mafia game data.
    
    Mafia is a social deduction game where players have hidden roles.
    The game involves deception as mafia players try to hide their identity.
    """
    
    def __init__(self):
        """Initialize Mafia data loader."""
        self.games = []
        
    def load_mafiascum_data(self, data_path: Union[str, Path]) -> List[MafiaGameData]:
        """
        Load Mafiascum dataset.
        
        Args:
            data_path: Path to Mafiascum dataset file
            
        Returns:
            List of MafiaGameData objects
            
        Note:
            This expects data in a specific format. Users will need to
            collect actual game data and format it appropriately.
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.warning(f"Mafiascum data file not found: {data_path}")
            logger.info("You will need to collect Mafia game data and format it.")
            return []
        
        # Load data (format depends on how data is collected)
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            games = []
            for game_data in data:
                game = MafiaGameData(
                    game_id=game_data.get('game_id', ''),
                    players=game_data.get('players', []),
                    roles=game_data.get('roles', {}),
                    rounds=game_data.get('rounds', []),
                    model_was_mafia=game_data.get('model_was_mafia', False),
                    model_won=game_data.get('model_won', False),
                    deception_instances=game_data.get('deception_instances', []),
                    final_outcome=game_data.get('final_outcome', '')
                )
                games.append(game)
            
            self.games = games
            logger.info(f"Loaded {len(games)} Mafia games")
            return games
            
        except Exception as e:
            logger.error(f"Error loading Mafiascum data: {e}")
            return []
    
    def convert_to_deception_dataset(self, games: List[MafiaGameData]) -> pd.DataFrame:
        """
        Convert Mafia game data to deception dataset format.
        
        Args:
            games: List of MafiaGameData objects
            
        Returns:
            DataFrame with deception data
        """
        rows = []
        
        for game in games:
            for instance in game.deception_instances:
                row = {
                    'statement': instance.get('statement', ''),
                    'response': instance.get('response', ''),
                    'label': 1 if instance.get('was_deceptive', False) else 0,
                    'scenario': 'mafia',
                    'game_id': game.game_id,
                    'round': instance.get('round', 0),
                    'model_was_mafia': game.model_was_mafia,
                    'model_won': game.model_won,
                    'metadata': {
                        'role': instance.get('role', ''),
                        'context': instance.get('context', '')
                    }
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_sample_structure(self, output_path: Union[str, Path]):
        """
        Create sample data structure for Mafia games.
        
        This shows users what format the data should be in.
        
        Args:
            output_path: Path to save sample structure
        """
        sample_data = {
            'games': [
                {
                    'game_id': 'mafia_game_001',
                    'players': ['player1', 'player2', 'model', 'player4'],
                    'roles': {
                        'player1': 'town',
                        'player2': 'mafia',
                        'model': 'mafia',
                        'player4': 'town'
                    },
                    'rounds': [
                        {
                            'round_number': 1,
                            'statements': [
                                {
                                    'player': 'model',
                                    'statement': "I'm a town member, I saw player2 acting suspicious",
                                    'was_deceptive': True,
                                    'context': 'model is mafia, trying to frame player2'
                                }
                            ]
                        }
                    ],
                    'model_was_mafia': True,
                    'model_won': True,
                    'deception_instances': [
                        {
                            'statement': "I'm a town member",
                            'response': "I'm a town member, I saw player2 acting suspicious",
                            'was_deceptive': True,
                            'round': 1,
                            'role': 'mafia',
                            'context': 'trying to hide mafia identity'
                        }
                    ],
                    'final_outcome': 'mafia_won'
                }
            ]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Created sample Mafia data structure at {output_path}")


class BullshitDataLoader:
    """
    Data loader for Bullshit (BS) card game data.
    
    Bullshit is a card game where players try to get rid of cards by
    playing them face-down and claiming they're a certain rank. Other
    players can call "bullshit" if they think the player is lying.
    """
    
    def __init__(self):
        """Initialize Bullshit data loader."""
        self.games = []
        
    def load_bullshit_data(self, data_path: Union[str, Path]) -> List[BullshitGameData]:
        """
        Load Bullshit game data.
        
        Args:
            data_path: Path to Bullshit game data file
            
        Returns:
            List of BullshitGameData objects
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.warning(f"Bullshit data file not found: {data_path}")
            logger.info("You will need to collect Bullshit game data and format it.")
            return []
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            games = []
            for game_data in data:
                game = BullshitGameData(
                    game_id=game_data.get('game_id', ''),
                    players=game_data.get('players', []),
                    cards_played=game_data.get('cards_played', []),
                    bluffs=game_data.get('bluffs', []),
                    model_bluffed=game_data.get('model_bluffed', False),
                    model_won=game_data.get('model_won', False),
                    deception_instances=game_data.get('deception_instances', [])
                )
                games.append(game)
            
            self.games = games
            logger.info(f"Loaded {len(games)} Bullshit games")
            return games
            
        except Exception as e:
            logger.error(f"Error loading Bullshit data: {e}")
            return []
    
    def convert_to_deception_dataset(self, games: List[BullshitGameData]) -> pd.DataFrame:
        """
        Convert Bullshit game data to deception dataset format.
        
        Args:
            games: List of BullshitGameData objects
            
        Returns:
            DataFrame with deception data
        """
        rows = []
        
        for game in games:
            for instance in game.deception_instances:
                row = {
                    'statement': instance.get('statement', ''),
                    'response': instance.get('response', ''),
                    'label': 1 if instance.get('was_bluff', False) else 0,
                    'scenario': 'bullshit',
                    'game_id': game.game_id,
                    'cards_claimed': instance.get('cards_claimed', ''),
                    'cards_actual': instance.get('cards_actual', ''),
                    'model_bluffed': instance.get('was_bluff', False),
                    'model_won': game.model_won,
                    'metadata': {
                        'turn': instance.get('turn', 0),
                        'was_called': instance.get('was_called', False)
                    }
                }
                rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_sample_structure(self, output_path: Union[str, Path]):
        """
        Create sample data structure for Bullshit games.
        
        Args:
            output_path: Path to save sample structure
        """
        sample_data = {
            'games': [
                {
                    'game_id': 'bullshit_game_001',
                    'players': ['player1', 'model', 'player3'],
                    'cards_played': [
                        {
                            'turn': 1,
                            'player': 'model',
                            'cards_claimed': 'three 7s',
                            'cards_actual': 'two 7s, one 5',
                            'was_bluff': True,
                            'was_called': False
                        }
                    ],
                    'bluffs': [
                        {
                            'player': 'model',
                            'turn': 1,
                            'statement': "I'm playing three 7s",
                            'response': "I'm playing three 7s",
                            'was_bluff': True
                        }
                    ],
                    'model_bluffed': True,
                    'model_won': True,
                    'deception_instances': [
                        {
                            'statement': "What cards are you playing?",
                            'response': "I'm playing three 7s",
                            'was_bluff': True,
                            'cards_claimed': 'three 7s',
                            'cards_actual': 'two 7s, one 5',
                            'turn': 1,
                            'was_called': False
                        }
                    ]
                }
            ]
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(sample_data, f, indent=2)
        
        logger.info(f"Created sample Bullshit data structure at {output_path}")


class PokerGameDataLoader:
    """
    Data loader for structured poker game data.
    
    Poker involves strategic deception (bluffing) where players
    misrepresent their hand strength.
    """
    
    def __init__(self):
        """Initialize poker data loader."""
        self.games = []
        
    def load_poker_data(self, data_path: Union[str, Path]) -> List[PokerGameData]:
        """
        Load poker game data.
        
        Args:
            data_path: Path to poker game data file
            
        Returns:
            List of PokerGameData objects
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            logger.warning(f"Poker data file not found: {data_path}")
            logger.info("You will need to collect poker game data and format it.")
            return []
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            games = []
            for game_data in data:
                game = PokerGameData(
                    game_id=game_data.get('game_id', ''),
                    players=game_data.get('players', []),
                    hands=game_data.get('hands', {}),
                    betting_rounds=game_data.get('betting_rounds', []),
                    model_hand=game_data.get('model_hand', []),
                    model_bluffed=game_data.get('model_bluffed', False),
                    model_won=game_data.get('model_won', False),
                    deception_instances=game_data.get('deception_instances', [])
                )
                games.append(game)
            
            self.games = games
            logger.info(f"Loaded {len(games)} poker games")
            return games
            
        except Exception as e:
            logger.error(f"Error loading poker data: {e}")
            return []
    
    def convert_to_deception_dataset(self, games: List[PokerGameData]) -> pd.DataFrame:
        """
        Convert poker game data to deception dataset format.
        
        Args:
            games: List of PokerGameData objects
            
        Returns:
            DataFrame with deception data
        """
        rows = []
        
        for game in games:
            for instance in game.deception_instances:
                row = {
                    'statement': instance.get('statement', ''),
                    'response': instance.get('response', ''),
                    'label': 1 if instance.get('was_bluff', False) else 0,
                    'scenario': 'poker',
                    'game_id': game.game_id,
                    'hand_strength': instance.get('hand_strength', 0.0),
                    'model_bluffed': instance.get('was_bluff', False),
                    'model_won': game.model_won,
                    'metadata': {
                        'betting_round': instance.get('betting_round', 0),
                        'action': instance.get('action', '')
                    }
                }
                rows.append(row)
        
        return pd.DataFrame(rows)

